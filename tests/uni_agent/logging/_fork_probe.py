"""Isolated fork probe: the outer test owns this process and its process group."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import threading
from pathlib import Path

from uni_agent.logging import handlers


class BlockingRawFile(io.RawIOBase):
    """Hold a real BufferedWriter's internal lock during its background flush."""

    def __init__(self, path: Path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self.entered = threading.Event()
        self.release = threading.Event()

    def writable(self):
        return True

    def fileno(self):
        return self.fd

    def write(self, data):
        self.entered.set()
        if not self.release.wait(timeout=20):
            raise RuntimeError("test did not release the controlled flush")
        return os.write(self.fd, data)

    def close(self):
        if not self.closed:
            try:
                os.close(self.fd)
            except OSError:
                pass
        super().close()


def main():
    mode, directory = sys.argv[1:]
    directory = Path(directory)
    dispatch = handlers._dispatch
    dispatch._shutdown()
    handlers._FLUSH_INTERVAL = 0.01
    dispatch._start()
    raw = BlockingRawFile(directory / "parent.log")
    file_obj = io.TextIOWrapper(io.BufferedWriter(raw), encoding="utf-8")
    file_obj.write("parent-record\n")
    dispatch._files["parent"] = file_obj
    dispatch._dirty.add(file_obj)
    assert raw.entered.wait(timeout=5), "writer did not enter the buffered flush"
    # A second buffered log has not been flushed yet. Discarding inherited
    # wrappers must not duplicate its pending parent bytes in any child.
    pending = open(directory / "pending-parent.log", "a", encoding="utf-8")
    pending.write("pending-parent-record\n")
    dispatch._files["pending-parent"] = pending

    fork_started = threading.Event()
    fork_parent_done = threading.Event()
    os.register_at_fork(after_in_parent=fork_parent_done.set)
    results = []
    errors = []

    async def child():
        if mode == "uvloop":
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "print('child-ready')",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            assert process.returncode == 0
            assert stdout == b"child-ready\n" and stderr == b""
        else:
            pid = os.fork()
            if pid == 0:
                # Exercise persistent Python children as well as fork+exec.
                assert not dispatch._files and not dispatch._log_ids
                assert dispatch._thread.is_alive()
                dispatch._submit(("open", "child", str(directory / "child.log")))
                dispatch._submit(("write", "child", "child-record\n"))
                dispatch._shutdown()
                os._exit(0)
            _, status = os.waitpid(pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0
        results.append(True)

    async def run():
        fork_started.set()
        await child()
        await asyncio.gather(*(child() for _ in range(127)))

    def worker():
        try:
            if mode == "uvloop":
                import uvloop

                loop = uvloop.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(run())
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
            else:
                asyncio.run(run())
        except BaseException as exc:
            errors.append(repr(exc))

    thread = threading.Thread(target=worker, name="probe-async-actor")
    thread.start()
    assert fork_started.wait(timeout=5)
    # Before the fix, the parent returns from fork while the writer's buffered
    # lock is held; the child then remains stuck even after this parent flushes.
    # With quiescing hooks, fork waits until release and inherits unlocked I/O.
    forked_while_locked = fork_parent_done.wait(timeout=0.5)
    print(json.dumps({"forked_while_locked": forked_while_locked}), flush=True)
    raw.release.set()
    thread.join(timeout=15)
    assert not thread.is_alive(), "async actor hung in fork/exec or child output drain"
    assert not errors, errors
    assert len(results) == 128
    dispatch._shutdown()
    assert (directory / "parent.log").read_text() == "parent-record\n"
    assert (directory / "pending-parent.log").read_text() == "pending-parent-record\n"
    if mode != "uvloop":
        assert (directory / "child.log").read_text().splitlines() == ["child-record"] * 128
    # Every fork/exec child must have been reaped by waitpid/the event loop.
    try:
        assert os.waitpid(-1, os.WNOHANG)[0] == -1, "an unreaped child remains"
    except ChildProcessError:
        pass
    print(json.dumps({"children": len(results), "reaped": True}), flush=True)


if __name__ == "__main__":
    main()
