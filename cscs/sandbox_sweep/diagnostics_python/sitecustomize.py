"""Opt-in stack capture for Python processes belonging to one diagnostic run."""
import faulthandler
import json
import os
import signal
from pathlib import Path

_stream = None

def _register():
    global _stream
    directory = os.environ.get('SWEEP_DIAGNOSTIC_DIR')
    if not directory:
        return
    try:
        if _stream is not None:
            faulthandler.unregister(signal.SIGUSR1)
            faulthandler.disable()
            _stream.close()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        descriptor = os.open(str(root / ('python-stack-%s.log' % pid)),
                             os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        _stream = os.fdopen(descriptor, 'a')
        faulthandler.enable(file=_stream, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=_stream, all_threads=True)
        fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        (root / ('armed-%s.json' % pid)).write_text(json.dumps({
            'pid': pid, 'start_ticks': fields[19], 'uid': os.getuid()}))
    except Exception as exc:
        # No registration file means the watchdog will never signal this process.
        try:
            os.write(2, ('Stack diagnostics unavailable: %s\n' % type(exc).__name__).encode())
        except OSError:
            pass

_register()
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_register)
