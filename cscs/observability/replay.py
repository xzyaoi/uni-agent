"""Replay paired native sandbox and model-call captures into fresh resources."""

import argparse
import asyncio
import base64
import dataclasses
import json
import os
import time
from pathlib import Path

import httpx

from uni_agent.sandbox.agentenv import AgentEnvSandbox


def read_events(path):
    events = []
    with path.open() as stream:
        for line in stream:
            try:
                events.append(json.loads(line))
            except ValueError:
                # A process killed during its final append can leave a partial line.
                continue
    return events


def decode(value):
    if isinstance(value, dict):
        if value.get("encoding") == "base64" and "data" in value:
            return base64.b64decode(value["data"])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def json_value(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode()}
    raise TypeError(type(value).__name__)


async def main(args):
    model_events = {}
    if args.model_dir:
        for path in sorted(args.model_dir.glob("llm-*.jsonl")):
            for event in read_events(path):
                reference = event.get("sandbox_trace")
                if reference and event.get("phase") == "begin":
                    model_events.setdefault(Path(reference).name, []).append(event)
    lanes = []
    for path in sorted(args.native_dir.glob("*.jsonl")):
        events = read_events(path)
        if not args.include_incomplete and not any(
            e.get("operation") == "stop" and e.get("phase") == "end" for e in events
        ):
            continue
        if not any(e.get("operation") == "start" and e.get("phase") == "begin" for e in events):
            continue
        if args.mode in ("gpu", "coupled") and not model_events.get(path.name):
            continue
        lanes.append((path, events))
        if args.limit and len(lanes) >= args.limit:
            break
    if not lanes:
        raise ValueError("No eligible native lifecycles; partial workloads require --include-incomplete")
    if args.mode in ("gpu", "coupled"):
        if not args.llm_base_url:
            raise ValueError("GPU/coupled replay requires --llm-base-url")
        if any(not model_events.get(path.name) for path, _ in lanes):
            raise ValueError("Every selected lane needs correlated incremental model captures")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["AGENTENV_TRACE_DIR"] = str(args.output.parent / (args.output.stem + "-native"))
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stream = os.fdopen(descriptor, "w")
    semaphore = asyncio.Semaphore(args.concurrency or len(lanes))

    def emit(**record):
        stream.write(json.dumps({"timestamp": time.time(), **record}, default=json_value) + "\n")
        stream.flush()

    async def lane(path, events):
        async with semaphore:
            config = next((e for e in events if e.get("operation") == "configure"), {})
            image = config.get("image") or events[0]["image"]
            sandbox = AgentEnvSandbox(
                image=image,
                env_vars=config.get("env_vars"),
                runtime_timeout=config.get("runtime_timeout", 14400),
                exec_timeout=config.get("exec_timeout", 180),
            )
            ends = {e.get("operation_id"): e for e in events if e.get("phase") in ("end", "error")}
            outer_spans = []
            for event in events:
                end = ends.get(event.get("operation_id"))
                if event.get("phase") == "begin" and end and event.get("operation") in ("start", "open_shell", "stop"):
                    outer_spans.append((event["timestamp"], end["timestamp"]))
            actions = [
                e
                for e in events
                if e.get("phase") == "begin" and not any(begin < e["timestamp"] < end for begin, end in outer_spans)
            ]
            actions.extend(model_events.get(path.name, []))
            actions.sort(key=lambda e: e["timestamp"])
            origin = actions[0]["timestamp"]
            started = time.monotonic()
            shells = {}
            async with httpx.AsyncClient(timeout=args.request_timeout) as client:
                try:
                    for event in actions:
                        if args.pace == "recorded":
                            await asyncio.sleep(max(0, event["timestamp"] - origin - (time.monotonic() - started)))
                        operation = event["operation"]
                        arguments = decode(event.get("arguments", {}))
                        tick = time.monotonic()
                        result = None
                        if operation == "chat_completion":
                            if args.mode != "sandbox":
                                body = dict(event["request"])
                                if args.model:
                                    body["model"] = args.model
                                response = await client.post(
                                    args.llm_base_url.rstrip("/") + "/chat/completions",
                                    json=body,
                                    headers={"Authorization": "Bearer " + os.getenv(args.llm_api_key_env, "EMPTY")},
                                )
                                response.raise_for_status()
                                result = response.json()
                        elif args.mode != "gpu":
                            if operation == "start":
                                await sandbox.start()
                            elif operation == "stop":
                                await sandbox.stop()
                            elif operation == "exec":
                                result = await sandbox._exec(**arguments)
                            elif operation == "write_file":
                                await sandbox.write_file(**arguments)
                            elif operation == "read_file":
                                result = await sandbox.read_file(**arguments)
                            elif operation == "open_shell":
                                end = ends.get(event["operation_id"], {})
                                original_id = (end.get("result") or {}).get("shell_id")
                                if not original_id:
                                    raise ValueError("Incomplete shell creation cannot be replayed")
                                shells[original_id] = await sandbox.open_shell(**arguments)
                            elif operation == "shell_run":
                                result = await shells[event["shell_id"]].run(**arguments)
                            elif operation == "shell_close":
                                await shells[event["shell_id"]].close()
                            else:
                                raise ValueError("Unsupported recorded operation: " + operation)
                        emit(
                            lane=path.name,
                            operation=operation,
                            original_operation_id=event.get("operation_id"),
                            duration=time.monotonic() - tick,
                            result=result,
                        )
                except Exception as exc:
                    emit(lane=path.name, error=type(exc).__name__ + ": " + str(exc))
                    return False
                finally:
                    try:
                        await sandbox.stop()
                    except Exception as exc:
                        emit(lane=path.name, cleanup_error=str(exc))
            return True

    try:
        results = await asyncio.gather(*(lane(path, events) for path, events in lanes))
        emit(
            summary={
                "lanes": len(results),
                "successful": sum(results),
                "mode": args.mode,
                "concurrency": args.concurrency,
                "pace": args.pace,
            }
        )
        if not all(results):
            raise RuntimeError("Some replay lanes failed; see incremental replay output")
    finally:
        stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--mode", choices=("sandbox", "gpu", "coupled"), default="sandbox")
    parser.add_argument("--pace", choices=("asap", "recorded"), default="asap")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model")
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.concurrency < 0 or options.limit < 0:
        parser.error("Concurrency and limit must be non-negative; 0 means all selected lanes")
    asyncio.run(main(options))
