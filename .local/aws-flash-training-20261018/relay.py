#!/usr/bin/env python3
"""Authenticate the first HTTP request, then relay the TCP stream unchanged."""

import argparse
import asyncio
import contextlib
import hmac
import logging
from pathlib import Path


MAX_HEADER = 64 * 1024
HEADER_TIMEOUT = 10
LOG = logging.getLogger("scheduler-relay")


def load_token(path):
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError("token file must not be accessible by group or others (use chmod 600)")
    token = path.read_bytes().strip()
    if not token or b"\r" in token or b"\n" in token:
        raise ValueError("token file must contain one nonempty token")
    return token


def authenticated(header, token):
    """Reject malformed/duplicate credentials; never decode or log secrets."""
    credentials = {}
    for line in header.split(b"\r\n")[1:-2]:
        if line[:1] in (b" ", b"\t") or b":" not in line:
            return False
        name, value = line.split(b":", 1)
        name = name.lower()
        if name not in (b"x-api-key", b"authorization"):
            continue
        if name in credentials:
            return False
        value = value.strip()
        if name == b"authorization":
            parts = value.split(None, 1)
            if len(parts) != 2 or parts[0].lower() != b"bearer":
                return False
            value = parts[1]
        credentials[name] = value
    matches = [hmac.compare_digest(value, token) for value in credentials.values()]
    return bool(matches) and all(matches)


async def respond(writer, status):
    writer.write(
        b"HTTP/1.1 " + status + b"\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
    )
    await writer.drain()


async def copy_stream(reader, writer):
    while data := await reader.read(64 * 1024):
        writer.write(data)
        await writer.drain()
    if writer.can_write_eof():
        writer.write_eof()
        await writer.drain()


async def close_writer(writer):
    writer.close()
    with contextlib.suppress(ConnectionError, OSError):
        await writer.wait_closed()


async def relay_connection(reader, writer, upstream_host, upstream_port, token, upstream_socket=None):
    upstream_writer = None
    pumps = []
    try:
        try:
            header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEADER_TIMEOUT)
        except asyncio.TimeoutError:
            await respond(writer, b"408 Request Timeout")
            return
        except asyncio.LimitOverrunError:
            await respond(writer, b"431 Request Header Fields Too Large")
            return
        except asyncio.IncompleteReadError:
            await respond(writer, b"400 Bad Request")
            return
        if len(header) > MAX_HEADER:
            await respond(writer, b"431 Request Header Fields Too Large")
            return
        if not authenticated(header, token):
            await respond(writer, b"401 Unauthorized")
            return
        try:
            if upstream_socket is not None:
                upstream_reader, upstream_writer = await asyncio.open_unix_connection(upstream_socket)
            else:
                upstream_reader, upstream_writer = await asyncio.open_connection(
                    upstream_host, upstream_port
                )
        except OSError:
            LOG.warning("Upstream connection failed")
            await respond(writer, b"502 Bad Gateway")
            return
        upstream_writer.write(header)
        await upstream_writer.drain()
        pumps = [
            asyncio.create_task(copy_stream(reader, upstream_writer)),
            asyncio.create_task(copy_stream(upstream_reader, writer)),
        ]
        await asyncio.gather(*pumps)
    except (ConnectionError, OSError):
        LOG.debug("Connection closed during forwarding")
    finally:
        for pump in pumps:
            if not pump.done():
                pump.cancel()
        if pumps:
            await asyncio.gather(*pumps, return_exceptions=True)
        if upstream_writer is not None:
            await close_writer(upstream_writer)
        await close_writer(writer)


async def serve(args, token):
    server = await asyncio.start_server(
        lambda reader, writer: relay_connection(
            reader, writer, args.upstream_host, args.upstream_port, token, args.upstream_socket
        ),
        args.listen_host,
        args.listen_port,
        limit=MAX_HEADER,
    )
    async with server:
        LOG.info("Listening on %s", ", ".join(str(sock.getsockname()) for sock in server.sockets))
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    upstream = parser.add_mutually_exclusive_group(required=True)
    upstream.add_argument("--upstream-port", type=int)
    upstream.add_argument("--upstream-socket")
    parser.add_argument("--token-file", required=True)
    args = parser.parse_args()
    try:
        token = load_token(args.token_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(serve(args, token))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
