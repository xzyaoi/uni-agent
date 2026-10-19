#!/usr/bin/env python3
"""Loopback integration tests; no credentials or remote services required."""

import asyncio
import tempfile
import unittest
from pathlib import Path

import relay


TOKEN = b"test-only-token"


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.servers = []

    async def asyncTearDown(self):
        for server in self.servers:
            server.close()
            await server.wait_closed()

    async def start(self, handler, **kwargs):
        server = await asyncio.start_server(handler, "127.0.0.1", 0, **kwargs)
        self.servers.append(server)
        return server.sockets[0].getsockname()[1]

    async def start_relay(self, upstream_port):
        return await self.start(
            lambda r, w: relay.relay_connection(r, w, "127.0.0.1", upstream_port, TOKEN),
            limit=relay.MAX_HEADER,
        )

    async def exchange(self, port, request):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(request)
            await writer.drain()
            return await asyncio.wait_for(reader.read(), 2)
        finally:
            await relay.close_writer(writer)

    async def test_missing_wrong_and_duplicate_credentials_never_reach_backend(self):
        connected = []

        async def backend(reader, writer):
            connected.append(True)
            await relay.close_writer(writer)

        port = await self.start_relay(await self.start(backend))
        for credentials in (
            b"",
            b"X-API-Key: wrong\r\n",
            b"Authorization: Bearer wrong\r\n",
            b"X-API-Key: " + TOKEN + b"\r\nX-API-Key: wrong\r\n",
        ):
            with self.subTest(credentials=credentials):
                result = await self.exchange(port, b"GET / HTTP/1.1\r\n" + credentials + b"\r\n")
                self.assertTrue(result.startswith(b"HTTP/1.1 401 Unauthorized\r\n"))
        self.assertEqual(connected, [])

    async def test_request_and_streamed_chunked_sse_bytes_preserved(self):
        for auth in (b"X-API-Key: ", b"Authorization: Bearer "):
            request = (
                b"POST /stream?value=%2F HTTP/1.1\r\nHost: localhost\r\n"
                + auth + TOKEN + b"\r\nContent-Length: 9\r\n\r\n\x00payload\xff"
            )
            events = [b"data: first\n\n", b"data: second\n\n"]
            response_header = (
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            chunks = [hex(len(event))[2:].encode() + b"\r\n" + event + b"\r\n" for event in events]
            received = asyncio.get_running_loop().create_future()
            first_consumed = asyncio.Event()

            async def backend(reader, writer):
                try:
                    raw = await reader.readexactly(len(request))
                    received.set_result(raw)
                    writer.write(response_header + chunks[0])
                    await writer.drain()
                    # The client must receive the first event before the backend finishes.
                    await asyncio.wait_for(first_consumed.wait(), 2)
                    for fragment in [chunks[1][:3], chunks[1][3:], b"0\r\n\r\n"]:
                        writer.write(fragment)
                        await writer.drain()
                finally:
                    await relay.close_writer(writer)

            port = await self.start_relay(await self.start(backend))
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                # Send body in the same write to verify buffered post-header bytes survive.
                writer.write(request)
                await writer.drain()
                first = response_header + chunks[0]
                self.assertEqual(await asyncio.wait_for(reader.readexactly(len(first)), 2), first)
                first_consumed.set()
                self.assertEqual(await asyncio.wait_for(reader.read(), 2), chunks[1] + b"0\r\n\r\n")
                self.assertEqual(await received, request)
            finally:
                await relay.close_writer(writer)

    async def test_backend_failure_is_502(self):
        server = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        unused_port = server.sockets[0].getsockname()[1]
        server.close()
        await server.wait_closed()
        port = await self.start_relay(unused_port)
        result = await self.exchange(port, b"GET / HTTP/1.1\r\nX-API-Key: " + TOKEN + b"\r\n\r\n")
        self.assertTrue(result.startswith(b"HTTP/1.1 502 Bad Gateway\r\n"))

    async def test_authenticated_relay_to_unix_socket(self):
        request = b"GET /unix HTTP/1.1\r\nAuthorization: Bearer " + TOKEN + b"\r\n\r\n"
        response = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"
        received = asyncio.get_running_loop().create_future()

        async def backend(reader, writer):
            try:
                received.set_result(await reader.readexactly(len(request)))
                writer.write(response)
                await writer.drain()
            finally:
                await relay.close_writer(writer)

        with tempfile.TemporaryDirectory() as directory:
            socket_path = str(Path(directory) / "upstream.sock")
            backend_server = await asyncio.start_unix_server(backend, path=socket_path)
            try:
                port = await self.start(
                    lambda r, w: relay.relay_connection(r, w, None, None, TOKEN, socket_path),
                    limit=relay.MAX_HEADER,
                )
                self.assertEqual(await self.exchange(port, request), response)
                self.assertEqual(await received, request)
            finally:
                backend_server.close()
                await backend_server.wait_closed()

    async def test_oversize_and_incomplete_headers_rejected(self):
        port = await self.start_relay(1)
        response = await self.exchange(port, b"GET / HTTP/1.1\r\nX: " + b"x" * relay.MAX_HEADER + b"\r\n\r\n")
        self.assertTrue(response.startswith(b"HTTP/1.1 431 "))
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(b"GET / HTTP/1.1\r\n")
            await writer.drain()
            writer.write_eof()
            response = await asyncio.wait_for(reader.read(), 2)
            self.assertTrue(response.startswith(b"HTTP/1.1 400 "))
        finally:
            await relay.close_writer(writer)

    async def test_header_timeout(self):
        original = relay.HEADER_TIMEOUT
        relay.HEADER_TIMEOUT = 0.05
        try:
            port = await self.start_relay(1)
            response = await self.exchange(port, b"GET / HTTP/1.1\r\n")
            self.assertTrue(response.startswith(b"HTTP/1.1 408 "))
        finally:
            relay.HEADER_TIMEOUT = original


if __name__ == "__main__":
    unittest.main()
