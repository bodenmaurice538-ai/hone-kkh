"""A local forwarding proxy that records the genuine model wire bodies.

Claude Code is pointed at this proxy with ``ANTHROPIC_BASE_URL``. The proxy
forwards every request to the real upstream (``https://api.anthropic.com`` by
default) unchanged, streaming the response back so the headless run behaves
normally, and records the exact request and response bytes of each
``POST /v1/messages`` call. Those recorded pairs become the trajectory's model
turns, so the uploaded trajectory carries the real bytes the subscription
produced rather than anything reconstructed.

The proxy never needs the model credential: it forwards whatever ``Authorization``
and ``anthropic-beta`` headers Claude Code sends. It also never writes that
credential anywhere -- :meth:`CaptureProxy.turns` returns bodies only, and the
bearer token lives solely in the forwarded headers, which are not retained.

Plain HTTP is served on loopback, so Claude Code talks to the proxy without TLS
interception; the proxy itself speaks HTTPS to the upstream. Bind only to
127.0.0.1.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import httpx

# Headers that describe one hop and must not be forwarded verbatim.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "accept-encoding",  # let httpx negotiate; we relay decoded bytes
    }
)

# The exact path whose bodies become trajectory model turns. Everything else --
# including /v1/messages/count_tokens and /v1/models -- is still forwarded, just
# not recorded, so a token-count call never becomes a bogus model turn.
_RECORDED_PATH = "/v1/messages"

_MAX_BODY_BYTES = 64 * 1024 * 1024


@dataclass
class ModelCall:
    """One recorded ``POST /v1/messages`` exchange, request and response bytes."""

    request_body: bytes
    response_body: bytes
    status_code: int
    # The ``content-type`` the upstream returned, e.g. ``application/json`` or
    # ``text/event-stream``; lets a reader tell a streamed response from a whole one.
    response_content_type: str = ""


@dataclass
class _Shared:
    upstream: str
    client: httpx.Client
    calls: list[ModelCall] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, call: ModelCall) -> None:
        with self.lock:
            self.calls.append(call)


def _forward_headers(raw_headers) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in raw_headers.items():
        if key.lower() in _HOP_BY_HOP:
            continue
        out[key] = value
    return out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # BaseHTTPRequestHandler logs every request to stderr; silence it.
    def log_message(self, *_args) -> None:  # noqa: D401,N802
        return

    @property
    def _shared(self) -> _Shared:
        return self.server._shared  # type: ignore[attr-defined]

    def _relay(self, method: str) -> None:
        shared = self._shared
        length = int(self.headers.get("Content-Length") or 0)
        if length < 0 or length > _MAX_BODY_BYTES:
            self.send_error(413, "request body too large")
            return
        body = self.rfile.read(length) if length else b""
        path = urlsplit(self.path).path
        recorded = method == "POST" and path == _RECORDED_PATH
        url = shared.upstream.rstrip("/") + self.path
        headers = _forward_headers(self.headers)
        try:
            upstream = shared.client.build_request(method, url, headers=headers, content=body)
            response = shared.client.send(upstream, stream=True)
        except httpx.HTTPError as error:
            self.send_error(502, f"upstream request failed: {type(error).__name__}")
            return

        captured = bytearray()
        try:
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() in _HOP_BY_HOP:
                    continue
                self.send_header(key, value)
            # We relay a decoded, length-unknown stream; force connection close so
            # neither side waits on a content-length we are not echoing.
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in response.iter_raw():
                if not chunk:
                    continue
                if recorded and len(captured) < _MAX_BODY_BYTES:
                    captured.extend(chunk)
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Claude Code hung up; stop relaying but keep what we captured.
                    break
        finally:
            response.close()

        if recorded:
            shared.record(
                ModelCall(
                    request_body=bytes(body),
                    response_body=bytes(captured),
                    status_code=response.status_code,
                    response_content_type=response.headers.get("content-type", ""),
                )
            )

    def do_POST(self) -> None:  # noqa: N802
        self._relay("POST")

    def do_GET(self) -> None:  # noqa: N802
        self._relay("GET")

    def do_DELETE(self) -> None:  # noqa: N802
        self._relay("DELETE")


class CaptureProxy:
    """A loopback forwarding proxy that records ``/v1/messages`` wire bodies.

    Use as a context manager. ``base_url`` is what to hand Claude Code as
    ``ANTHROPIC_BASE_URL``; ``turns`` returns the recorded calls in order.
    """

    def __init__(
        self,
        *,
        upstream: str = "https://api.anthropic.com",
        timeout_s: float = 660.0,
        host: str = "127.0.0.1",
    ) -> None:
        if not upstream.startswith("https://") and not upstream.startswith("http://"):
            raise ValueError("upstream must be an http(s) URL")
        self._host = host
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=20.0), follow_redirects=False
        )
        self._shared = _Shared(upstream=upstream, client=self._client)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "CaptureProxy":
        server = ThreadingHTTPServer((self._host, 0), _Handler)
        server._shared = self._shared  # type: ignore[attr-defined]
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="codeagent-proxy", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self._client.close()

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("proxy is not running")
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self.port}"

    @property
    def turns(self) -> list[ModelCall]:
        """The recorded model calls so far, oldest first (a copy)."""
        with self._shared.lock:
            return list(self._shared.calls)
