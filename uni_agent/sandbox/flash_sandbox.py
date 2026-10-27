"""OpenTela hosted Flash Sandbox provider.

Drives a single remote sandbox over the ``flash-sandbox`` SDK
(https://pypi.org/project/flash-sandbox/): lifecycle through
``AsyncHTTPClient.start_sandbox`` / ``stop_sandbox`` and the data plane
through ``exec_command`` / ``read_file`` / ``write_file``. The agent loop
stays in the uni-agent framework process; this provider is only the tool
boundary, so it never touches the policy endpoint (the gateway) itself.

Run-wide config (``SandboxConfig.sandbox_kwargs`` or the env vars below):

* ``endpoint`` / ``FLASH_SANDBOX_ENDPOINT`` -- orchestrator base URL.
  Default ``https://api.opentela.ai/v1/service/flash-sandbox`` (the same
  endpoint the rllm Flash Sandbox backend targets).
* ``api_key_env`` -- name of the env var holding the API key. Default
  ``OTELA_API`` (set it once on the login node / Ray runtime env).
* ``sandbox_type`` / ``FLASH_SANDBOX_TYPE`` -- backend hint forwarded to
  ``start_sandbox(type=...)`` (e.g. ``"docker"``). Empty = let the cluster
  pick any available backend.
* ``kernel_image`` / ``initrd_path`` -- optional microVM knobs for the
  ``firecracker``/``libkrun`` backends (forwarded only when set).
* ``cpu_cores`` / ``memory_mb`` -- resource request/limit per sandbox.
* ``start_timeout`` / ``exec_timeout`` -- startup/command budgets (seconds).
* ``fenced_creates`` -- opt into one stable create ID, operation and absolute
  infrastructure deadline across startup retries; requires a modern SDK. The
  benchmark ownership wrapper requires this mode automatically.
* ``env_vars`` -- startup variables, also used as defaults for commands and
  persistent shells; per-command/shell variables take precedence.
* ``keep_alive`` / ``keep_alive_interval_sec`` -- a background task pings
  ``keepalive`` so a long episode never trips the wall-clock ``ttl_seconds``
  cap. ``keep_alive=False`` opts out (the TTL alone governs lifetime).

Provider-agnostic SWE rows carry the canonical open-source image ref
(``swebench/sweb.eval.x86_64.<id>``); this provider pulls it directly from
the Docker registry the orchestrator is configured against, mirroring how
``flash-sandbox`` pulls its default ``python:3.11-slim``. Override the
image in the per-row task config or via ``sandbox_kwargs["image"]``.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import math
import os
import random
import shlex
import time
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

from .base import ExecResult, Sandbox
from .exec_context import current_exec_metadata, reset_exec_metadata, set_exec_metadata
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig

logger = logging.getLogger(__name__)

#: Default orchestrator endpoint (override with FLASH_SANDBOX_ENDPOINT).
DEFAULT_FLASH_SANDBOX_ENDPOINT = os.environ.get(
    "FLASH_SANDBOX_ENDPOINT", "https://api.opentela.ai/v1/service/flash-sandbox"
)
#: Default API key env var name.
_DEFAULT_API_KEY_ENV = "OTELA_API"

# Org-aware registry-rewrite rules for :meth:`FlashSandboxSandbox._resolve_image`:
# source Docker Hub org -> whether the mirror DROPS the org from the path.
# ``swebench`` mirrors live directly under the registry root
# (ghcr.io/swe-images/sweb.eval.*), while the Terminal-Bench mirror keeps the
# org as a namespace (ghcr.io/swe-images/terminal-bench/<task>:<tag>). Unknown
# orgs keep the legacy drop-org behavior.
_FLASH_IMAGE_ORG_RULES: dict[str, bool] = {
    "swebench": True,
    "terminal-bench": False,
}
#: Sandbox statuses that accept commands (mirror the SDK's _RUNNING_STATUSES).
_RUNNING_STATUSES = frozenset({"running", "created", "restarting", "resuming"})

#: Bootstrap a rootless-compatible apt-get so a generic image can install
#: packages under the sandbox's rootless namespace (proven by the rllm Flash
#: Sandbox backend). Best-effort: a failure is logged, never fatal, so a
#: locked-down image (no apt, read-only /usr/local/bin) still starts.
#:
#: The flash-sandbox is rootless, so apt's default sandbox (chown download
#: dirs to the ``_apt`` system user) fails with EPERM and ``apt-get update``
#: writes no index -> ``apt-get install`` can't locate the package. The shim
#: fixes both: ``APT::Sandbox::User=root`` skips the ``_apt`` chown, and
#: ``install`` runs its OWN ``update`` into a writable temp dir before
#: downloading, so it works regardless of the caller's separate ``update``.
_ROOTLESS_APT_GET_PATH = "/usr/local/bin/apt-get"
_ROOTLESS_APT_GET = b"""#!/bin/bash
set -euo pipefail

real_apt_get=/usr/bin/apt-get
# Rootless: there is no _apt user, so apt's sandbox (chown to _apt) fails
# with EPERM on /var/lib/apt/lists. Run the apt methods as root so no chown
# is attempted, and redirect the lists cache to a writable per-call temp dir.
apt_rootless() {
    local lists="$(mktemp -d)"
    "$real_apt_get" -o APT::Sandbox::User=root -o "Dir::State::lists=$lists" "$@"
    local rc=$?
    rm -rf "$lists"
    return $rc
}

is_install=false
for argument in "$@"; do
    if [[ "$argument" == "install" ]]; then
        is_install=true
        break
    fi
done

if [[ "$is_install" != true ]]; then
    # update / source / etc.: run rootless into a throwaway dir. Never fatal
    # (callers do `apt-get update || true`); the real index work happens in
    # the install branch below.
    apt_rootless "$@" || true
    exit 0
fi

# install: refresh the index into a fresh temp dir (the caller's `update`
# likely failed rootless), then download + extract the .deb(s) -- no dpkg
# needed, so this works even with a read-only dpkg database.
lists="$(mktemp -d)"
archive_dir="$(mktemp -d)"
trap 'rm -rf "$lists" "$archive_dir"' EXIT
"$real_apt_get" -o APT::Sandbox::User=root -o "Dir::State::lists=$lists" update -qq || true
"$real_apt_get" -o APT::Sandbox::User=root -o "Dir::State::lists=$lists" \
    -o "Dir::Cache::archives=$archive_dir/" --download-only "$@"
find "$archive_dir" -maxdepth 1 -type f -name '*.deb' -print0 |
    while IFS= read -r -d '' package; do
        dpkg-deb --fsys-tarfile "$package" | tar --no-same-owner -xf - -C /
    done
"""


def _sandbox_http_error_types() -> tuple[type, ...]:
    """Lazily import the flash-sandbox HTTP error types.

    Returns ``()`` when the SDK is unavailable (e.g. a stripped install) so
    the name-based fallback in :func:`_is_transient_http_error` is used.
    """
    try:
        from flash_sandbox import SandboxHTTPError, SandboxTimeoutError
    except Exception:  # pragma: no cover - env-dependent
        return ()
    return (SandboxHTTPError, SandboxTimeoutError)


_TRANSIENT_HTTP_STATUSES = frozenset({400, 502, 503, 504})

# Cache for :func:`_accepts_kwarg`, keyed by (module, qualname, kwarg name) so
# the hot exec/shell path never re-runs ``inspect.signature`` per call.
_ACCEPTS_KWARG_CACHE: dict[tuple[str, str, str], bool] = {}


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Whether ``fn`` accepts a keyword named ``name`` (cached).

    Used to forward ambient exec metadata only to SDK methods that advertise a
    ``metadata`` parameter, so a stale dependency (e.g. an installed
    ``flash-sandbox`` whose ``shell_exec`` predates the metadata kwarg) is a
    silent no-op rather than a ``TypeError`` on the agent's primary command
    path. ``**kwargs`` also counts as accepting it.
    """
    key = (getattr(fn, "__module__", ""), getattr(fn, "__qualname__", ""), name)
    cached = _ACCEPTS_KWARG_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        params = inspect.signature(fn).parameters
        ok = name in params and params[name].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ) or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (ValueError, TypeError):  # pragma: no cover - C-builtins, proxies
        ok = False
    _ACCEPTS_KWARG_CACHE[key] = ok
    return ok


def _forward_metadata(kwargs: dict[str, Any], fn: Any, *, name: str = "metadata") -> None:
    """Inject ambient exec metadata into ``kwargs`` when ``fn`` accepts it.

    Reads :func:`current_exec_metadata` (threaded from the training loop via
    :mod:`uni_agent.sandbox.exec_context`) and, only when the bag is non-empty
    AND ``fn`` advertises a ``metadata`` keyword, sets ``kwargs[name]`` to a
    **deep copy** of the bag. An uninstrumented call path (empty bag) or a
    stale SDK (no kwarg) is a silent no-op, so the wire shape matches
    pre-instrumentation behaviour exactly and this never raises.

    The deep copy is load-bearing: the flash-sandbox SDK serialises the body
    lazily -- :meth:`AsyncHTTPClient._request` awaits ``_get_session()`` (which
    yields the loop) *before* aiohttp JSON-encodes ``json=payload``. If
    ``kwargs[name]`` shared a nested dict (e.g. ``rlvr``) with the ambient bag,
    any in-place mutation of that nested dict between this call and the eventual
    serialization would corrupt the request body. Deep-copying here makes every
    request fully independent of the ambient bag and of any later
    ``bind_exec_metadata`` / ``terminal_exec`` overlay, regardless of SDK timing.
    """
    meta = current_exec_metadata()
    accepts = _accepts_kwarg(fn, name)
    if not meta:
        logger.info(
            "_forward_metadata SKIP: current_exec_metadata empty "
            "(fn=%s name=%s accepts=%s)",
            getattr(fn, "__qualname__", fn), name, accepts,
        )
        return
    if not accepts:
        logger.info(
            "_forward_metadata SKIP: fn=%s does not accept kwarg %r "
            "(meta keys=%s)",
            getattr(fn, "__qualname__", fn), name, sorted(meta.keys()),
        )
        return
    # Deep-copy so the request body is isolated from the ambient bag: the SDK
    # serialises lazily (after an await), and a shallow copy would let a later
    # in-place mutation of rlvr/llm bleed into a request already in flight.
    kwargs[name] = copy.deepcopy(meta)
    logger.info(
        "_forward_metadata OK: fn=%s name=%s meta keys=%s rlvr=%s",
        getattr(fn, "__qualname__", fn), name, sorted(meta.keys()),
        meta.get("rlvr"),
    )


def _is_transient_http_error(exc: BaseException, *, idempotent: bool) -> bool:
    """Retry commands only after an authoritative admission refusal.

    A gateway error or lost response can leave the original command running.
    Non-idempotent calls require structured worker/coordinator evidence that
    admission was cancelled before dispatch. Idempotent file operations retain
    their transient HTTP and transport retries; error prose is never evidence.
    """
    if isinstance(exc, asyncio.CancelledError):
        return False
    name = type(exc).__name__
    if not idempotent:
        return (
            name == "SandboxHTTPError"
            and getattr(exc, "status_code", None) in {408, 503}
            and getattr(exc, "code", None) == "admission_cancelled"
            and getattr(exc, "origin", None) in {"coordinator", "worker"}
            and getattr(exc, "outcome", None) == "not_dispatched"
        )
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return idempotent
    if isinstance(exc, OSError):  # ConnectionError, ConnectionResetError, ...
        return True
    if name == "SandboxTimeoutError":
        return idempotent
    if name == "SandboxHTTPError":
        status = getattr(exc, "status_code", 0) or 0
        if status in _TRANSIENT_HTTP_STATUSES:
            return True
        if status == 500 and idempotent:
            return True
    return False


def _env_number(name: str, default: float) -> float:
    """Read env var ``name`` as a number, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using default %s", name, raw, default)
        return default


def _wrap_command(argv: list[str], workdir: str | None, env: dict[str, str] | None) -> list[str]:
    """Build the argv passed to ``exec_command``.

    Apply explicit variables after the login profile, which may reset PATH.
    Quote complete assignments so their values are never re-expanded, and
    retain the rootless tar default unless the caller overrides it.
    """
    inner = _inner_shell_string(argv)
    if workdir:
        inner = f"cd {shlex.quote(workdir)} && {inner}"
    variables = {"TAR_OPTIONS": "--no-same-owner", **(env or {})}
    exports = shlex.join([f"{key}={value}" for key, value in variables.items()])
    return ["/bin/bash", "-lc", f"export -- {exports} && {inner}"]


def _inner_shell_string(argv: list[str]) -> str:
    """Collapse ``["bash","-c","..."]`` shapes back to the inner script."""
    if len(argv) == 3 and argv[1] == "-c" and argv[0] in {"sh", "/bin/sh", "bash", "/bin/bash"}:
        return argv[2]
    return shlex.join(argv)


class _FlashShellHandle:
    """Adapt the SDK's persistent ``shell_exec`` to the :class:`Shell` contract.

    ``shell_exec`` runs every command in-place in one long-lived shell, so
    ``cd`` / ``export`` / ``&&`` / ``;`` survive across calls with no tmux
    fallback (the sweb.eval images ship WITHOUT tmux, and rootless apt can't
    install it reliably). Each call is hard-capped server-side via
    ``timeout_ms``: the orchestrator kills the command and replies promptly
    while the shell stays immediately usable with state intact (verified:
    no stray output, no queueing) -- which we surface as
    :class:`asyncio.TimeoutError` so the framework's
    :class:`~uni_agent.tools.shell.SandboxShell` marks the result ``timed_out``.
    """

    def __init__(self, sandbox: FlashSandboxSandbox) -> None:
        self._sandbox = sandbox

    async def start(self) -> None:
        return None  # the persistent shell exists once the sandbox is started.

    async def run(self, command: str, *, timeout: float = 120.0) -> ExecResult:
        result = await self._sandbox._shell_exec(command, timeout_ms=int(timeout * 1000))
        # Server-side ``timeout_ms`` kill -> exit_code -1 + a "context
        # deadline exceeded" HTTP marker. Surface it as a real timeout so
        # the agent gets the "command cancelled" observation, not exit -1.
        if result.exit_code < 0 and "context deadline exceeded" in (result.stderr or "").lower():
            raise asyncio.TimeoutError(
                f"flash-sandbox shell_exec exceeded the {timeout:g}s per-command timeout"
            )
        return result

    async def close(self) -> None:
        return None  # the sandbox lifecycle (stop()) owns teardown.


class FlashCreateRecoveryError(RuntimeError):
    """A create cannot be restarted; its original ownership remains available."""

    def __init__(self, message: str, recovery: dict[str, Any]) -> None:
        super().__init__(message)
        self.sandbox_id = recovery["sandbox_id"]
        self.operation_id = recovery["operation_id"]
        self.infrastructure_deadline_unix_ms = recovery["infrastructure_deadline_unix_ms"]


@register_sandbox("flash_sandbox")
class FlashSandboxSandbox(Sandbox):
    """One remote sandbox on the OpenTela Flash Sandbox orchestrator.

    The provider owns the client + sandbox lifecycle (:meth:`start` /
    :meth:`stop`) and implements the data-plane primitives
    (:meth:`_exec` / :meth:`read_file` / :meth:`write_file`) the framework's
    tools depend on. A background keepalive task, when enabled, extends the
    sandbox lease so long episodes outlive the create-time TTL.
    """

    def __init__(
        self,
        *,
        image: str = "python:3.12",
        runtime_timeout: float = 3600.0,
        endpoint: str | None = None,
        api_key_env: str = _DEFAULT_API_KEY_ENV,
        sandbox_type: str | None = None,
        kernel_image: str | None = None,
        initrd_path: str | None = None,
        cpu_cores: float | None = None,
        memory_mb: int | None = None,
        start_timeout: float | None = None,
        exec_timeout: float | None = None,
        fenced_creates: bool = False,
        env_vars: dict[str, str] | None = None,
        keep_alive: bool = True,
        keep_alive_interval_sec: float = 45.0,
        **kwargs: Any,
    ) -> None:
        self.image = self._resolve_image(image)
        self.runtime_timeout = float(runtime_timeout)
        self.endpoint = endpoint or os.getenv("FLASH_SANDBOX_ENDPOINT", DEFAULT_FLASH_SANDBOX_ENDPOINT)
        self._api_key_env = api_key_env
        self._sandbox_type = sandbox_type if sandbox_type is not None else os.getenv("FLASH_SANDBOX_TYPE", "")
        self._kernel_image = kernel_image or os.getenv("FLASH_SANDBOX_KERNEL_IMAGE") or None
        self._initrd_path = initrd_path or os.getenv("FLASH_SANDBOX_INITRD_PATH") or None
        self._cpu_cores = cpu_cores
        self._memory_mb = memory_mb
        self.env_vars = dict(env_vars or {})
        self._start_timeout = float(start_timeout) if start_timeout is not None else _env_number(
            "FLASH_SANDBOX_START_TIMEOUT_S", 300.0
        )
        self._exec_timeout = float(exec_timeout) if exec_timeout is not None else _env_number(
            "FLASH_SANDBOX_EXEC_TIMEOUT_S", 3600.0
        )
        self._keep_alive = keep_alive
        self._keep_alive_interval_sec = float(keep_alive_interval_sec)
        # Retry budget for transient file-operation failures and commands
        # refused before dispatch with authoritative admission evidence.
        self._http_retries = int(_env_number("FLASH_SANDBOX_HTTP_RETRIES", 3.0))
        if kwargs:
            logger.debug("Ignoring unsupported flash-sandbox options: %s", sorted(kwargs))

        self._fenced_creates = bool(fenced_creates)
        self._create_payload: dict[str, Any] | None = None
        self._create_state = "new"
        self._start_lock = asyncio.Lock()
        self._client_factory: Any | None = None
        self._client: Any | None = None
        self._sandbox: Any | None = None
        self._sandbox_id: str | None = None
        self._keepalive_task: asyncio.Task[None] | None = None

    @staticmethod
    def _resolve_image(image: str) -> str:
        """Map canonical open-source task-image refs to the configured registry.

        SWE rows carry ``<org>/sweb.eval.x86_64.<id>`` (e.g.
        ``swebench/sweb.eval.x86_64.astropy_1776_astropy-12907``), the canonical
        ref published on Docker Hub. Some orchestrators (OpenTela) silently fall
        back to their ``python:3.x`` default instead of pulling that ref, which
        ships without /testbed, git and the SWE harness -> the agent has
        nothing to edit (the stateful shell itself needs no tmux; see
        :meth:`open_shell`).

        ``FLASH_SANDBOX_IMAGE_REGISTRY`` rewrites bare Docker Hub org refs to
        the mirror. The rewrite is org-aware (:data:`_FLASH_IMAGE_ORG_RULES`):

        * ``swebench/<name>`` -> ``<registry>/<name>`` (the SWE-bench
          Verified mirror lives directly under ``ghcr.io/swe-images``);
        * ``terminal-bench/<name>`` -> ``<registry>/terminal-bench/<name>``
          (the Terminal-Bench mirror KEEPS the org as a namespace, e.g.
          ``terminal-bench/pypi-server:20251031`` ->
          ``ghcr.io/swe-images/terminal-bench/pypi-server:20251031``);
        * any other bare org -> ``<registry>/<name>`` (legacy behavior).

        Already-qualified refs (``ghcr.io/...``, ``docker.io/...``) and
        tag-only refs (``python:3.12``) are passed through unchanged.
        """
        registry = os.getenv("FLASH_SANDBOX_IMAGE_REGISTRY", "").strip()
        if not registry or "/" not in image:
            return image
        first = image.split("/", 1)[0]
        # A registry host carries a "." (ghcr.io), a ":" (localhost:5000), or
        # is "localhost" -- anything else is a bare Docker Hub org to rewrite.
        if "." in first or ":" in first or first == "localhost":
            return image
        name = image.split("/", 1)[1]
        if not _FLASH_IMAGE_ORG_RULES.get(first, True):
            name = f"{first}/{name}"
        resolved = f"{registry.rstrip('/')}/{name}"
        if resolved != image:
            logger.info("flash-sandbox image registry remap: %s -> %s", image, resolved)
        return resolved

    @classmethod
    def from_config(cls, config: SandboxConfig) -> FlashSandboxSandbox:
        # Standard fields feed the constructor; provider-specific knobs
        # (endpoint, api_key_env, sandbox_type, resources, keep_alive ...)
        # ride in sandbox_kwargs. Runtime timeout maps to the sandbox TTL so
        # the lease lasts at least as long as the task's per-episode cap.
        return cls(image=config.image, runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs)

    # ----- control plane -----
    @property
    def create_recovery(self) -> dict[str, Any] | None:
        """Retained ownership for bounded cleanup/reconciliation, including lost ACKs.

        A pending create is not canceled by closing its client or by an absent-ID
        DELETE. Its absolute infrastructure deadline bounds further creation.
        """
        if self._create_payload is None:
            return None
        return {"state": self._create_state, **{
            name: self._create_payload[name] for name in
            ("sandbox_id", "operation_id", "infrastructure_deadline_unix_ms")
        }}

    async def start(self) -> None:
        async with self._start_lock:
            await self._start_once()

    async def _start_once(self) -> None:
        if self._create_payload is not None and self._create_state in {"closed", "cleanup_pending"}:
            raise FlashCreateRecoveryError(
                "This fenced create has entered cleanup; use a new provider for a new episode", self.create_recovery
            )
        if self._sandbox is not None:
            return  # already started

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Flash Sandbox requires {self._api_key_env} to be set "
                "(e.g. export OTELA_API=...)"
            )

        try:
            from flash_sandbox import AsyncHTTPClient
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "The flash-sandbox provider requires the 'flash-sandbox' package. "
                "Install it with: pip install flash-sandbox"
            ) from exc

        self._fenced_creates = self._fenced_creates or bool(
            getattr(AsyncHTTPClient, "requires_fenced_create", False)
        )
        if self._fenced_creates:
            if not math.isfinite(self._start_timeout) or self._start_timeout <= 0:
                raise ValueError("Fenced creation requires a positive finite start_timeout")
            if self._create_payload is not None and time.time() * 1000 >= self._create_payload[
                "infrastructure_deadline_unix_ms"
            ]:
                raise FlashCreateRecoveryError(
                    "Fenced create deadline expired; ownership retained for reconciliation", self.create_recovery
                )

        # The HTTP client timeout is a fixed per-request budget. It must
        # OUTWAIT the longest server-side enforcement (exec commands up to
        # ``exec_timeout``), so a command that completes just under the cap
        # still returns rather than tripping the client's own clock. A 60s
        # margin covers network latency and clock skew.
        client_timeout = max(self._exec_timeout + 60.0, self._start_timeout + 30.0, 360.0)
        if self._client is None:
            self._client_factory = AsyncHTTPClient
            self._client = AsyncHTTPClient(self.endpoint, api_key=api_key, timeout=client_timeout)
        if self._fenced_creates:
            required = ("sandbox_id", "operation_id", "infrastructure_deadline_unix_ms")
            if not all(_accepts_kwarg(self._client.start_sandbox, name) for name in required):
                raise RuntimeError("Fenced creation requires an SDK with explicit operation/deadline support")

        start_kwargs: dict[str, Any] = {
            "image": self.image,
            "type": self._sandbox_type,
            "command": ["tail", "-f", "/dev/null"],
            "ttl_seconds": int(self.runtime_timeout),
            "request_timeout": float(self._start_timeout),
        }
        if self._kernel_image is not None:
            start_kwargs["kernel_image"] = self._kernel_image
        if self._initrd_path is not None:
            start_kwargs["initrd_path"] = self._initrd_path
        if self._cpu_cores is not None:
            start_kwargs["cpu_cores"] = float(self._cpu_cores)
        if self._memory_mb is not None:
            start_kwargs["memory_mb"] = int(self._memory_mb)
        if self.env_vars:
            start_kwargs["env"] = dict(self.env_vars)

        # Tag the sandbox with the rollout-group / trajectory metadata the
        # coordinator consumes for RLVR-aware placement and create-queue
        # ordering (see flash-sandbox docs/design/rlvr-group-completion-
        # scheduler.md). The bag is threaded from the framework's
        # ``_run_session`` via :mod:`uni_agent.sandbox.exec_context`; empty
        # (e.g. a standalone run) -> nothing is sent, matching prior behaviour.
        if self._create_payload is None:
            _forward_metadata(start_kwargs, self._client.start_sandbox)
            if self._fenced_creates:
                start_kwargs.update(
                    sandbox_id=str(uuid4()),
                    operation_id=str(uuid4()),
                    infrastructure_deadline_unix_ms=int((time.time() + self._start_timeout) * 1000),
                    # HTTP must outlive the fixed infrastructure deadline plus
                    # the SDK's five-second response grace, on every attempt.
                    request_timeout=self._start_timeout + 6.0,
                )
                self._create_payload = copy.deepcopy(start_kwargs)
                self._create_state = "pending"
        if self._create_payload is not None:
            start_kwargs = copy.deepcopy(self._create_payload)

        # Self-diagnostic: log the EXACT (endpoint, type, image) the SDK gets,
        # so backend-selection fixes (e.g. FLASH_SANDBOX_TYPE=libkrun when
        # docker is 503) are verifiable in the run log without a separate
        # probe. type='' lets the cluster pick (resolves to docker on
        # api.opentela.ai as of 08-30 -> 503 'no healthy nodes for backend
        # docker'; libkrun is the healthy microVM backend). print() (not
        # logger.info) so it ALWAYS reaches the .out regardless of the
        # framework's per-logger level config.
        import sys as _sys
        print(
            f"FlashSandbox start_sandbox: endpoint={self.endpoint} "
            f"type={self._sandbox_type!r} image={self.image} "
            f"registry={os.getenv('FLASH_SANDBOX_IMAGE_REGISTRY', '')!r} "
            f"(FLASH_SANDBOX_TYPE={os.getenv('FLASH_SANDBOX_TYPE', '')!r})",
            flush=True, file=_sys.stderr,
        )

        created = await self._client.start_sandbox(**start_kwargs)
        if self._fenced_creates and str(getattr(created, "id", created)) != start_kwargs["sandbox_id"]:
            raise RuntimeError("Fenced create returned a different sandbox ID; original ownership retained")
        self._sandbox = created
        if self._fenced_creates:
            self._create_state = "acknowledged"
        # start_sandbox returns an AsyncSandbox (with .id); accept a bare id for
        # custom client implementations the same way the SDK's create() does.
        self._sandbox_id = str(getattr(self._sandbox, "id", self._sandbox))

        if self._keep_alive:
            self._keepalive_task = asyncio.create_task(
                self._keepalive_loop(), name=f"flash-sandbox-keepalive-{self._sandbox_id}"
            )

        await self._bootstrap()
        logger.info("FlashSandbox created (id: %s, image: %s)", self._sandbox_id, self.image)

    async def _bootstrap(self) -> None:
        """Best-effort rootless-apt + git safe.directory setup.

        Mirrors the rllm Flash Sandbox bootstrap so a generic image can run
        ``apt-get install`` and ``git`` under the sandbox's rootless
        namespace. Every step is guarded: a locked-down image (read-only
        /usr/local/bin, no apt, no git) still starts cleanly.
        """
        try:
            await self._client.write_file(self._sandbox_id, _ROOTLESS_APT_GET_PATH, _ROOTLESS_APT_GET)
            await self._exec(["chmod", "755", _ROOTLESS_APT_GET_PATH])
        except Exception:
            logger.debug("flash-sandbox: rootless apt-get shim not installed", exc_info=True)

        try:
            await self._exec(
                ["/bin/bash", "-c", "if command -v git >/dev/null 2>&1; then "
                 "git config --system --get-all safe.directory | grep -Fxq '*' || "
                 "git config --system --add safe.directory '*'; fi"]
            )
        except Exception:
            logger.debug("flash-sandbox: git safe.directory not set", exc_info=True)

    async def _keepalive_loop(self) -> None:
        """Extend this sandbox's lease until :meth:`stop` cancels us.

        A ping that reports the sandbox already gone (SandboxNotFoundError)
        ends the loop; transient errors are logged and retried on the next
        tick so one failed ping never tears down a live episode.
        """
        try:
            from flash_sandbox import SandboxNotFoundError
        except ImportError:  # pragma: no cover - env-dependent
            SandboxNotFoundError = ()  # type: ignore[assignment,misc]

        ttl = max(int(self.runtime_timeout), 120)
        while True:
            try:
                await asyncio.sleep(self._keep_alive_interval_sec)
                await self._client.keepalive(self._sandbox_id, ttl)
            except SandboxNotFoundError:
                return  # sandbox already gone; nothing to keep alive
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # transient -- log and retry next tick
                logger.debug("flash-sandbox keepalive ping failed for %s: %s", self._sandbox_id, exc)

    async def stop(self) -> None:
        # Idempotent via the None checks below: a second call finds nothing to do.
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._keepalive_task = None

        if self._create_state == "cleanup_pending" and self._sandbox_id is not None and self._client is None:
            api_key = os.environ.get(self._api_key_env)
            if not api_key or self._client_factory is None:
                raise FlashCreateRecoveryError("Owned cleanup requires its authenticated client", self.create_recovery)
            # Retain the exact client factory, including the benchmark's
            # synchronous deletion contract, after startup substitution ends.
            self._client = self._client_factory(self.endpoint, api_key=api_key, timeout=150.0)
        if self._create_state == "pending":
            logger.warning("Fenced create outcome is unresolved; retaining ownership: %s", self.create_recovery)
            # No DELETE: a queued create can outlive a 404, and deleting a
            # lost-ACK success would turn the next same-fence retry into a
            # cached success for an already-deleted sandbox.
        if self._sandbox_id is not None and self._client is not None:
            if self._create_payload is not None:
                self._create_state = "cleanup_pending"
            try:
                await self._client.stop_sandbox(self._sandbox_id, cleanup=True, timeout=120.0)
            except Exception:
                logger.warning(
                    "Failed to stop Flash sandbox %s; owned cleanup remains necessary",
                    self._sandbox_id,
                    exc_info=True,
                )
            else:
                if self._create_payload is not None:
                    self._create_state = "closed"
                self._sandbox_id = None
        self._sandbox = None
        if self._create_payload is None:
            self._sandbox_id = None

        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                logger.debug("flash-sandbox client close failed", exc_info=True)
            self._client = None

    async def _with_retry(self, desc: str, fn: Any, *, idempotent: bool) -> Any:
        """Run a flash-sandbox SDK call with retry on transient HTTP errors.

        ``fn`` is a zero-arg awaitable factory so each attempt is a fresh
        call (the underlying request must not be replayed mid-flight). Non-
        transient errors re-raise immediately so the existing
        :meth:`Sandbox.exec` -> :meth:`is_alive` contract still decides
        between downgrading to ``exit_code=127`` and re-raising.
        """
        last: BaseException | None = None
        for attempt in range(self._http_retries + 1):
            try:
                return await fn()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not _is_transient_http_error(exc, idempotent=idempotent):
                    raise
                last = exc
                if attempt < self._http_retries:
                    delay = min(0.5 * (2 ** attempt) + random.random() * 0.25, 5.0)
                    logger.debug(
                        "flash-sandbox: %s transient error (attempt %d/%d): %s; retrying in %.2fs",
                        desc, attempt + 1, self._http_retries + 1, exc, delay,
                    )
                    await asyncio.sleep(delay)
        assert last is not None
        raise last

    def _require(self) -> Any:
        if self._client is None or self._sandbox_id is None:
            raise RuntimeError("FlashSandbox not started; call start() first")
        return self._client

    #: Native persistent shell via the SDK's ``shell_exec`` (no tmux needed):
    #: ``cd`` / ``export`` survive across calls and each call is hard-capped
    #: server-side via ``timeout_ms``. See :class:`_FlashShellHandle` and
    #: :meth:`open_shell`.
    supports_shell: ClassVar[bool] = True

    async def open_shell(
        self,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> Any:
        """Return a long-lived shell handle backed by the SDK's ``shell_exec``.

        ``cd`` / ``export`` / ``&&`` / ``;`` persist in-place across calls
        (no tmux fallback -- the sweb.eval images ship without tmux, and the
        persistent ``shell_exec`` makes it unnecessary). ``env`` / ``cwd`` are
        applied once in the actual persistent shell, after its startup scripts.
        Later commands can change its environment and working directory. The
        handle enforces the per-command timeout server-side via ``timeout_ms``.
        """
        self._require()
        shell = _FlashShellHandle(self)
        setup = [f"cd -- {shlex.quote(cwd)}"] if cwd else []
        variables = {**self.env_vars, **(env or {})}
        if variables:
            setup.append("export -- " + shlex.join([f"{key}={value}" for key, value in variables.items()]))
        if setup:
            # Initialization is not the agent action (or terminal evaluation)
            # whose metadata the caller may already have bound.
            token = set_exec_metadata(None)
            try:
                result = await shell.run(" && ".join(setup), timeout=self._start_timeout)
            finally:
                reset_exec_metadata(token)
            if result.exit_code != 0:
                raise RuntimeError("Unable to initialize Flash Sandbox persistent shell environment")
        return shell

    # ----- data plane -----
    async def _exec(
        self,
        argv: list[str],
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run ``argv`` once in the sandbox and return its captured result.

        ``timeout`` (seconds) is forwarded as ``timeout_ms`` so the
        orchestrator enforces the cap and replies with a non-zero result
        instead of dropping the connection. A client-side transport timeout
        (HTTP budget exceeded) surfaces as :class:`asyncio.TimeoutError`;
        :meth:`Sandbox._is_timeout_error` -- overridden below -- turns that
        into an ``exit_code=-1`` ExecResult via the public :meth:`exec`
        wrapper, so a slow command is reported, never raised into the loop.
        """
        client = self._require()
        wrapped = _wrap_command(argv, workdir, {**self.env_vars, **(env or {})})
        # The one-shot /exec path is the coordinator's RLVR-observed channel
        # (``ObserveExecOutcome`` advances ``k_g`` when ``rlvr["terminal"]`` is
        # set), and carries the LLM input/output for the turn. ``exec_command``
        # always accepts ``metadata`` in supported SDKs; ``_forward_metadata``
        # is a no-op when the ambient bag is empty (standalone runs).
        exec_kwargs: dict[str, Any] = {
            "timeout_ms": int((timeout or self._exec_timeout) * 1000),
        }
        # Recent SDKs default to /shell. Verifier and transfer commands need
        # an isolated process, without changing the agent's persistent shell.
        if _accepts_kwarg(client.exec_command, "persistent"):
            exec_kwargs["persistent"] = False
        _forward_metadata(exec_kwargs, client.exec_command)

        async def _call() -> Any:
            return await client.exec_command(
                self._sandbox_id,
                wrapped,
                **exec_kwargs,
            )

        result = await self._with_retry("exec", _call, idempotent=False)
        # Normalise to the framework's ExecResult (3 fields). The SDK returns
        # its own dataclass with the same field names plus an optional
        # ``error``; fold ``error`` into stderr so the model still sees it.
        stdout = getattr(result, "stdout", None)
        stderr = getattr(result, "stderr", None)
        error = getattr(result, "error", None)
        stderr = "" if stderr is None else str(stderr)
        if error:
            stderr = f"{stderr}\n{error}" if stderr else str(error)
        return ExecResult(
            exit_code=int(getattr(result, "exit_code", -1) or 0),
            stdout="" if stdout is None else str(stdout),
            stderr=stderr,
        )

    async def _shell_exec(self, command: str, *, timeout_ms: int) -> ExecResult:
        """Run ``command`` in the persistent shell (one call, server-side cap).

        Like :meth:`_exec` but against the SDK's persistent ``shell_exec`` so
        cwd / env / exports survive across calls (``cd /a && export X=1`` then
        a later ``shell_exec`` sees ``/a`` and ``X=1``). ``timeout_ms`` is
        enforced server-side: a command that overruns it is killed and the
        shell stays immediately usable with state intact (no queueing),
        which we report as exit code -1 + a ``context deadline exceeded``
        marker in stderr for :class:`_FlashShellHandle` to surface as a timeout.
        """
        client = self._require()
        shell_kwargs: dict[str, Any] = {"timeout_ms": timeout_ms}
        # The persistent shell is the agent's primary command path. The SDK's
        # ``shell_exec`` only recently grew a ``metadata`` kwarg (the
        # coordinator's /shell route is a dumb forward today, so per-call RLVR
        # signals land on the node's structured event log, not the coordinator
        # observer). Forward only when the installed SDK advertises the kwarg so
        # a stale dependency never crashes this hot path with a ``TypeError``.
        _forward_metadata(shell_kwargs, client.shell_exec)

        async def _call() -> Any:
            # The SDK wraps string commands in `sh -c`, losing cd/exports in
            # a child process. An argv for eval runs the script as a builtin
            # in the actual persistent shell; the server quotes each argument.
            return await client.shell_exec(self._sandbox_id, ["eval", command], **shell_kwargs)

        result = await self._with_retry("shell_exec", _call, idempotent=False)
        stdout = getattr(result, "stdout", None)
        stderr = getattr(result, "stderr", None)
        error = getattr(result, "error", None)
        stderr = "" if stderr is None else str(stderr)
        if error:
            stderr = f"{stderr}\n{error}" if stderr else str(error)
        return ExecResult(
            exit_code=int(getattr(result, "exit_code", -1) or 0),
            stdout="" if stdout is None else str(stdout),
            stderr=stderr,
        )

    async def read_file(self, path: str) -> bytes:
        client = self._require()
        return await self._with_retry(
            "read_file", lambda: client.read_file(self._sandbox_id, path), idempotent=True
        )

    async def write_file(self, path: str, content: bytes | str) -> None:
        client = self._require()
        data = content.encode() if isinstance(content, str) else content
        # Ensure the parent directory exists (the orchestrator's write_file
        # does not guarantee parent creation); cheap and idempotent. Best-effort:
        # a root-owned parent that mkdir cannot create may still be writable
        # by the orchestrator (it writes as the container's root).
        parent = path.rsplit("/", 1)[0]
        if parent and parent != "/":
            try:
                await self._exec(["mkdir", "-p", parent])
            except Exception:
                logger.debug("flash-sandbox: mkdir -p %s failed (continuing)", parent, exc_info=True)
        await self._with_retry(
            "write_file", lambda: client.write_file(self._sandbox_id, path, data), idempotent=True
        )

    async def is_alive(self) -> bool:
        # Per the base contract this must never raise: a not-started sandbox is
        # simply not alive, and any status probe error degrades to ``False``.
        if self._client is None or self._sandbox_id is None:
            return False
        try:
            status = await self._client.get_status(self._sandbox_id)
        except Exception:
            return False
        return str(status).strip().lower() in _RUNNING_STATUSES

    def _is_timeout_error(self, exc: BaseException) -> bool:
        """Recognise client-side transport/command timeouts.

        The base covers :class:`asyncio.TimeoutError` / :class:`TimeoutError`.
        The flash-sandbox SDK raises a bespoke ``SandboxTimeoutError`` when its
        HTTP budget is exceeded; we also accept a ``SandboxHTTPError`` whose
        message mentions a timeout. Server-side command timeouts arrive as a
        normal result (handled in :meth:`_exec`), so they never reach here.
        """
        if isinstance(exc, asyncio.TimeoutError | TimeoutError):
            return True
        name = type(exc).__name__
        if name == "SandboxTimeoutError":
            return True
        return name == "SandboxHTTPError" and "timeout" in str(exc).lower()
