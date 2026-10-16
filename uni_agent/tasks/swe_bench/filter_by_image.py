# ruff: noqa: E501
"""Filter a SWE-bench dataset to instances whose sandbox image is available.

Each row carries its canonical open-source image ref
(``swebench/sweb.eval.x86_64.<id>``) in ``extra_info.tools_kwargs.task.sandbox.image``;
at run time the flash-sandbox provider rewrites that bare Docker Hub ref against
:envvar:`FLASH_SANDBOX_IMAGE_REGISTRY` (default ``ghcr.io/swe-images``) via
``FlashSandboxSandbox._resolve_image``. This filter resolves each row the *same*
way and HEADs the resulting OCI/Docker manifest on the registry.

Public SWE-bench images need only an anonymous pull token, so this works without
registry credentials. A row whose manifest returns non-2xx (or errors) is dropped
and its ``instance_id`` is reported on stderr; the survivors are written to a new
parquet that ``TRAIN_FILE``/``TEST_FILE`` can point straight at, so the agent
loop never wastes an episode budget on a sandbox it cannot start.

Example::

    python -m uni_agent.tasks.swe_bench.filter_by_image \
        --input  data/uni_agent/swe_bench_verified.parquet \
        --output data/uni_agent/swe_bench_verified_available.parquet \
        --registry ghcr.io/swe-images --concurrency 24
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from uni_agent.sandbox.flash_sandbox import FlashSandboxSandbox


# Mirrors uni_agent.tasks.swe_bench.preprocess.get_image_name. Inlined (rather
# than imported) so this filter stays dependency-light: preprocess pulls in the
# `datasets` library at module import, and `get_image_name` is a one-liner.
def get_image_name(instance_id: str) -> str:
    """Canonical open-source SWE-bench image ref for ``instance_id``."""
    return f"swebench/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}"

# OCI index (multi-arch SWE images) + the single-manifest types, so a HEAD
# succeeds regardless of whether the registry stored an index or a manifest.
_MANIFEST_ACCEPT = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.v2+json"
)


def resolve_image(image: str, *, registry: str | None = None) -> str:
    """Resolve a row's image ref the way the flash-sandbox provider does.

    Mirrors :meth:`FlashSandboxSandbox._resolve_image` but lets the caller
    override ``FLASH_SANDBOX_IMAGE_REGISTRY`` (e.g. to inspect a different
    mirror without exporting an env var).
    """
    if registry is not None:
        old = os.environ.get("FLASH_SANDBOX_IMAGE_REGISTRY")
        os.environ["FLASH_SANDBOX_IMAGE_REGISTRY"] = registry
        try:
            return FlashSandboxSandbox._resolve_image(image)
        finally:
            if old is None:
                os.environ.pop("FLASH_SANDBOX_IMAGE_REGISTRY", None)
            else:
                os.environ["FLASH_SANDBOX_IMAGE_REGISTRY"] = old
    return FlashSandboxSandbox._resolve_image(image)


@dataclass(frozen=True)
class ImageProbe:
    instance_id: str
    ref: str        # already-resolved "host/path[:tag]"
    status: str     # "HTTP_200" / "HTTP_404" / "token_err:..." / ...
    available: bool


def _split_ref(ref: str) -> tuple[str, str]:
    """Split a resolved ref into ``(registry_repo, tag)`` for the v2 API."""
    # Strip any leading "http(s)://" that _resolve_image never adds; the
    # flash-sandbox registry form is "host/namespace/name[:tag]".
    name, sep, tag = ref.rpartition(":")
    if not sep or "/" in tag:
        # No tag (or the part after ':' is a port, e.g. localhost:5000/x),
        # fall back to the default tag the run uses.
        registry_repo = ref
        tag = "latest"
    else:
        registry_repo = name
    return registry_repo, tag


def probe_image(instance_id: str, image: str, *, registry: str, timeout: float) -> ImageProbe:
    """HEAD the image manifest on ``registry`` (anonymous pull token)."""
    ref = resolve_image(image, registry=registry)
    repo, tag = _split_ref(ref)
    # Registry host is everything before the first '/'; repo is the rest.
    host = repo.split("/", 1)[0]
    repo_path = repo.split("/", 1)[1] if "/" in repo else repo
    token_url = f"https://{host}/token?scope=repository:{repo_path}:pull&service={host}"
    try:
        with urllib.request.urlopen(token_url, timeout=timeout) as resp:
            token = json.load(resp)["token"]
    except urllib.error.HTTPError as exc:
        return ImageProbe(instance_id, ref, f"token_HTTP_{exc.code}", False)
    except Exception as exc:  # noqa: BLE001 -- surface a short reason
        return ImageProbe(instance_id, ref, f"token_err:{type(exc).__name__}", False)

    manifest_url = f"https://{host}/v2/{repo_path}/manifests/{tag}"
    req = urllib.request.Request(
        manifest_url,
        method="HEAD",
        headers={"Authorization": f"Bearer {token}", "Accept": _MANIFEST_ACCEPT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = f"HTTP_{resp.status}"
            return ImageProbe(instance_id, ref, status, 200 <= resp.status < 300)
    except urllib.error.HTTPError as exc:
        return ImageProbe(instance_id, ref, f"HTTP_{exc.code}", 200 <= exc.code < 300)
    except Exception as exc:  # noqa: BLE001
        return ImageProbe(instance_id, ref, f"err:{type(exc).__name__}", False)


def _row_image(extra_info: dict) -> tuple[str, str]:
    """Extract ``(instance_id, image)`` from a row's ``extra_info``."""
    task = (extra_info or {}).get("tools_kwargs", {}).get("task", {})
    image = (task.get("sandbox") or {}).get("image")
    instance_id = (task.get("metadata") or {}).get("instance_id")
    if not image and instance_id:
        # Defensive: reconstruct the canonical ref if a row omitted ``image``
        # but kept the metadata (matches build_swe_bench_verified).
        image = get_image_name(instance_id)
    return instance_id, image


def filter_by_image(
    input_path: str,
    output_path: str,
    *,
    registry: str,
    concurrency: int = 24,
    timeout: float = 40.0,
) -> tuple[int, int]:
    """Write a parquet containing only rows whose image is available.

    Returns ``(kept, dropped)``.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(input_path)
    extra_infos = table.column("extra_info").to_pylist()
    keep_mask = [True] * len(extra_infos)

    targets: list[tuple[int, str, str]] = []  # (row, instance_id, image)
    for i, ei in enumerate(extra_infos):
        instance_id, image = _row_image(ei)
        if not image:
            keep_mask[i] = False  # nothing to resolve -> drop
            continue
        targets.append((i, instance_id, image))

    probes: list[ImageProbe] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(probe_image, iid, img, registry=registry, timeout=timeout)
                   for _, iid, img in targets]
        for (row, iid, _img), fut in zip(targets, futures, strict=True):
            probes.append(fut.result())

    dropped: list[ImageProbe] = []
    for (row, iid, _img), probe in zip(targets, probes, strict=True):
        if not probe.available:
            keep_mask[row] = False
            dropped.append(probe)

    kept = sum(keep_mask)
    filtered = table.filter(keep_mask)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        pq.write_table(filtered, output_path)

    total = len(extra_infos)
    print(f"input:  {total} rows from {input_path}", flush=True)
    print(f"registry: {registry}", flush=True)
    print(f"kept:   {kept} rows -> {output_path or '(no output)'}", flush=True)
    print(f"dropped: {total - kept} rows (image not available)", flush=True)
    if dropped:
        # Group by reason for a concise report.
        from collections import Counter
        reasons = Counter(p.status for p in dropped)
        for reason, n in reasons.most_common():
            print(f"  {reason}: {n}", flush=True)
        for p in dropped[:50]:
            print(f"  - {p.instance_id}  [{p.ref}]  {p.status}", flush=True)
        if len(dropped) > 50:
            print(f"  ... and {len(dropped) - 50} more", flush=True)
    return kept, total - kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Input SWE-bench parquet.")
    parser.add_argument("--output", required=True, help="Output (filtered) parquet.")
    parser.add_argument(
        "--registry",
        default=os.getenv("FLASH_SANDBOX_IMAGE_REGISTRY", "ghcr.io/swe-images"),
        help="Registry the flash-sandbox provider resolves images against "
        "(default: $FLASH_SANDBOX_IMAGE_REGISTRY or ghcr.io/swe-images).",
    )
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=40.0, help="Per-request timeout (s).")
    args = parser.parse_args()

    filter_by_image(
        args.input,
        args.output,
        registry=args.registry,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
