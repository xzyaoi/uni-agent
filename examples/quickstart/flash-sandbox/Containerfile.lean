# Lean verl + Harbor + Megatron image for Bristen (x86 A100, sm_80).
#
# GOAL: an install layer INDEPENDENT of source edits AND of the rllm
# checkout. The proven build (rllm/Containerfile.megatron, job 80024) did
# `COPY . /workspace/rllm`, so any source edit invalidated the ~30-min pip
# layer. The previous lean build COPY'd rllm/pyproject.toml + rllm-model-gateway/
# (400 KB) from a SEPARATE rllm checkout (RLLM_DIR). This build is fully
# SELF-CONTAINED: pyproject.toml + the empty lean_image_deps/ package live
# in this build context (examples/quickstart/flash-sandbox/), so there is no
# RLLM_DIR dependency.
#
# The local pyproject.toml is a byte-for-byte copy of rllm's [verl] +
# [harbor] extras + [tool.uv] resolution knobs (override-dependencies,
# extra-build-dependencies), with rllm + rllm-model-gateway DROPPED:
#   * nothing in uni_agent/ verl/ or the train script does `import rllm`
#     (verified); verl 0.9.0.dev + uni_agent are imported at runtime from the
#     MOUNTED repo on shared SCRATCH via PYTHONPATH (runtime_env_flash.yaml).
#   * rllm-model-gateway is a FastAPI console app our training never launches
#     (uv itself flagged it "Unnecessary" in the 80024 build). Its four deps
#     not otherwise covered (fastapi, uvicorn, httpx, renderers) are inlined
#     as base deps so the installed runtime set is identical to 80024.
# `uv pip install '.[verl,harbor]'` resolves verl[verl]+harbor[harbor] deps as
# a system (litellm[proxy]==1.83.0, numpy>=1.26.0, rich>=14.1.0, and flash-attn
# built against in-image torch) -- choices a flat -r/inline pins cannot
# express. No `-e` (nothing to edit on the runtime image) and no `uninstall`
# (the tiny lean-image-deps package is harmless and shadows nothing).
#
# LAYER ORDERING (phase reorder vs the 80184 build, job 81004's image): the
# SLOW source builds -- apex (~20min) and Transformer Engine (~30min) -- plus
# the --no-deps megatron wheels are HOISTED above the `COPY pyproject.toml`
# line, guarded by a pinned-torch layer. They depend ONLY on apt
# (setuptools/wheel/ninja) + that torch pin, NOT on pyproject.toml, so a
# [harbor]/[verl] dep bump (e.g. a verl/harbor pin bump) invalidates
# ONLY the manifest layer (~30min, mostly the flash-attn sdist) and keeps
# apex+TE+megatron cached. flash-sandbox is intentionally NOT in
# [harbor]/[verl] (harbor==0.3.0 does not depend on it -- verified on PyPI),
# so a flash-sandbox bump invalidates ONLY the thin final layer in PHASE 3
# below, never the 30-min manifest. The pinned torch MUST match what [verl,harbor]
# resolves (torch>=2.10.0 -> 2.11.0+cu130 on the cu130 index); keep it in
# sync with [verl].torch when bumping.
#
#   Hoist safety (verified against lean-image-80184.out): apex resolves 2
#   pkgs, installs ONLY itself (needs torch); TE resolves 49 pkgs, installs
#   ONLY its own 7 (onnx/onnxscript/nvdlfw/TE-cu13/TE-torch, none of which
#   are [verl,harbor] transitives); megatron-core/bridge are --no-deps. The
#   original build proved TE accepts numpy>=2.5.2, so any numpy TE picks when
#   hoisted still satisfies the manifest's numpy>=1.26.0 override. Only
#   nvidia-modelopt and flash-sandbox are kept below the manifest
#   (modelopt: large 50-pkg footprint + the rich>=14.1.0 /
#   litellm[proxy]==1.83.0 overrides; flash-sandbox: isolates its version
#   bumps from the 30-min manifest). Each installs in ~1s, so re-running
#   either on a dep bump is free.
#
# Build for Bristen A100 with:  --build-arg TORCH_CUDA_ARCH_LIST=8.0
FROM docker.io/nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
ARG TORCH_CUDA_ARCH_LIST=8.0
ARG MAX_JOBS=8
ARG TE_VERSION=2.18.0
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python-is-python3 python3-dev \
        ninja-build build-essential pkg-config git curl ca-certificates \
        libsndfile1 libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*
ENV UV_INSTALL_DIR=/usr/local/bin \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS="${MAX_JOBS}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version
WORKDIR /workspace

# ===== PHASE 1: torch + cached source builds (hoisted above manifest) ========
# These layers depend ONLY on apt (setuptools/wheel/ninja) + the pinned torch
# below -- NOT on pyproject.toml -- so a [harbor]/[verl] dep bump invalidates
# only the manifest layer (Phase 2), keeping apex+TE+megatron cached. uv pip
# install is additive: torch 2.11.0+cu130 satisfies the manifest's
# torch>=2.10.0 later, so it is NOT reinstalled/upgraded there. Keep this pin
# in sync with [verl].torch on bumps (a torch bump SHOULD rebuild apex+TE).
RUN uv pip install --break-system-packages --index-strategy unsafe-best-match \
        --index https://download.pytorch.org/whl/cu130 \
        --index https://pypi.org/simple \
        'torch==2.11.0+cu130'
# [1/5] megatron-core 0.17.1 (> 0.15.0 needed for numpy>=2.0.0). --no-deps:
# its bundled extras (incl. the cu13 TE dep) are installed explicitly below,
# so this step installs only the wheel (independent of pyproject.toml).
RUN uv pip install --no-deps megatron-core==0.17.1
# [2/5] megatron-bridge @ 691a377f (2026-05-19: "Add external trainer
# integration helpers (#3813)"). verl 0.8.0 requires LinearForLastLayer which
# only landed in this post-v0.4.2 commit. --no-deps: at runtime it uses the
# TE + megatron installed here, not its bundled 3rdparty/.
RUN uv pip install --no-deps git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@691a377f
# [3/5] NVIDIA Apex (gradient accumulation fusion for Megatron). Builds C++/CUDA
# against the pinned torch with --no-build-isolation (setuptools/wheel/ninja
# from apt). Independent of TE, so built before it -- apex stays cached if
# only TE changes later.
RUN MAX_JOBS=32 APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    uv pip install -v --no-cache --no-build-isolation \
        git+https://github.com/NVIDIA/apex.git
# [4/5] Transformer Engine 2.18.0 (cu13). The [core-cu13] extra pulls
# transformer_engine_cu13==${TE_VERSION} (PREBUILT x86_64 wheel, ~1 min, links
# libcublas.so.13); [pytorch] builds transformer_engine_torch==${TE_VERSION}
# from sdist against the pinned torch + the installed cu13 core.
# --no-build-isolation so the sdist build uses the in-image torch/cuda/headers.
RUN MAX_JOBS=64 uv pip install --no-cache --no-build-isolation \
        --index-strategy unsafe-best-match \
        --index https://download.pytorch.org/whl/cu130 \
        --index https://pypi.org/simple \
        "transformer_engine[core-cu13,pytorch]==${TE_VERSION}"

# ===== PHASE 2: manifest layer (invalidated on pyproject.toml / dep bump) ====
COPY pyproject.toml lean_image_deps/ /workspace/lean-image-deps/
RUN cd /workspace/lean-image-deps \
 && uv pip install --break-system-packages --index-strategy unsafe-best-match \
        --index https://download.pytorch.org/whl/cu130 \
        --index https://pypi.org/simple \
        '.[verl,harbor]' \
 && uv pip install --break-system-packages cupy-cuda13x==14.1.1 \
 && uv pip install --break-system-packages 'swebench==4.1.0' \
 && uv pip install --break-system-packages \
        liger-kernel latex2sympy2-extended pylatexenc math-verify \
        'TransferQueue==0.1.8' 'flash-linear-attention==0.5.2'
ENV PYTHONNOUSERSITE=1
# Verify the manifest layer imports. flash-sandbox is verified separately in
# PHASE 3 below (it now installs in its own thin final layer, NOT here, so a
# flash-sandbox bump never invalidates this expensive layer).
RUN python -c "import cupy, flash_attn, harbor, torch, verl, vllm, \
                       liger_kernel, math_verify, transfer_queue, fla; \
print('OK torch', torch.__version__, 'verl', verl.__version__, 'vllm', vllm.__version__, \
'flash_attn', flash_attn.__version__, 'cupy', cupy.__version__, 'fla', fla.__version__)"

# ===== PHASE 3: small wheels (below COPY; fast, ~1s each) ===================
# [5/6] nvidia-modelopt (quantization/optim tooling used by verl+mbridge). Kept
# below the manifest layer as a conservative choice: it resolves a large 50-pkg
# footprint (rich>=14.1.0 / litellm[proxy]==1.83.0 overrides live in the
# manifest), but installs only modelopt+pulp+scipy once those are satisfied --
# so re-running it on a dep bump costs ~1s and cannot perturb the override
# resolution that Phase 2 already pinned.
RUN uv pip install --break-system-packages 'nvidia-modelopt>=0.37.0'

# [6/6] flash-sandbox, in its OWN thin final layer (NOT in the [harbor] extra
# in pyproject.toml above) so a future version bump (0.3.0 -> 0.3.1) rebuilds
# only this ~1s layer instead of the whole Phase-2 manifest layer
# (vLLM/ray/flashinfer). harbor==0.3.0 does not declare flash-sandbox
# (verified on PyPI), so lifting the pin out of [harbor] leaves the manifest
# resolution unchanged -- the big layer stays cached across flash-sandbox
# bumps. flash-sandbox's own deps (httpx2/build/twine) not already satisfied
# by the manifest come in here; rich+typer are already satisfied by harbor,
# so this layer stays small.
RUN uv pip install --break-system-packages 'flash-sandbox==0.3.0'
# Verify flash-sandbox accepts the `metadata` kwarg on the three hot paths the
# provider drives (start_sandbox / exec_command / shell_exec). A regressed pin
# (e.g. 0.2.11, which predates metadata) fails the build here -- this is the
# whole point of the 0.2.12 bump (uni_agent/sandbox/flash_sandbox.py:
# _forward_metadata gates on _accepts_kwarg).
RUN python -c "import inspect; \
from importlib.metadata import version; \
from flash_sandbox import AsyncHTTPClient; \
print('flash-sandbox', version('flash-sandbox')); \
print('flash-sandbox metadata kwarg:', all('metadata' in \
inspect.signature(getattr(AsyncHTTPClient,m)).parameters \
for m in ('start_sandbox','exec_command','shell_exec')))"

# Verify all Megatron packages are INSTALLED (versions via metadata, NOT
# import -- megatron.core -> mbridge -> transformer_engine -> core .so eagerly
# dlopens libcuda.so.1, a HOST-DRIVER component absent from the build
# container). Full import is deferred to a runtime pre-check in run_ray_node.sh.
RUN python - <<'PY'
import torch
from importlib.metadata import version, PackageNotFoundError
def v(n):
    try: return version(n)
    except PackageNotFoundError: return "NOT FOUND"
print("OK torch", torch.__version__,
      "megatron-core", v("megatron-core"),
      "megatron-bridge", v("megatron-bridge"),
      "transformer-engine", v("transformer-engine"),
      "apex", v("apex"))
print("megatron.core/TE full import deferred to runtime (needs libcuda.so.1)")
PY
CMD ["bash"]
