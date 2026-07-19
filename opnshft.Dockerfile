# =============================================================================
# OpenShift-compatible build. Differs from the stock Dockerfile because
# OpenShift's restricted-v2 SCC runs containers as an arbitrary high UID
# with GID 0 — NOT as root, and NOT as the UID in the USER directive.
#
# Build:  docker build -f opnshft.Dockerfile -t casusoftx-backend:ocp .
# Verify: docker run --rm -u 1000650000:0 -p 5000:5000 casusoftx-backend:ocp
#
# Three changes vs the original, all load-bearing:
#   1. HF_HOME set BEFORE the model download, so models land in /opt/hf
#      instead of /root/.cache (mode 700 — unreadable to the arbitrary UID,
#      which silently breaks the "no network at startup" guarantee).
#   2. chgrp 0 + chmod g=u on everything the process reads or writes.
#   3. Numeric USER, no port below 1024.
# =============================================================================

FROM python:3.11-slim

# Must be set before anything downloads or writes.
#   HF_HOME  - where sentence-transformers caches models
#   HOME     - the arbitrary UID has no /etc/passwd entry, so ~ resolves to /
#              and anything expanding it fails. Point it somewhere writable.
ENV HF_HOME=/opt/hf \
    HF_HUB_DISABLE_XET=1 \
    HOME=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps: PyMuPDF needs no compiler on slim wheels, but build-essential
# covers any source builds; curl for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (the encoder runs on CPU; avoids the giant CUDA build).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Then the rest. Copy requirements alone first so this layer caches across
# code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the models into the image so startup needs no network and never stalls
# on HuggingFace. Lands in /opt/hf now, not /root/.cache.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sdadas/mmlw-retrieval-roberta-large-v2'); \
    CrossEncoder('sdadas/polish-reranker-roberta-v3')"

# App code last, so code edits don't bust the dependency/model layers.
COPY . .

# --- Arbitrary-UID fixup -----------------------------------------------------
# The container user is always a member of the root group (GID 0), whatever
# UID it gets assigned. So: root group owns everything, and group perms mirror
# owner perms. /opt/hf must be WRITABLE, not just readable — sentence-
# transformers takes .lock files in the cache even when the model is present.
RUN chgrp -R 0 /app /opt/hf && \
    chmod -R g=u /app /opt/hf
# -----------------------------------------------------------------------------

# Numeric, never a username. OpenShift overrides this with its own UID anyway;
# it's here so the image also behaves on plain Kubernetes and passes SCC
# validation that rejects images declaring a root user.
USER 1001

# Must be >1024. 5000 is fine.
EXPOSE 5000

# Waitress is already your server; backend.py calls serve() itself.
# NOTE: serve() must bind 0.0.0.0, not 127.0.0.1, or the Service can't reach it.
CMD ["python", "backend.py"]
