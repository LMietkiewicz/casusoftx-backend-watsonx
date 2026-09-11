# =============================================================================
# CasuSoftX backend — OpenShift-compatible build, s390x (LinuxONE).
#
# Base image is IBM's s390x PyTorch build. PyPI publishes no torch for s390x,
# but IBM does, with optional Telum (z16+) acceleration via zDNN. The image is
# RHEL 10, which no longer packages LibreOffice — hence the converter sidecar
# (see converter.Dockerfile); this image holds no LibreOffice at all.
#
# OpenShift's restricted-v2 SCC runs containers as an arbitrary high UID with
# GID 0 — NOT root, and NOT the UID in the USER directive. Hence the group
# ownership fixup and the writable HOME.
#
# Verified against the image (2026-09), emulated s390x:
#   OS                RHEL 10.2 (Coughlan)
#   Python            3.12.13 (system python; torch in /usr/local/lib64)
#   torch             2.11.0+cpu
#   preinstalled      numpy 2.5.1, transformers 5.14.1
#   repos enabled     ubi-10 BaseOS + AppStream, CodeReady Builder, EPEL
#   missing           libpq (hence postgresql-libs below)
#   default user      NON-ROOT (hence USER root before dnf)
#
# Build:  docker build -f opnshft.Dockerfile -t casusoftx-backend:ocp .
# =============================================================================

FROM icr.io/ibmz/ibmz-accelerated-for-pytorch:1.5.0

# HF_HOME  - model cache location. /root/.cache is mode 700 and unreadable to
#            the arbitrary UID, which would silently break the offline guarantee.
# HOME     - the arbitrary UID has no /etc/passwd entry, so ~ resolves to /.
# The offline flags are deliberately NOT here: they would block the model
# download below. They are set immediately after it.
ENV HF_HOME=/opt/hf \
    HF_HUB_DISABLE_XET=1 \
    HOME=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .

# The base image runs as a NON-ROOT user, so dnf refuses with "This command has
# to be run with superuser privileges". USER 1001 at the bottom switches back.
USER root

# Package names verified against this image with `dnf --downloadonly`: all
# resolve (32 packages, 206 MB) from the enabled repos, which include
# CodeReady Builder — where the -devel packages live.
#
# Still compiled from source (no s390x wheel on PyPI, ever): scipy,
# scikit-learn, pandas, cryptography (needs Rust — its sdist declares maturin),
# grpcio, markupsafe. numpy and transformers are ALREADY in the base image and
# are left alone.
#
# The toolchain is installed, used, and removed in ONE layer so its 206 MB does
# not ship. postgresql-libs and curl are installed AFTER the removal, so dnf
# cannot take them out as dependencies of something being removed.
#
# pdfplumber is installed --no-deps first: it declares Pillow>=12.2.0, which has
# no s390x wheel either, but Pillow is imported lazily and only inside
# to_image(), which this codebase never calls (verified with an import blocker:
# text and table extraction both run with PIL absent). pdfminer.six and
# pypdfium2 (its real dependencies) are pinned explicitly in requirements.txt.
#
# torch is NOT in requirements.txt: the base image supplies 2.11.0+cpu and pip
# leaves it alone because it satisfies sentence-transformers' torch>=1.11.0.
#
# NOTE: python3-devel pulls python3 3.12.14, a patch above the image's 3.12.13.
# Same minor version, so ABI-compatible for the extensions built here.
RUN dnf install -y --setopt=install_weak_deps=False \
        gcc gcc-c++ gcc-gfortran make python3-devel \
        openssl-devel openblas-devel rust cargo \
 && pip install --no-cache-dir --no-deps pdfplumber==0.11.10 \
 && pip install --no-cache-dir -r requirements.txt \
 && dnf remove -y gcc gcc-c++ gcc-gfortran make python3-devel \
        openssl-devel openblas-devel rust cargo \
 && dnf install -y --setopt=install_weak_deps=False curl postgresql-libs \
 && dnf clean all && rm -rf /var/cache/dnf /root/.cargo

# Fail the build if pip replaced the vendor torch with a PyPI build.
RUN python3 -c "import torch; \
    assert not torch.version.cuda, f'non-IBM torch: {torch.__version__}'; \
    print('torch OK:', torch.__version__, torch.__file__)"

# Fail the build if the BINARY psycopg got installed — psycopg-binary has no
# s390x wheel, so its presence means psycopg[binary] crept into requirements.
# The pure-Python implementation loads libpq at import, hence postgresql-libs.
RUN python3 -c "import psycopg; \
    assert psycopg.pq.__impl__ == 'python', f'unexpected impl: {psycopg.pq.__impl__}'; \
    print('psycopg', psycopg.__version__, 'impl', psycopg.pq.__impl__, \
          'libpq', psycopg.pq.version())"

# Fail the build if any direct dependency is missing or broken. Runs while the
# toolchain is already gone, so it also proves nothing needed it at runtime.
RUN python3 -c "import flask, waitress, requests, pypdfium2, pdfplumber, \
    pdfminer, unoserver.client, sentence_transformers, psycopg, psycopg_pool, \
    ibm_watsonx_ai, pydantic, cryptography, dotenv; \
    print('all direct imports OK')"

# Bake the models in so startup needs no network.
RUN python3 -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sdadas/mmlw-retrieval-roberta-large-v2'); \
    CrossEncoder('sdadas/polish-reranker-roberta-v3')"

# Models are baked. From here on, forbid HuggingFace network access: a cached
# model still triggers an update check at load time, which blocks indefinitely
# on a network that accepts connections without answering (measured: >180s hang
# vs 0.13s with these set). The pod would fail its probes and restart-loop.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Can only pass offline — turns "no network at startup" from an intention into
# a build-time check.
RUN python3 -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sdadas/mmlw-retrieval-roberta-large-v2'); \
    CrossEncoder('sdadas/polish-reranker-roberta-v3'); \
    print('models resolved offline')"

# App code last, so code edits don't bust the dependency/model layers.
COPY . .

# --- Arbitrary-UID fixup -----------------------------------------------------
# The container user is always a member of the root group (GID 0), whatever UID
# it gets. /opt/hf must be WRITABLE, not just readable — sentence-transformers
# takes .lock files in the cache even when the model is already present.
RUN chgrp -R 0 /app /opt/hf && \
    chmod -R g=u /app /opt/hf
# -----------------------------------------------------------------------------

# Numeric, never a username. OpenShift overrides this with its own UID; it is
# here so the image also behaves on plain Kubernetes and passes SCC validation
# that rejects images declaring a root user.
USER 1001

# Must be >1024.
EXPOSE 5000

# Waitress is already the server; backend.py calls serve() itself.
# NOTE: serve() must bind 0.0.0.0, not 127.0.0.1, or the Service can't reach it.
CMD ["python3", "backend.py"]
