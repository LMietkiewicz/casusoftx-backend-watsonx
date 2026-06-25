FROM python:3.11-slim

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
# on HuggingFace. Disable Xet for the build-time download too.
ENV HF_HUB_DISABLE_XET=1
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sdadas/mmlw-retrieval-roberta-large-v2'); \
    CrossEncoder('sdadas/polish-reranker-roberta-v3')"

# App code last, so code edits don't bust the dependency/model layers.
COPY . .

EXPOSE 5000

# Waitress is already your server; backend.py calls serve() itself.
CMD ["python", "backend.py"]