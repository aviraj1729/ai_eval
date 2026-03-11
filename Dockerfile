# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps needed to compile wheels (libgomp for ONNX, libgl for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .

# Install into an isolated prefix so we can copy only the site-packages
RUN pip install --no-cache-dir --prefix=/install/pkg -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.description="Trust-score API (lightweight)"

# Runtime-only native libs
# libgl1       → required by cv2 (even the headless wheel links against it)
# libgomp1     → required by ONNX Runtime
# libglib2.0-0 → required by cv2 / gthread
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install/pkg /usr/local

WORKDIR /app
COPY main.py .

# ── Model pre-download at build time ──────────────────────────────────────────
# buffalo_sc is fetched on first FaceAnalysis.prepare(); baking it into the
# image avoids a cold-start download and makes the container self-contained.
RUN python - <<'EOF'
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1)
print("buffalo_sc downloaded and cached.")
EOF

# Pre-cache RapidOCR models (downloaded on first use otherwise)
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()(None)"  || true

# Pre-cache FER model weights
RUN python -c "from fer import FER; FER(mtcnn=False)" || true

# ── Runtime config ─────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    # Prevent ONNX Runtime from spawning more threads than cores
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

EXPOSE 8000

# Use 1 worker per container; scale horizontally with replicas
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "uvloop", "--http", "h11"]
