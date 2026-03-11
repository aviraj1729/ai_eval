# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/pkg -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.description="Trust-score API (lightweight)"

# libgl1       → cv2 links against it even in the headless wheel
# libgomp1     → ONNX Runtime threading
# libglib2.0-0 → cv2 / gthread
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install/pkg /usr/local

WORKDIR /app
COPY main.py .

# ── Bake model weights into the image at build time ───────────────────────────
# InsightFace buffalo_sc (~100 MB)
RUN python -c "\
from insightface.app import FaceAnalysis; \
fa = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']); \
fa.prepare(ctx_id=-1); \
print('buffalo_sc ready.')"

# RapidOCR ONNX models
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" || true

# DeepFace emotion model — triggers download of the FER+ weights (~80 MB).
# We use enforce_detection=False + a blank image so it won't fail on no-face.
RUN python -c "\
import numpy as np; \
from deepface import DeepFace; \
blank = np.zeros((48,48,3), dtype=np.uint8); \
DeepFace.analyze(img_path=blank, actions=['emotion'], \
    enforce_detection=False, detector_backend='opencv', silent=True); \
print('DeepFace emotion model ready.')" || true

# ── Runtime config ─────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "uvloop", "--http", "h11"]
