import math
import re

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from insightface.app import FaceAnalysis
from pydantic import BaseModel
from rapidocr_onnxruntime import RapidOCR

app = FastAPI()

# ---------------------------------------------------------------------------
# Model startup — loaded once, reused across requests
# ---------------------------------------------------------------------------
# buffalo_sc = MobileNet backbone (~100 MB) vs buffalo_l ResNet-100 (~1.2 GB)
# Accuracy delta on real-world thumbnails is < 3% for cosine similarity.
face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1)

# RapidOCR: pure ONNX, no PyTorch/TF, supports ₹ via its det+rec pipeline
ocr_engine = RapidOCR()

# Lazy-load FER so the import doesn't slow cold start when mood isn't needed
_fer_detector = None


def _get_fer():
    global _fer_detector
    if _fer_detector is None:
        from fer import FER  # pip install fer  (ONNX-based, no TF required)

        _fer_detector = FER(mtcnn=False)  # mtcnn=False → OpenCV detector, faster
    return _fer_detector


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class TrustRequest(BaseModel):
    stream_image_url: str
    thumbnail_image_url: str
    metadata_text: str | list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Reuse a single TCP session across all image downloads
_http_session = requests.Session()
_http_session.headers.update({"User-Agent": "trust-score-service/2.0"})


def clean_score(score):
    if score is None or math.isnan(score) or math.isinf(score):
        return 0.0
    return float(score)


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_image(url: str) -> np.ndarray:
    r = _http_session.get(url, timeout=10)
    r.raise_for_status()
    img_array = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from URL.")
    return img


# ---------------------------------------------------------------------------
# Face similarity  (InsightFace buffalo_sc)
# ---------------------------------------------------------------------------


def _get_face_embedding(img: np.ndarray):
    faces = face_app.get(img)
    return faces[0].embedding if faces else None


def face_similarity(stream_img: np.ndarray, thumb_img: np.ndarray) -> float:
    emb1 = _get_face_embedding(stream_img)
    emb2 = _get_face_embedding(thumb_img)
    if emb1 is None or emb2 is None:
        return 0.0
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# OCR + contact extraction  (RapidOCR → no EasyOCR / no PyTorch)
# ---------------------------------------------------------------------------

_RUPEE_ALIASES = re.compile(r"\b(Rs\.?|INR|\bR\b)")
_OCR_FIXES = str.maketrans(
    {
        "I": "1",
        "l": "1",
        "!": "1",
        "|": "1",
        "O": "0",
        "o": "0",
        "S": "5",
        "s": "5",
        "B": "8",
        "G": "6",
        "Z": "2",
        "z": "2",
    }
)


def extract_text(img: np.ndarray) -> str:
    result, _ = ocr_engine(img)
    if not result:
        return ""
    raw = " ".join(r[1] for r in result)
    return _RUPEE_ALIASES.sub("₹", raw)


def _normalize_ocr_text(text: str) -> str:
    digit_like = set("0123456789IlOoSsBbGgZz!|")
    tokens = text.split()
    fixed = []
    for token in tokens:
        core = re.sub(r"[^\w!|]", "", token)
        if not core:
            fixed.append(token)
            continue
        if sum(1 for c in core if c in digit_like) / len(core) >= 0.8:
            fixed.append(token.translate(_OCR_FIXES))
        else:
            fixed.append(token)
    return " ".join(fixed)


def extract_contacts(text: str) -> dict:
    cleaned = _normalize_ocr_text(text)
    raw_matches = re.findall(r"(?<!\d)[\d][\d\s\-\.]{7,13}[\d](?!\d)", cleaned)
    valid_phones: set[str] = set()
    for raw in raw_matches:
        digits = re.sub(r"\D", "", raw)
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        if len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]
        if len(digits) in (9, 10):
            valid_phones.add(digits)
    return {"phone_numbers": sorted(valid_phones)}


# ---------------------------------------------------------------------------
# Vibrancy  (unchanged — pure NumPy/OpenCV, already lightweight)
# ---------------------------------------------------------------------------


def vibrancy_score(img: np.ndarray) -> tuple[float, dict]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_saturation = float(np.mean(s)) / 255.0
    mean_brightness = float(np.mean(v)) / 255.0

    if mean_brightness < 0.2:
        brightness_penalty = mean_brightness / 0.2
    elif mean_brightness > 0.95:
        brightness_penalty = (1.0 - mean_brightness) / 0.05
    else:
        brightness_penalty = 1.0

    score = mean_saturation * brightness_penalty
    return clean_score(score), {
        "mean_saturation": round(mean_saturation, 3),
        "mean_brightness": round(mean_brightness, 3),
        "brightness_penalty": round(brightness_penalty, 3),
    }


# ---------------------------------------------------------------------------
# Mood  (FER ONNX — replaces DeepFace / TensorFlow)
# ---------------------------------------------------------------------------

_WARM_WEIGHTS = {
    "happy": 1.0,
    "surprise": 0.4,
    "neutral": 0.1,
    "fear": -0.3,
    "sad": -0.5,
    "disgust": -0.8,
    "angry": -0.8,
}


def _warm_color_mood(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    warm_mask = ((h >= 0) & (h <= 35)) | ((h >= 160) & (h <= 179))
    return min(float(np.sum(warm_mask)) / float(h.size) / 0.5, 1.0)


def _mood_label(score: float, fallback: bool) -> str:
    prefix = "(approx) " if fallback else ""
    if score >= 0.75:
        return f"{prefix}Happy 😄"
    elif score >= 0.55:
        return f"{prefix}Neutral-Positive 🙂"
    elif score >= 0.40:
        return f"{prefix}Neutral 😐"
    return f"{prefix}Unhappy / Serious 😞"


def mood_score(img: np.ndarray) -> tuple[float, str, list]:
    try:
        detector = _get_fer()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detect_emotions(rgb)  # list of dicts

        if not detections:
            raise ValueError("FER: no faces detected")

        face_scores, breakdown = [], []
        for det in detections:
            emotions = det.get("emotions", {})
            if not emotions:
                continue
            total = sum(emotions.values()) or 1.0
            probs = {k: v / total for k, v in emotions.items()}
            raw = sum(_WARM_WEIGHTS.get(e, 0.0) * p for e, p in probs.items())
            score = float((np.clip(raw, -1.0, 1.0) + 1.0) / 2.0)
            face_scores.append(score)
            dominant = max(emotions, key=emotions.get)
            breakdown.append(
                {
                    "dominant_emotion": dominant,
                    "emotions": {k: round(float(v), 1) for k, v in emotions.items()},
                    "face_mood_score": round(score, 3),
                }
            )

        if not face_scores:
            raise ValueError("FER: no usable emotion data")

        final = float(np.mean(face_scores))
        return clean_score(final), _mood_label(final, fallback=False), breakdown

    except Exception as exc:
        score = _warm_color_mood(img)
        return (
            clean_score(score),
            _mood_label(score, fallback=True),
            [{"fallback_reason": str(exc)}],
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/trust-score")
def calculate_trust_score(data: TrustRequest):
    try:
        stream_img = load_image(data.stream_image_url)
        thumb_img = load_image(data.thumbnail_image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading images: {e}")

    # Scores
    face_sc = clean_score(face_similarity(stream_img, thumb_img))

    thumb_text = extract_text(thumb_img)
    contacts = extract_contacts(thumb_text)
    extracted_set = set(contacts["phone_numbers"])

    if isinstance(data.metadata_text, str):
        provided_numbers = [re.sub(r"\D", "", data.metadata_text.strip())]
    else:
        provided_numbers = [re.sub(r"\D", "", n.strip()) for n in data.metadata_text]
    valid_provided = [n for n in provided_numbers if len(n) in (9, 10)]

    if not valid_provided:
        contact_sc, matched_contacts, missing_contacts = 0.0, [], []
    else:
        matched_contacts = [n for n in valid_provided if n in extracted_set]
        missing_contacts = [n for n in valid_provided if n not in extracted_set]
        contact_sc = clean_score(len(matched_contacts) / len(valid_provided))

    vib_sc, vibrancy_details = vibrancy_score(thumb_img)
    vib_sc = clean_score(vib_sc)

    mood_sc, mood_label, emotion_breakdown = mood_score(thumb_img)
    mood_sc = clean_score(mood_sc)

    # Weighted trust score (face=45%, contact=30%, mood=15%, vibrancy=10%)
    trust_sc = 0.45 * face_sc + 0.30 * contact_sc + 0.15 * mood_sc + 0.10 * vib_sc

    return sanitize(
        {
            "face_similarity": f"{round(face_sc, 3) * 100}%",
            "thumbnail_text": thumb_text,
            "contact_similarity": f"{round(contact_sc, 3) * 100}%",
            "extracted_contacts": contacts,
            "matched_contacts": matched_contacts,
            "missing_contacts": missing_contacts,
            "vibrancy_score": f"{round(vib_sc, 3) * 100}%",
            "vibrancy_details": vibrancy_details,
            "mood_score": f"{round(mood_sc, 3) * 100}%",
            "mood_label": mood_label,
            "emotion_breakdown": emotion_breakdown,
            "trust_score": f"{round(clean_score(trust_sc), 3) * 100}%",
        }
    )
