import os

import joblib

from app.schemas import AnalyzeResponse
from app.traditional import RESPONSES, is_security_related, priority_for

MODEL_PATH = os.path.join("models", "ticket_classifier.joblib")
CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.40"))

class MLError(Exception):
    pass
_model = None

def _load_model():
    """Carga el modelo una sola vez (lazy) y lo reutiliza en cada petición."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise MLError("El modelo ML no está entrenado. Ejecute: python scripts/train_model.py")
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:
            raise MLError("No se pudo cargar el modelo ML. Reentrene con: python scripts/train_model.py") from None
    return _model


def analyze_ml(message: str) -> AnalyzeResponse:
    model = _load_model()

    probs = model.predict_proba([message])[0]
    best = probs.argmax()
    category = str(model.classes_[best])
    confidence = float(probs[best])

    low_confidence = confidence < CONFIDENCE_THRESHOLD

    review = category == "SECURITY" or is_security_related(message) or low_confidence

    return AnalyzeResponse(
        category=category,
        priority=priority_for(category, message),
        response=RESPONSES[category],
        requiresHumanReview=review,
        mode="ml",
    )