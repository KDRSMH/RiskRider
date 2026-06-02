"""Detection and risk scoring utilities for RiskRider."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "best.pt"

# Colors in RGB – keys must match model output names exactly
BOX_COLORS: Dict[str, Tuple[int, int, int]] = {
    "A_helmet_not_worn": (220, 20, 60),   # Kırmızı
    "B_helmet_worn": (46, 204, 113),      # Yeşil
    "C_Motorcycle": (52, 152, 219),       # Mavi
    "D_person": (255, 215, 0),            # Sarı
}

CLASS_NAMES: Dict[str, str] = {
    "A_helmet_not_worn": "Kasksız",
    "B_helmet_worn": "Kasklı",
    "C_Motorcycle": "Motosiklet",
    "D_person": "Kişi",
}

RISK_WEIGHTS: Dict[str, int] = {
    "A_helmet_not_worn": 40,
}


_model: YOLO | None = None


def load_model() -> YOLO:
    """Load custom YOLO model (cached — only loaded from disk once)."""
    global _model
    if _model is not None:
        return _model
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_PATH}. "
            "Lütfen eğitilmiş 'best.pt' dosyasını proje kök dizinine koyun."
        )
    _model = YOLO(MODEL_PATH)
    return _model


def is_model_available() -> bool:
    """Return True only if the model file physically exists on disk."""
    return Path(MODEL_PATH).exists()


def _safe_conf(conf: float) -> float:
    return float(min(max(conf, 0.0), 1.0))


def run_detection(image: Image.Image, confidence: float) -> Tuple[Image.Image, List[Dict]]:
    """Run YOLO detection and return annotated image plus raw detections list."""
    model = load_model()
    conf = _safe_conf(confidence)

    results = model.predict(image, conf=conf, verbose=False)
    if not results:
        return image.copy(), []

    result = results[0]
    plotted = result.plot()
    if plotted is None:
        return image.copy(), []

    # Ultralytics returns BGR; convert to RGB for PIL.
    rgb = plotted[:, :, ::-1]
    annotated = Image.fromarray(rgb)

    detections: List[Dict] = []
    names = result.names
    if result.boxes is None:
        return annotated, detections

    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        class_name = names.get(cls_id, str(cls_id))
        display_name = CLASS_NAMES.get(class_name, class_name)
        conf_score = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()
        detections.append(
            {
                "class_name": class_name,
                "display_name": display_name,
                "confidence": conf_score,
                "box": xyxy,
            }
        )

    return annotated, detections


def calculate_risk_score(
    detections: List[Dict],
) -> Tuple[int, str, List[Dict]]:
    """Calculate risk score, level label, and triggered factors from detections."""
    score = 100
    triggered: List[Dict] = []
    seen = set()

    for det in detections:
        class_name = det.get("class_name")
        if class_name in RISK_WEIGHTS and class_name not in seen:
            seen.add(class_name)
            delta = RISK_WEIGHTS[class_name]
            score -= delta
            triggered.append(
                {
                    "class_name": class_name,
                    "label": CLASS_NAMES.get(class_name, class_name),
                    "delta": -delta,
                }
            )

    score = max(0, score)

    if score >= 80:
        level = "Düşük Risk 🟢"
    elif score >= 50:
        level = "Orta Risk 🟡"
    elif score >= 20:
        level = "Yüksek Risk 🟠"
    else:
        level = "Kritik Risk 🔴"

    return score, level, triggered
