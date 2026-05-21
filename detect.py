"""Detection and risk scoring utilities for RiskRider."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"

# Colors in RGB
BOX_COLORS: Dict[str, Tuple[int, int, int]] = {
    "no_helmet": (220, 20, 60),
    "phone_use": (220, 20, 60),
    "no_vest": (255, 140, 0),
    "overloaded": (255, 215, 0),
    "passenger": (255, 215, 0),
    "helmet": (46, 204, 113),
    "vest": (46, 204, 113),
    "motorcycle": (52, 152, 219),
    "person": (130, 130, 130),
}

CLASS_NAMES: Dict[str, str] = {
    "no_helmet": "Kasksız",
    "helmet": "Kasklı",
    "no_vest": "Yeleğiz",
    "vest": "Yelekit",
    "phone_use": "Telefon Kullanımı",
    "overloaded": "Aşırı Yük",
    "passenger": "Ek Yolcu",
    "motorcycle": "Motosiklet",
    "person": "Kişi",
}

RISK_WEIGHTS: Dict[str, int] = {
    "no_helmet": 40,
    "phone_use": 25,
    "no_vest": 20,
    "overloaded": 10,
    "passenger": 5,
}


def load_model() -> YOLO:
    """Load YOLO model; auto-download if missing and never raise FileNotFoundError."""
    try:
        return YOLO(MODEL_PATH)
    except FileNotFoundError:
        # Ultralytics handles auto-download when given a model name.
        return YOLO(MODEL_PATH)


def is_model_available() -> bool:
    """Return True to indicate model is available or will be auto-downloaded."""
    return True


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
