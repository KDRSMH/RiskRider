"""Streamlit UI for RiskRider."""

from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import streamlit as st
from PIL import Image

from detect import (
    BOX_COLORS,
    CLASS_NAMES,
    calculate_risk_score,
    is_model_available,
    run_detection,
)

st.set_page_config(page_title="RiskRider", page_icon="🏍️", layout="wide")

CSS = """
<style>
:root {
  --bg1: #0b0f2a;
  --bg2: #1a1444;
  --bg3: #0a2a4a;
  --accent1: #7f5cff;
  --accent2: #2bd1ff;
  --accent3: #2fff9a;
  --glass: rgba(255, 255, 255, 0.08);
  --stroke: rgba(255, 255, 255, 0.15);
}

html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 700px at 10% 10%, #16225a 0%, transparent 60%),
              radial-gradient(900px 600px at 90% 0%, #0c3b5d 0%, transparent 55%),
              linear-gradient(135deg, var(--bg1), var(--bg2), var(--bg3));
  color: #f4f6ff;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { visibility: hidden; height: 0; }

.hero-title {
  font-size: 3.4rem;
  font-weight: 800;
  letter-spacing: 0.5px;
  background: linear-gradient(90deg, var(--accent1), var(--accent2), var(--accent3));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  margin-bottom: 0.2rem;
}

.hero-subtitle {
  font-size: 1.1rem;
  opacity: 0.85;
  margin-bottom: 1.5rem;
}

.section-card {
  background: var(--glass);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.25);
}

.metric-card {
  background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04));
  border: 1px solid var(--stroke);
  border-radius: 14px;
  padding: 14px 16px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 22px rgba(0,0,0,0.25);
}

.score-box {
  background: linear-gradient(135deg, rgba(127,92,255,0.25), rgba(43,209,255,0.2));
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.score-value {
  font-size: 3rem;
  font-weight: 800;
}

.risk-factor-row {
  border-left: 5px solid #ff5c7a;
  background: rgba(0,0,0,0.2);
  padding: 10px 12px;
  border-radius: 10px;
  margin-bottom: 8px;
}

.det-row {
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--stroke);
  border-radius: 10px;
  padding: 8px 10px;
  margin-bottom: 6px;
  display: flex;
  justify-content: space-between;
}

.footer {
  opacity: 0.7;
  text-align: center;
  margin-top: 30px;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="hero-title">🏍️ RiskRider</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Yapay Zeka Destekli Motosiklet Sürücü Risk Analiz Sistemi</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Ayarlar")
    confidence = st.slider("Güven Skoru (Confidence)", 0.10, 0.90, 0.40, 0.05)

    st.markdown("---")
    st.subheader("Renk Kodu Rehberi")
    for key, color in BOX_COLORS.items():
        label = CLASS_NAMES.get(key, key)
        color_css = f"rgb({color[0]}, {color[1]}, {color[2]})"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<div style='width:12px;height:12px;border-radius:50%;background:{color_css};'></div>"
            f"<div>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Model Durumu")
    if is_model_available():
        st.success("Model hazır / otomatik indirilebilir")
    else:
        st.error("Model bulunamadı")

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Görüntü Yükle")
    upload = st.file_uploader("Bir motosiklet sürücüsü görüntüsü yükleyin", type=["jpg", "jpeg", "png"])
    original_image = None
    if upload is not None:
        original_image = Image.open(io.BytesIO(upload.read())).convert("RGB")
        st.image(original_image, caption="Orijinal Görüntü", use_container_width=True)

    analyze = st.button("Analiz Et", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Tespit Sonucu")
    if "annotated" in st.session_state:
        st.image(st.session_state["annotated"], caption="Tespitli Görüntü", use_container_width=True)
    else:
        st.info("Analiz sonrası tespitli görüntü burada görünecek.")
    st.markdown("</div>", unsafe_allow_html=True)

if analyze and original_image is not None:
    annotated, detections = run_detection(original_image, confidence)
    st.session_state["annotated"] = annotated
    st.session_state["detections"] = detections
    score, level, triggered = calculate_risk_score(detections)
    st.session_state["score"] = score
    st.session_state["level"] = level
    st.session_state["triggered"] = triggered

if "detections" in st.session_state:
    detections: List[Dict] = st.session_state["detections"]
    score = st.session_state["score"]
    level = st.session_state["level"]
    triggered = st.session_state["triggered"]

    st.markdown("## Analiz Sonuçları")

    st.markdown(
        f"<div class='score-box'><div><div style='opacity:0.8'>Risk Skoru</div>"
        f"<div class='score-value'>{score}</div></div>"
        f"<div style='font-size:1.2rem;font-weight:700'>{level}</div></div>",
        unsafe_allow_html=True,
    )
    st.progress(score / 100.0)

    st.markdown("### Tetiklenen Risk Faktörleri")
    if triggered:
        emoji_map = {
            "no_helmet": "⛔",
            "phone_use": "📵",
            "no_vest": "🦺",
            "overloaded": "📦",
            "passenger": "👥",
        }
        for item in triggered:
            emoji = emoji_map.get(item["class_name"], "⚠️")
            st.markdown(
                f"<div class='risk-factor-row'>{emoji} <b>{item['label']}</b>"
                f" — {item['delta']} puan</div>",
                unsafe_allow_html=True,
            )
    else:
        st.success("Risk faktörü tespit edilmedi.")

    st.markdown("### Güvenlik Metrikleri")
    counts = {"helmet": 0, "no_helmet": 0, "vest": 0, "no_vest": 0, "phone_use": 0}
    for det in detections:
        key = det.get("class_name")
        if key in counts:
            counts[key] += 1

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(
        f"<div class='metric-card'><div>Kasklı</div><div style='font-size:1.6rem'>{counts['helmet']}</div></div>",
        unsafe_allow_html=True,
    )
    m2.markdown(
        f"<div class='metric-card'><div>Kasksız</div><div style='font-size:1.6rem'>{counts['no_helmet']}</div></div>",
        unsafe_allow_html=True,
    )
    m3.markdown(
        f"<div class='metric-card'><div>Yelekit</div><div style='font-size:1.6rem'>{counts['vest']}</div></div>",
        unsafe_allow_html=True,
    )
    m4.markdown(
        f"<div class='metric-card'><div>Yeleğiz</div><div style='font-size:1.6rem'>{counts['no_vest']}</div></div>",
        unsafe_allow_html=True,
    )
    m5.markdown(
        f"<div class='metric-card'><div>Telefon Kullanan</div><div style='font-size:1.6rem'>{counts['phone_use']}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Bireysel Tespitler")
    if detections:
        for det in detections:
            class_name = det["class_name"]
            display_name = det["display_name"]
            conf = det["confidence"]
            color = BOX_COLORS.get(class_name, (200, 200, 200))
            color_css = f"rgb({color[0]}, {color[1]}, {color[2]})"
            st.markdown(
                f"<div class='det-row' style='border-left:5px solid {color_css};'>"
                f"<div>{display_name}</div><div>%{conf*100:.1f}</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Hiç tespit bulunamadı.")

st.markdown(
    '<div class="footer">RiskRider | YOLOv11 · Streamlit | Motosiklet Güvenliği İçin Yapay Zeka</div>',
    unsafe_allow_html=True,
)
