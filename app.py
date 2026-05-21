"""Streamlit UI for RiskRider."""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image

from detect import (
    BOX_COLORS,
    CLASS_NAMES,
    calculate_risk_score,
    is_model_available,
    run_detection,
)

st.set_page_config(page_title="RiskRider", page_icon="R", layout="wide")

CSS = """
<style>
:root {
  --bg: #0a0f1e;
  --card: #111827;
  --border: #1f2937;
  --accent: #3b82f6;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --text: #f9fafb;
  --muted: #6b7280;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, Segoe UI, Arial, sans-serif;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { visibility: hidden; height: 0; }

.top-accent {
  height: 2px;
  background: var(--accent);
  margin: -1rem -1rem 1.2rem -1rem;
}

.header-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 0.6rem;
}

.title {
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: 0.2px;
  color: var(--text);
}

.subtitle {
  font-size: 0.95rem;
  color: var(--muted);
  margin-top: 0.2rem;
}

.version {
  font-size: 0.75rem;
  color: var(--muted);
  border: 1px solid var(--border);
  padding: 4px 8px;
  border-radius: 6px;
}

.section-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  box-shadow: none;
}

.section-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.6rem;
}

.upload-hint {
  border: 1px dashed var(--text);
  border-color: var(--border);
  padding: 16px;
  border-radius: 8px;
  text-align: center;
  color: var(--text);
}

.upload-subtext {
  color: var(--muted);
  font-size: 0.8rem;
  margin-top: 4px;
}

.button-primary button {
  background: var(--accent) !important;
  color: #ffffff !important;
  border: 1px solid var(--accent) !important;
  border-radius: 8px !important;
  transition: all 0.15s ease;
}

.button-primary button:hover {
  filter: brightness(1.1);
}

.waiting-panel {
  border: 1px dashed var(--border);
  border-radius: 8px;
  height: 360px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  background: rgba(255,255,255,0.02);
}

.score-box {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.score-value {
  font-size: 2.6rem;
  font-weight: 800;
}

.progress-wrap [data-testid="stProgress"] > div {
  background: var(--border);
}

.progress-wrap [data-testid="stProgress"] div[role="progressbar"] {
  background: var(--accent);
  transition: width 0.4s ease;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
}

.metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
}

.metric-label {
  font-size: 0.75rem;
  color: var(--muted);
  margin-bottom: 4px;
}

.metric-value {
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--text);
}

.metric-line {
  height: 2px;
  margin-top: 8px;
}

.risk-row {
  border-left: 4px solid var(--danger);
  background: rgba(255,255,255,0.02);
  padding: 10px 12px;
  border-radius: 6px;
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.det-header {
  display: grid;
  grid-template-columns: 1fr 140px;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.8rem;
}

.det-row {
  display: grid;
  grid-template-columns: 1fr 140px;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
}

.det-row.alt {
  background: rgba(255,255,255,0.02);
}

[data-testid="stSidebar"] {
  width: 260px;
}

.sidebar-title {
  font-size: 0.85rem;
  color: var(--muted);
  margin-bottom: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.sidebar-line {
  height: 1px;
  background: var(--border);
  margin: 14px 0;
}

.status-row {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 0.85rem;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
}

@media (max-width: 1100px) {
  .metric-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="top-accent"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="header-row">
      <div>
        <div class="title">RiskRider</div>
        <div class="subtitle">Motosiklet Sürücü Risk Analiz Sistemi</div>
      </div>
      <div class="version">v1.0 BETA</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-title">Parametreler</div>', unsafe_allow_html=True)
    confidence = st.slider("Güven Skoru", 0.10, 0.90, 0.40, 0.05)

    st.markdown('<div class="sidebar-line"></div>', unsafe_allow_html=True)
    if is_model_available():
        st.markdown(
            '<div class="status-row"><div class="status-dot"></div><div>Sistem Aktif</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-row"><div class="status-dot" style="background:var(--danger)"></div>'
            '<div>Sistem Pasif</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-line"></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:var(--muted);font-size:0.75rem;">RiskRider v1.0</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Görüntü Yükleme</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="upload-hint">
          <div>Görüntü yükleyin</div>
          <div class="upload-subtext">Desteklenen formatlar: JPG, JPEG, PNG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    upload = st.file_uploader("Görüntü yükleyin", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    original_image = None
    if upload is not None:
        original_image = Image.open(io.BytesIO(upload.read())).convert("RGB")
        st.image(original_image, caption="Orijinal Görüntü", use_container_width=True)

    st.markdown('<div class="button-primary">', unsafe_allow_html=True)
    analyze = st.button("Analiz Et", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tespit Sonucu</div>', unsafe_allow_html=True)
    if "annotated" in st.session_state:
        st.image(st.session_state["annotated"], caption="Tespitli Görüntü", use_container_width=True)
    else:
        st.markdown('<div class="waiting-panel">Analiz bekleniyor</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if analyze and original_image is not None:
    annotated, detections = run_detection(original_image, confidence)
    st.session_state["annotated"] = annotated
    st.session_state["detections"] = detections
    score, level, triggered = calculate_risk_score(detections)
    st.session_state["score"] = score
    st.session_state["level"] = level
    st.session_state["triggered"] = triggered

if analyze and original_image is None:
    st.warning("Lütfen önce bir görüntü yükleyin.")

if "detections" in st.session_state:
    detections: List[Dict] = st.session_state["detections"]
    score = st.session_state["score"]
    level = st.session_state["level"]
    triggered = st.session_state["triggered"]

    if score >= 80:
        color = "var(--success)"
        bg = "rgba(16, 185, 129, 0.08)"
    elif score >= 50:
        color = "var(--warning)"
        bg = "rgba(245, 158, 11, 0.08)"
    elif score >= 20:
        color = "var(--danger)"
        bg = "rgba(239, 68, 68, 0.08)"
    else:
        color = "var(--danger)"
        bg = "rgba(239, 68, 68, 0.14)"

    st.markdown("## Analiz Sonuçları")

    st.markdown(
        f"""
        <div class="score-box" style="background:{bg}">
          <div>
            <div style="color:var(--muted);font-size:0.85rem;">Risk Skoru</div>
            <div class="score-value" style="color:{color}">{score}</div>
          </div>
          <div style="font-size:1rem;font-weight:700;">{level}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="progress-wrap">', unsafe_allow_html=True)
    st.progress(score / 100.0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Tetiklenen Risk Faktörleri")
    if triggered:
        for item in triggered:
            class_name = item["class_name"]
            label = item["label"]
            delta = item["delta"]
            color_tuple: Tuple[int, int, int] = BOX_COLORS.get(class_name, (239, 68, 68))
            color_css = f"rgb({color_tuple[0]}, {color_tuple[1]}, {color_tuple[2]})"
            st.markdown(
                f"<div class='risk-row' style='border-left-color:{color_css};'>"
                f"<div>{label}</div><div>{delta} puan</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Risk faktörü tespit edilmedi.")

    st.markdown("### Güvenlik Metrikleri")
    counts = {"helmet": 0, "no_helmet": 0, "vest": 0, "no_vest": 0, "phone_use": 0}
    for det in detections:
        key = det.get("class_name")
        if key in counts:
            counts[key] += 1

    metric_colors = {
        "helmet": "var(--success)",
        "no_helmet": "var(--danger)",
        "vest": "var(--success)",
        "no_vest": "var(--warning)",
        "phone_use": "var(--danger)",
    }

    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Kasklı</div>
            <div class="metric-value">{counts['helmet']}</div>
            <div class="metric-line" style="background:{metric_colors['helmet']}"></div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Kasksız</div>
            <div class="metric-value">{counts['no_helmet']}</div>
            <div class="metric-line" style="background:{metric_colors['no_helmet']}"></div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Yelekit</div>
            <div class="metric-value">{counts['vest']}</div>
            <div class="metric-line" style="background:{metric_colors['vest']}"></div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Yeleğiz</div>
            <div class="metric-value">{counts['no_vest']}</div>
            <div class="metric-line" style="background:{metric_colors['no_vest']}"></div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Telefon Kullanan</div>
            <div class="metric-value">{counts['phone_use']}</div>
            <div class="metric-line" style="background:{metric_colors['phone_use']}"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Bireysel Tespitler")
    if detections:
        st.markdown(
            """
            <div class="section-card" style="padding:0;">
              <div class="det-header">
                <div>Sınıf</div>
                <div>Güven Skoru</div>
              </div>
            """,
            unsafe_allow_html=True,
        )
        for i, det in enumerate(detections):
            display_name = det["display_name"]
            conf = det["confidence"]
            alt_class = " alt" if i % 2 else ""
            st.markdown(
                f"<div class='det-row{alt_class}'>"
                f"<div>{display_name}</div><div>%{conf*100:.1f}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Hiç tespit bulunamadı.")
