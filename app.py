"""Streamlit UI for RiskRider."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress noisy FFmpeg/OpenCV warnings (e.g. "Stream ends prematurely")
# MUST be set before importing cv2.
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "error"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2  # noqa: E402
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from detect import (
    BOX_COLORS,
    CLASS_NAMES,
    calculate_risk_score,
    is_model_available,
    run_detection,
)
from stream import frame_generator, open_stream, release_stream
from video import analyze_video

st.set_page_config(page_title="RiskRider", page_icon="🏍", layout="wide")

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

/* Stream frame styling */
.stream-frame {
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

/* Video section */
.video-section {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
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
      <div class="version">v2.0 BETA</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────
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
    st.markdown('<div style="color:var(--muted);font-size:0.75rem;">RiskRider v2.0</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  Helper: render risk results (shared by image & video tabs)
# ═══════════════════════════════════════════════════════════════════════

def _render_risk_results(detections: List[Dict], score: int, level: str, triggered: List[Dict]):
    """Render risk score card, triggered factors, safety metrics, and detection table."""
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
    counts = {"B_helmet_worn": 0, "A_helmet_not_worn": 0, "C_Motorcycle": 0, "D_person": 0}
    for det in detections:
        key = det.get("class_name")
        if key in counts:
            counts[key] += 1

    metric_colors = {
        "B_helmet_worn": "var(--success)",
        "A_helmet_not_worn": "var(--danger)",
        "C_Motorcycle": "var(--accent)",
        "D_person": "var(--warning)",
    }

    st.markdown(
        f"""
        <div class="metric-grid" style="grid-template-columns: repeat(4, 1fr);">
          <div class="metric-card">
            <div class="metric-label">Kasklı</div>
            <div class="metric-value">{counts['B_helmet_worn']}</div>
            <div class="metric-line" style="background:{metric_colors['B_helmet_worn']}"></div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Kasksız</div>
            <div class="metric-value">{counts['A_helmet_not_worn']}</div>
            <div class="metric-line" style="background:{metric_colors['A_helmet_not_worn']}"></div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Motosiklet</div>
            <div class="metric-value">{counts['C_Motorcycle']}</div>
            <div class="metric-line" style="background:{metric_colors['C_Motorcycle']}"></div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Kişi</div>
            <div class="metric-value">{counts['D_person']}</div>
            <div class="metric-line" style="background:{metric_colors['D_person']}"></div>
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


# ═══════════════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════════════

tab_image, tab_stream, tab_video = st.tabs(
    ["📷 Görüntü Analizi", "📡 Canlı Stream", "🎬 Video Analizi"]
)

# ── Tab 1: Görüntü Analizi ────────────────────────────────────────────
with tab_image:
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
        upload = st.file_uploader("Görüntü yükleyin", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="img_upload")
        original_image = None
        if upload is not None:
            original_image = Image.open(io.BytesIO(upload.read())).convert("RGB")
            st.image(original_image, caption="Orijinal Görüntü", use_container_width=True)

        st.markdown('<div class="button-primary">', unsafe_allow_html=True)
        analyze = st.button("Analiz Et", use_container_width=True, key="img_analyze")
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
        _render_risk_results(
            st.session_state["detections"],
            st.session_state["score"],
            st.session_state["level"],
            st.session_state["triggered"],
        )


# ── Tab 2: Canlı Stream ──────────────────────────────────────────────
with tab_stream:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📡 Canlı Kamera Akışı</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="upload-subtext" style="margin-bottom:12px;">'
        "IP Webcam veya RTSP kamera URL'sini girin (örn: http://192.168.1.5:8080/video)</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    stream_url = st.text_input(
        "Stream URL",
        placeholder="http://192.168.1.5:8080/video",
        label_visibility="collapsed",
        key="stream_url",
    )

    col_start, col_stop = st.columns(2)
    with col_start:
        st.markdown('<div class="button-primary">', unsafe_allow_html=True)
        start_stream = st.button("▶ Başlat", use_container_width=True, key="stream_start")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_stop:
        stop_stream = st.button("⏹ Durdur", use_container_width=True, key="stream_stop")

    # Placeholders for live view
    frame_placeholder = st.empty()
    score_placeholder = st.empty()

    if stop_stream:
        st.session_state["streaming"] = False

    if start_stream and stream_url:
        st.session_state["streaming"] = True

        # ── URL validation & auto-correction ──────────────────────────
        url = stream_url.strip()

        # IP Webcam doesn't support HTTPS
        if url.startswith("https://"):
            url = url.replace("https://", "http://", 1)
            st.warning("⚠️ IP Webcam HTTPS desteklemez — otomatik olarak HTTP'ye çevrildi.")

        # Ensure the URL has a path (IP Webcam needs /video)
        if url.count("/") == 2 and not url.endswith("/"):
            # URL is like http://IP:PORT with no path → append /video
            url = url + "/video"
            st.info(f"ℹ️ Endpoint eksikti, otomatik eklendi: `{url}`")
        elif url.endswith("/"):
            url = url + "video"
            st.info(f"ℹ️ Endpoint eksikti, otomatik eklendi: `{url}`")

        try:
            cap = open_stream(url)
        except ConnectionError as exc:
            st.error(str(exc))
            st.markdown(
                "**💡 İpucu:** IP Webcam için doğru format → `http://IP_ADRESI:8080/video`",
                unsafe_allow_html=True,
            )
            st.session_state["streaming"] = False
            cap = None

        if cap is not None:
            try:
                for pil_img in frame_generator(cap, url=url, interval=5, as_pil=True):
                    if not st.session_state.get("streaming", False):
                        break

                    annotated_pil, detections = run_detection(pil_img, confidence)
                    score, level, triggered = calculate_risk_score(detections)

                    frame_placeholder.image(annotated_pil, caption="Canlı Akış", use_container_width=True)

                    if score >= 80:
                        s_color = "var(--success)"
                        s_bg = "rgba(16, 185, 129, 0.08)"
                    elif score >= 50:
                        s_color = "var(--warning)"
                        s_bg = "rgba(245, 158, 11, 0.08)"
                    else:
                        s_color = "var(--danger)"
                        s_bg = "rgba(239, 68, 68, 0.08)"

                    score_placeholder.markdown(
                        f"""
                        <div class="score-box" style="background:{s_bg}">
                          <div>
                            <div style="color:var(--muted);font-size:0.85rem;">Anlık Risk Skoru</div>
                            <div class="score-value" style="color:{s_color}">{score}</div>
                          </div>
                          <div style="font-size:1rem;font-weight:700;">{level}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Generator exhausted — all reconnect attempts failed
                    st.error("📡 Kamera bağlantısı kalıcı olarak kesildi. Lütfen URL'yi kontrol edip tekrar deneyin.")
            finally:
                release_stream(cap)
                st.session_state["streaming"] = False

    elif start_stream and not stream_url:
        st.warning("Lütfen bir stream URL'si girin.")

    if not st.session_state.get("streaming", False):
        frame_placeholder.markdown(
            '<div class="waiting-panel">Akış bekleniyor – URL girin ve Başlat\'a tıklayın</div>',
            unsafe_allow_html=True,
        )


# ── Tab 3: Video Analizi ─────────────────────────────────────────────
with tab_video:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎬 Video Dosyası Analizi</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="upload-hint">
          <div>MP4 video yükleyin</div>
          <div class="upload-subtext">Video dosyasını yükleyin, sistem kare kare analiz edecektir</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    video_file = st.file_uploader(
        "Video yükleyin",
        type=["mp4"],
        label_visibility="collapsed",
        key="video_upload",
    )

    st.markdown('<div class="button-primary">', unsafe_allow_html=True)
    analyze_vid = st.button("Videoyu Analiz Et", use_container_width=True, key="video_analyze")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_vid and video_file is not None:
        # Save uploaded video to a temp file
        tmp_input = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_input.write(video_file.read())
        tmp_input.flush()
        tmp_input_path = tmp_input.name

        progress_bar = st.progress(0.0, text="Video analiz ediliyor...")

        def _update_progress(ratio: float) -> None:
            progress_bar.progress(min(ratio, 1.0), text=f"İşleniyor: %{ratio*100:.0f}")

        try:
            output_path, timeline = analyze_video(
                tmp_input_path,
                confidence=confidence,
                process_fps=2.0,
                progress_callback=_update_progress,
            )
            progress_bar.progress(1.0, text="Analiz tamamlandı ✅")
        except Exception as exc:
            st.error(f"Video işlenirken hata: {exc}")
            output_path, timeline = None, []

        if output_path and Path(output_path).exists():
            st.markdown("### İşlenmiş Video")
            with open(output_path, "rb") as vf:
                video_bytes = vf.read()

            st.download_button(
                label="📥 İşlenmiş Videoyu İndir",
                data=video_bytes,
                file_name="riskrider_analyzed.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        if timeline:
            st.markdown("### Zaman Bazlı Risk Değişim Grafiği")

            timestamps = [t[0] for t in timeline]
            scores = [t[1] for t in timeline]

            fig = go.Figure()

            # Main risk score line
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=scores,
                    mode="lines+markers",
                    name="Risk Skoru",
                    line=dict(color="#3b82f6", width=2.5),
                    marker=dict(size=5, color="#3b82f6"),
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.08)",
                )
            )

            # Threshold lines
            fig.add_hline(
                y=80,
                line_dash="dash",
                line_color="#10b981",
                line_width=1.5,
                annotation_text="Düşük Risk (80)",
                annotation_position="top left",
                annotation_font_color="#10b981",
            )
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="#f59e0b",
                line_width=1.5,
                annotation_text="Orta Risk (50)",
                annotation_position="top left",
                annotation_font_color="#f59e0b",
            )
            fig.add_hline(
                y=20,
                line_dash="dash",
                line_color="#ef4444",
                line_width=1.5,
                annotation_text="Yüksek Risk (20)",
                annotation_position="top left",
                annotation_font_color="#ef4444",
            )

            fig.update_layout(
                xaxis_title="Zaman (saniye)",
                yaxis_title="Risk Skoru",
                yaxis=dict(range=[0, 105]),
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#0a0f1e",
                font=dict(color="#f9fafb", family="system-ui, -apple-system, Segoe UI, Arial, sans-serif"),
                margin=dict(l=40, r=20, t=30, b=40),
                height=420,
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)

    elif analyze_vid and video_file is None:
        st.warning("Lütfen önce bir MP4 video dosyası yükleyin.")
