"""Stream utilities for RiskRider – RTSP/HTTP live camera support.

Handles IP Webcam (Android), RTSP cameras, and HTTP MJPEG streams with
automatic reconnection, FFmpeg warning suppression, and drop-frame logic
to survive transient network hiccups without crashing.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Generator, Optional, Union

# ── FFmpeg / libav log suppression ──────────────────────────────────────
# MUST be set BEFORE importing cv2 — OpenCV reads these at init time.
# "Stream ends prematurely" is an FFmpeg-internal warning triggered by
# HTTP chunked-transfer streams (IP Webcam) that never send a proper
# Content-Length. It is harmless for live streams.
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "error"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2  # noqa: E402  — must come after env vars are set
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Maximum consecutive read failures before triggering a reconnect attempt.
_MAX_CONSECUTIVE_FAILURES = 60  # ~3 seconds at 0.05 s sleep per failure

# Maximum reconnect attempts before giving up entirely.
_MAX_RECONNECT_ATTEMPTS = 5

# Delay between reconnect attempts (seconds).
_RECONNECT_DELAY = 2.0


def open_stream(url: str) -> cv2.VideoCapture:
    """Open an RTSP or HTTP video stream and return a VideoCapture object.

    Applies optimal capture settings for live IP camera streams:
    - Small buffer (2 frames) to minimise latency and stale-frame issues.
    - Tries multiple backends in order (auto-detect → FFMPEG → GStreamer)
      to maximise compatibility across different OpenCV builds.

    Args:
        url: Stream URL (e.g. ``rtsp://...`` or ``http://.../video``).

    Returns:
        An opened ``cv2.VideoCapture`` instance.

    Raises:
        ConnectionError: If the stream cannot be opened with any backend.
    """
    # Backend priority: let OpenCV pick the best one first, then try
    # specific backends as fallbacks.
    backends = [
        (cv2.CAP_ANY, "AUTO"),
        (cv2.CAP_FFMPEG, "FFMPEG"),
        (cv2.CAP_GSTREAMER, "GSTREAMER"),
    ]

    for api, name in backends:
        try:
            cap = cv2.VideoCapture(url, api)
        except Exception:
            logger.debug("Backend %s açılırken istisna oluştu, atlanıyor.", name)
            continue

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            logger.info("Stream bağlantısı kuruldu (%s backend): %s", name, url)
            return cap

        # Opened but not usable — release before trying next backend
        cap.release()
        logger.debug("Backend %s stream'i açamadı, sıradaki deneniyor.", name)

    raise ConnectionError(f"Stream açılamadı (tüm backend'ler denendi): {url}")


def _reconnect(url: str) -> Optional[cv2.VideoCapture]:
    """Try to re-establish the stream connection.

    Returns a new ``VideoCapture`` on success, or ``None`` after exhausting
    all retry attempts.
    """
    for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
        logger.warning(
            "Yeniden bağlanma denemesi %d/%d …", attempt, _MAX_RECONNECT_ATTEMPTS
        )
        time.sleep(_RECONNECT_DELAY)
        try:
            cap = open_stream(url)
            logger.info("Yeniden bağlantı başarılı (deneme %d).", attempt)
            return cap
        except ConnectionError:
            continue
    logger.error("Tüm yeniden bağlanma denemeleri başarısız oldu.")
    return None


def ensure_portrait(frame: np.ndarray) -> np.ndarray:
    """Rotate a landscape frame 90° counter-clockwise to make it portrait.

    If the frame is already portrait (height >= width), it is returned
    unchanged.  This fixes the common IP Webcam issue where the phone
    camera sends landscape-oriented frames even when held vertically.
    """
    h, w = frame.shape[:2]
    if w > h:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def to_pil_image(frame: np.ndarray) -> Image.Image:
    """Convert a BGR OpenCV frame into a PIL image safely.

    Uses direct matrix conversion (cvtColor + fromarray) — no intermediate
    BytesIO or encoding step.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def frame_generator(
    cap: cv2.VideoCapture,
    url: str = "",
    interval: int = 5,
    as_pil: bool = False,
    force_portrait: bool = True,
) -> Generator[Union[np.ndarray, Image.Image], None, None]:
    """Yield frames from *cap*, skipping frames to reduce processing load.

    Only every *interval*-th successfully-read frame is yielded, effectively
    lowering the processing FPS without discarding the stream connection.

    Resilience features
    -------------------
    * **Drop-frame**: Failed reads (``ret=False`` or ``frame is None``) are
      silently skipped with a short sleep — the loop never breaks on a
      single bad frame.
    * **Auto-reconnect**: After ``_MAX_CONSECUTIVE_FAILURES`` successive
      read failures the generator releases the old capture and attempts to
      open a fresh connection (up to ``_MAX_RECONNECT_ATTEMPTS`` times).

    Args:
        cap: An opened ``cv2.VideoCapture``.
        url: Original stream URL, used for automatic reconnection.
             Pass an empty string to disable reconnection.
        interval: Number of frames to skip between yields (default 5).
        as_pil: If ``True``, yield ``PIL.Image.Image`` instead of numpy.
        force_portrait: If ``True`` (default), landscape frames are
            automatically rotated 90° to portrait orientation.

    Yields:
        BGR ``numpy.ndarray`` or ``PIL.Image.Image`` frames ready for
        inference.
    """
    frame_count = 0
    fail_count = 0

    while True:
        # ── Safety: check capture is still alive ──────────────────────
        if cap is None or not cap.isOpened():
            if url:
                cap = _reconnect(url)
                if cap is None:
                    return  # give up
                fail_count = 0
                continue
            else:
                return

        # ── Read a frame ──────────────────────────────────────────────
        ret, frame = cap.read()

        if not ret or frame is None:
            fail_count += 1
            if fail_count >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "%d ardışık okuma hatası — yeniden bağlanma tetikleniyor.",
                    fail_count,
                )
                release_stream(cap)
                cap = None
                fail_count = 0
                continue  # will hit the reconnect block above
            # Drop this frame and retry shortly
            time.sleep(0.05)
            continue

        # ── Successful read — reset failure counter ───────────────────
        fail_count = 0
        frame_count += 1

        if frame_count % interval != 0:
            # Skip this frame (don't yield), but still throttle slightly
            # to avoid a tight CPU-burning loop.
            time.sleep(0.01)
            continue

        # ── Orient & yield the frame ──────────────────────────────────
        if force_portrait:
            frame = ensure_portrait(frame)

        if as_pil:
            yield to_pil_image(frame)
        else:
            yield frame

        # Prevent Streamlit / UI thread starvation
        time.sleep(0.03)


def release_stream(cap: Optional[cv2.VideoCapture]) -> None:
    """Safely release a VideoCapture object.

    Args:
        cap: The capture to release. ``None`` is silently ignored.
    """
    if cap is not None:
        try:
            if cap.isOpened():
                cap.release()
                logger.info("Stream bağlantısı kapatıldı.")
        except Exception:
            pass  # never crash on cleanup
