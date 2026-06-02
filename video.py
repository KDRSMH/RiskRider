"""Video analysis module for RiskRider."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from detect import calculate_risk_score, run_detection


def analyze_video(
    input_path: str,
    confidence: float = 0.40,
    process_fps: float = 2.0,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[str, List[Tuple[float, int, str]]]:
    """Analyse a video file frame-by-frame and produce an annotated output.

    Args:
        input_path: Path to the source MP4 video.
        confidence: YOLO detection confidence threshold.
        process_fps: How many frames per second to actually process
            (others are copied as-is).  Lower values = faster.
        progress_callback: Optional ``fn(ratio)`` called with 0.0-1.0.

    Returns:
        A tuple of ``(output_video_path, timeline)`` where *timeline* is a
        list of ``(timestamp_seconds, risk_score, risk_level)`` entries.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Video açılamadı: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine frame interval – process 1 frame every `skip` frames.
    skip = max(1, int(round(src_fps / process_fps)))

    # Prepare output writer in a temp file.
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp.name, fourcc, src_fps, (width, height))

    timeline: List[Tuple[float, int, str]] = []
    frame_idx = 0
    last_annotated_bgr: Optional[np.ndarray] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            # Convert BGR → RGB PIL Image for detection.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            annotated_pil, detections = run_detection(pil_img, confidence)
            score, level, _ = calculate_risk_score(detections)

            timestamp = frame_idx / src_fps
            timeline.append((round(timestamp, 2), score, level))

            # Convert annotated PIL back to BGR for VideoWriter.
            annotated_rgb = np.array(annotated_pil)
            last_annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # Resize if dimensions changed (safety).
            if last_annotated_bgr.shape[:2] != (height, width):
                last_annotated_bgr = cv2.resize(
                    last_annotated_bgr, (width, height)
                )

        # Write the most recently annotated frame (or original if first).
        if last_annotated_bgr is not None:
            out.write(last_annotated_bgr)
        else:
            out.write(frame)

        frame_idx += 1

        if progress_callback and total_frames > 0:
            progress_callback(min(frame_idx / total_frames, 1.0))

    cap.release()
    out.release()

    return tmp.name, timeline
