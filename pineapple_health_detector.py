"""
Pineapple Plant Health Detection System
========================================
Runs on a Raspberry Pi with the Raspberry Pi Camera Module.
Captures images from a farming drone and analyses pineapple-leaf colour to
classify each plant's health status.

Usage (on the Raspberry Pi):
    python pineapple_health_detector.py [--single] [--output-dir DIR]

    --single        Capture and analyse one image then exit.
    --output-dir    Directory for saved images / CSV log (default: "captures").

Dependencies:
    pip install -r requirements.txt
"""

import argparse
import csv
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Health status labels ───────────────────────────────────────────────────────
STATUS_HEALTHY  = "Healthy"
STATUS_MODERATE = "Moderate Stress"
STATUS_SEVERE   = "Severe Stress / Diseased"


# ─────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_camera():
    """
    Return a camera capture object.

    Tries picamera2 first (Raspberry Pi OS Bookworm), then falls back to
    picamera (legacy), then to a standard USB/CSI camera via OpenCV.
    """
    try:
        from picamera2 import Picamera2  # type: ignore
        cam = Picamera2()
        cam_config = cam.create_still_configuration(
            main={"size": config.CAMERA_RESOLUTION}
        )
        cam.configure(cam_config)
        cam.start()
        time.sleep(config.CAPTURE_DELAY_SEC)
        logger.info("Using picamera2.")
        return cam, "picamera2"
    except ImportError:
        pass

    try:
        import picamera  # type: ignore
        cam = picamera.PiCamera()
        cam.resolution = config.CAMERA_RESOLUTION
        cam.framerate = config.CAMERA_FRAMERATE
        cam.rotation  = config.CAMERA_ROTATION
        time.sleep(config.CAPTURE_DELAY_SEC)
        logger.info("Using picamera (legacy).")
        return cam, "picamera"
    except ImportError:
        pass

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError(
            "No camera found. Install picamera2 / picamera, or connect a USB camera."
        )
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_RESOLUTION[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
    time.sleep(config.CAPTURE_DELAY_SEC)
    logger.info("Using OpenCV VideoCapture (USB/CSI).")
    return cam, "opencv"


def capture_frame(camera, camera_type: str) -> np.ndarray:
    """Capture a single BGR frame from *camera*."""
    if camera_type == "picamera2":
        rgb = camera.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if camera_type == "picamera":
        import io
        stream = io.BytesIO()
        camera.capture(stream, format="jpeg")
        stream.seek(0)
        arr = np.frombuffer(stream.read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # opencv
    ret, frame = camera.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")
    return frame


def release_camera(camera, camera_type: str) -> None:
    """Release / close the camera."""
    if camera_type in ("picamera2", "picamera"):
        camera.close()
    else:
        camera.release()


# ─────────────────────────────────────────────────────────────────────────────
# Image-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Apply mild preprocessing:
    - Gaussian blur to reduce sensor noise.
    - CLAHE on the V channel to compensate for uneven outdoor lighting.
    """
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def segment_plant_regions(frame: np.ndarray) -> np.ndarray:
    """
    Return a binary mask isolating green / plant pixels to exclude bare soil,
    sky, drone body, etc.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 20, 20], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask


def classify_health(frame: np.ndarray, plant_mask: np.ndarray) -> dict:
    """
    Analyse colour distribution inside *plant_mask* to produce health metrics.

    Returns a dict with keys:
        total_plant_pixels, healthy_px, yellowing_px, brown_px,
        healthy_pct, yellowing_pct, brown_pct, status
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def _mask(lower, upper):
        m = cv2.inRange(hsv,
                        np.array(lower, dtype=np.uint8),
                        np.array(upper, dtype=np.uint8))
        return cv2.bitwise_and(m, m, mask=plant_mask)

    healthy_mask   = _mask(config.HEALTHY_HSV_LOWER,   config.HEALTHY_HSV_UPPER)
    yellowing_mask = _mask(config.YELLOWING_HSV_LOWER, config.YELLOWING_HSV_UPPER)
    brown_mask     = _mask(config.BROWN_HSV_LOWER,     config.BROWN_HSV_UPPER)

    total  = int(np.count_nonzero(plant_mask))
    h_px   = int(np.count_nonzero(healthy_mask))
    y_px   = int(np.count_nonzero(yellowing_mask))
    b_px   = int(np.count_nonzero(brown_mask))

    if total == 0:
        return {
            "total_plant_pixels": 0,
            "healthy_px": 0, "yellowing_px": 0, "brown_px": 0,
            "healthy_pct": 0.0, "yellowing_pct": 0.0, "brown_pct": 0.0,
            "status": "No plant detected",
        }

    h_pct = h_px / total * 100
    y_pct = y_px / total * 100
    b_pct = b_px / total * 100

    if h_pct >= config.HEALTHY_THRESHOLD:
        status = STATUS_HEALTHY
    elif h_pct >= config.MODERATE_THRESHOLD:
        status = STATUS_MODERATE
    else:
        status = STATUS_SEVERE

    return {
        "total_plant_pixels": total,
        "healthy_px":    h_px,   "yellowing_px":    y_px,   "brown_px":    b_px,
        "healthy_pct":   h_pct,  "yellowing_pct":   y_pct,  "brown_pct":   b_pct,
        "status": status,
    }


def detect_plant_contours(plant_mask: np.ndarray) -> list:
    """Return a list of significant plant contours sorted by area (largest first)."""
    contours, _ = cv2.findContours(
        plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return sorted(
        [c for c in contours if cv2.contourArea(c) >= config.MIN_PLANT_CONTOUR_AREA],
        key=cv2.contourArea,
        reverse=True,
    )


def annotate_frame(frame: np.ndarray, metrics: dict, contours: list) -> np.ndarray:
    """
    Draw health contours and a status overlay on a copy of *frame*.
    """
    annotated = frame.copy()

    status = metrics["status"]
    colour = {
        STATUS_HEALTHY:  (0,   200, 0),    # green
        STATUS_MODERATE: (0,   165, 255),  # orange
        STATUS_SEVERE:   (0,   0,   220),  # red
    }.get(status, (200, 200, 200))

    cv2.drawContours(annotated, contours, -1, colour, 2)

    lines = [
        f"Status : {status}",
        f"Healthy : {metrics['healthy_pct']:.1f}%",
        f"Yellowing: {metrics['yellowing_pct']:.1f}%",
        f"Brown/Rot: {metrics['brown_pct']:.1f}%",
    ]
    y0 = 30
    for i, line in enumerate(lines):
        cv2.putText(
            annotated, line,
            (10, y0 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2,
            cv2.LINE_AA,
        )

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_output_dir(directory: str) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(
    output_dir: Path,
    timestamp: str,
    frame: np.ndarray,
    annotated: np.ndarray,
    metrics: dict,
    log_file: str,
) -> None:
    """Save the raw/annotated image pair and append a CSV log entry."""
    base = output_dir / timestamp
    cv2.imwrite(str(base) + "_raw.jpg",       frame)
    if config.SAVE_ANNOTATED_IMAGES:
        cv2.imwrite(str(base) + "_annotated.jpg", annotated)

    csv_path = output_dir / log_file
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "timestamp", "status",
            "healthy_pct", "yellowing_pct", "brown_pct",
            "total_plant_pixels",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":          timestamp,
            "status":             metrics["status"],
            "healthy_pct":        f"{metrics['healthy_pct']:.2f}",
            "yellowing_pct":      f"{metrics['yellowing_pct']:.2f}",
            "brown_pct":          f"{metrics['brown_pct']:.2f}",
            "total_plant_pixels": metrics["total_plant_pixels"],
        })


# ─────────────────────────────────────────────────────────────────────────────
# Core processing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, list]:
    """
    Full analysis pipeline for one frame.

    Returns:
        metrics   – health metrics dict
        processed – preprocessed BGR frame
        annotated – annotated BGR frame
        contours  – list of plant contours
    """
    processed    = preprocess(frame)
    plant_mask   = segment_plant_regions(processed)
    contours     = detect_plant_contours(plant_mask)
    metrics      = classify_health(processed, plant_mask)
    annotated    = annotate_frame(processed, metrics, contours)
    return metrics, processed, annotated, contours


def analyze_image_file(image_path: str) -> dict:
    """
    Convenience function: load an image from disk and run the full pipeline.
    Returns the metrics dict.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    metrics, _, _, _ = process_frame(frame)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(single_shot: bool = False, output_dir: str = config.OUTPUT_DIR) -> None:
    """
    Main loop: capture → process → log → (optionally) repeat.

    Args:
        single_shot: If True, capture one frame and exit; otherwise loop.
        output_dir:  Directory for saved images and CSV log.
    """
    out = ensure_output_dir(output_dir)
    camera, camera_type = create_camera()

    logger.info("Pineapple Health Detection started. Output → %s", out)

    try:
        while True:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info("Capturing frame …")

            frame   = capture_frame(camera, camera_type)
            metrics, _, annotated, contours = process_frame(frame)

            logger.info(
                "Status: %-25s | Healthy: %5.1f%% | Yellowing: %5.1f%% | Brown: %5.1f%%",
                metrics["status"],
                metrics["healthy_pct"],
                metrics["yellowing_pct"],
                metrics["brown_pct"],
            )

            save_results(out, timestamp, frame, annotated, metrics, config.LOG_FILE)

            if single_shot:
                break

            time.sleep(config.CAPTURE_INTERVAL_SEC)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        release_camera(camera, camera_type)
        logger.info("Camera released. Exiting.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pineapple plant health detection using Raspberry Pi Camera."
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Capture and analyse a single frame then exit.",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help=f"Output directory for images and CSV log (default: {config.OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    run(single_shot=args.single, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
