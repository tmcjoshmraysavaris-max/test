# Pineapple Plant Health Detection System

A Raspberry Pi-based image-processing pipeline for farming drones that
captures images of pineapple crops and automatically classifies each plant's
health status using colour analysis.

---

## Features

| Feature | Detail |
|---|---|
| **Camera support** | picamera2 (Bookworm), picamera (legacy), or USB/OpenCV fallback |
| **Health statuses** | Healthy · Moderate Stress · Severe Stress / Diseased |
| **Indicators detected** | Vibrant green leaves, yellowing (nutrient stress), brown/necrotic tissue |
| **Output** | Annotated JPEG images + CSV health log per capture session |
| **Loop or single-shot** | Continuous capture loop or one-shot for drone fly-by triggers |

---

## Hardware Requirements

- Raspberry Pi 4 / 5 (or Zero 2 W for lightweight builds)
- Raspberry Pi Camera Module v2 / HQ Camera / v3
- MicroSD card (≥ 16 GB, Class 10)
- Power supply compatible with your drone frame

---

## Software Setup

### 1 – Install system dependencies (Raspberry Pi OS Bookworm)

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv
```

### 2 – Install Python packages

```bash
pip install -r requirements.txt
```

> **Legacy OS (Bullseye / Buster):**  
> Uncomment `# picamera>=1.13` in `requirements.txt` and install with  
> `pip install picamera`.

### 3 – Enable the camera interface

```bash
sudo raspi-config
# Interface Options → Camera → Enable
```

---

## Usage

### Continuous capture loop (default)

```bash
python pineapple_health_detector.py
```

Press **Ctrl+C** to stop.

### Single-shot capture (useful for drone trigger integration)

```bash
python pineapple_health_detector.py --single
```

### Custom output directory

```bash
python pineapple_health_detector.py --output-dir /mnt/usb/field_data
```

---

## Output

```
captures/
├── 20260303_091530_raw.jpg        # original captured frame
├── 20260303_091530_annotated.jpg  # frame with health overlay
├── 20260303_091535_raw.jpg
├── 20260303_091535_annotated.jpg
└── health_log.csv                 # timestamped CSV log
```

### Sample `health_log.csv`

```csv
timestamp,status,healthy_pct,yellowing_pct,brown_pct,total_plant_pixels
20260303_091530,Healthy,72.34,18.21,9.45,184320
20260303_091535,Moderate Stress,42.10,41.50,16.40,176000
```

### Annotated image overlay

The annotated image shows:
- **Green contours** – Healthy plants
- **Orange contours** – Moderate stress
- **Red contours** – Severe stress / disease

Text overlay in the top-left corner displays the percentage breakdown for
the current frame.

---

## Configuration

Edit **`config.py`** to tune the system to your field conditions:

| Parameter | Default | Description |
|---|---|---|
| `CAMERA_RESOLUTION` | `(1920, 1080)` | Capture resolution |
| `CAPTURE_INTERVAL_SEC` | `5.0` | Seconds between captures |
| `HEALTHY_HSV_LOWER/UPPER` | green range | HSV thresholds for healthy leaves |
| `YELLOWING_HSV_LOWER/UPPER` | yellow-green range | HSV thresholds for yellowing |
| `BROWN_HSV_LOWER/UPPER` | brown range | HSV thresholds for brown/necrotic tissue |
| `HEALTHY_THRESHOLD` | `60` | Min green % to classify as Healthy |
| `MODERATE_THRESHOLD` | `35` | Min green % for Moderate Stress |
| `SAVE_ANNOTATED_IMAGES` | `True` | Save annotated frames |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project Structure

```
.
├── pineapple_health_detector.py   # Main detection module
├── config.py                      # All tunable parameters
├── requirements.txt               # Python dependencies
├── tests/
│   └── test_pineapple_health_detector.py
└── captures/                      # Created at runtime
```