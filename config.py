"""
Configuration settings for the Pineapple Plant Health Detection System.
Adjust these values to tune detection for your field conditions.
"""

# ── Camera settings ──────────────────────────────────────────────────────────
CAMERA_RESOLUTION = (1920, 1080)   # Width x Height in pixels
CAMERA_FRAMERATE  = 30             # Frames per second
CAMERA_ROTATION   = 0              # Degrees (0 / 90 / 180 / 270)

# ── Image capture ────────────────────────────────────────────────────────────
CAPTURE_DELAY_SEC    = 2.0         # Warm-up time before the first capture
CAPTURE_INTERVAL_SEC = 5.0         # Seconds between consecutive captures
OUTPUT_DIR           = "captures"  # Directory for saved images and reports

# ── HSV colour thresholds for pineapple leaf health classification ───────────
# Format: (H_low, S_low, V_low), (H_high, S_high, V_high)  — OpenCV HSV range
#         H: 0-179, S: 0-255, V: 0-255

# Healthy leaf: vibrant green
HEALTHY_HSV_LOWER = (35, 40, 40)
HEALTHY_HSV_UPPER = (85, 255, 255)

# Yellowing / nutrient deficiency: yellow-green / yellow
YELLOWING_HSV_LOWER = (20, 40, 40)
YELLOWING_HSV_UPPER = (35, 255, 255)

# Brown / necrotic tissue (disease, rot)
BROWN_HSV_LOWER = (5, 40, 20)
BROWN_HSV_UPPER = (20, 255, 180)

# ── Health score thresholds (percentage of pixels classified as healthy) ─────
HEALTHY_THRESHOLD    = 60   # ≥ 60 % green → Healthy
MODERATE_THRESHOLD   = 35   # 35–59 % green → Moderate stress
# below MODERATE_THRESHOLD → Severe stress / diseased

# ── Minimum contour area to consider a plant region (in pixels²) ─────────────
MIN_PLANT_CONTOUR_AREA = 5000

# ── Output / reporting ────────────────────────────────────────────────────────
SAVE_ANNOTATED_IMAGES = True       # Overlay health labels on saved images
LOG_FILE              = "health_log.csv"  # CSV log of every reading
