"""
Application Configuration
-------------------------
Central configuration file for the Drowsiness Detection System.
"""

# =====================================================
# Model Paths
# =====================================================

YOLO_MODEL = "models/yolo/yolov8s.pt"
EYE_MODEL = "models/eye/eye_state_model.keras"

AGE_PROTO = "models/age/deploy_age.prototxt"
AGE_MODEL = "models/age/age_net.caffemodel"

# =====================================================
# Detection Thresholds
# =====================================================

PERSON_CONFIDENCE = 0.40

EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 20

# =====================================================
# Window
# =====================================================

WINDOW_TITLE = "AI Drowsiness Detection System"

# =====================================================
# Output Directories
# =====================================================

OUTPUT_IMAGE_DIR = "outputs/images/"
OUTPUT_VIDEO_DIR = "outputs/videos/"
OUTPUT_REPORT_DIR = "outputs/reports/"

# =====================================================
# Colors (BGR)
# =====================================================

COLOR_AWAKE = (0, 255, 0)
COLOR_SLEEP = (0, 0, 255)
COLOR_FACE = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)

# =====================================================
# Font
# =====================================================

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ----------------------------------
# Sleep Detection
# ----------------------------------

EAR_THRESHOLD = 0.23
CONSECUTIVE_FRAMES = 20


# ----------------------------------
# Age Groups
# ----------------------------------

AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]


# ----------------------------------
# Popup
# ----------------------------------

ENABLE_POPUP = True
