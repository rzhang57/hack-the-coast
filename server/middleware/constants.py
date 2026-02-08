# -----------------------------
# False alarm logic
# -----------------------------
FALSE_ALARM_LIMIT = 3

# -----------------------------
# Variance thresholds
# -----------------------------
LOW_VARIANCE_THRESHOLD = 0.15
VARIANCE_IMPROVEMENT_THRESHOLD = 0.05  # recommended for long-term tightening

# -----------------------------
# Weight adjustments
# -----------------------------
WEIGHT_DECREMENT = 0.05
MIN_WEIGHT = 0.10

# -----------------------------
# Threshold adjustments (relaxing)
# -----------------------------
GAZE_THRESHOLD_INCREMENT = 0.5
BLINK_THRESHOLD_INCREMENT = 0.05
HEAD_THRESHOLD_INCREMENT = 0.05
EXPRESSION_THRESHOLD_INCREMENT = 0.05

# -----------------------------
# Threshold tightening (optional)
# -----------------------------
MIN_GAZE_THRESHOLD = 1.0
MIN_BLINK_THRESHOLD = 0.20
MIN_HEAD_THRESHOLD = 0.10
MIN_EXPRESSION_THRESHOLD = 0.05

# -----------------------------
# Engagement thresholds
# -----------------------------
HIGH_ENGAGEMENT_THRESHOLD = 0.75

# -----------------------------
# Good behavior logic
# -----------------------------
GOOD_BEHAVIOR_LIMIT = 60  # e.g., 60 cycles of good behavior
