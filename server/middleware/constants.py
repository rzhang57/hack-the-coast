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


# TODO adjust these names and order and stuff.
THRESHOLD_KEYS = {
    "gaze": "max_offscreen_time",
    "blink": "blink_rate_threshold",
    "head": "head_movement_tolerance",
    "expression": "expression_drop_threshold"
}

# -----------------------------
# Threshold adjustments (relaxing)
# -----------------------------
THRESHOLD_LOOSEN = 1.05

# -----------------------------
# Threshold tightening (optional)
# -----------------------------
THRESHOLD_TIGHTEN = 0.975

# -----------------------------
# Engagement thresholds
# -----------------------------
HIGH_ENGAGEMENT_THRESHOLD = 0.75

# -----------------------------
# Good behavior logic
# -----------------------------
GOOD_BEHAVIOR_LIMIT = 60  # e.g., 60 cycles of good behavior
