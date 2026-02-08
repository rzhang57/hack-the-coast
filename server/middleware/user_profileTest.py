import sys
import os

from state import State

# Instantiate state (this will load the JSON if present)
state = State()

# -----------------------------
# PRINT EVERYTHING
# -----------------------------

print("\n=== WEIGHTS ===")
for k, v in state.weights.items():
    print(f"{k}: {v}")

print("\n=== THRESHOLDS ===")
for k, v in state.thresholds.items():
    print(f"{k}: {v}")

print("\n=== FALSE ALARM HISTORY ===")
for k, v in state.false_alarm_history.items():
    print(f"{k}: {v}")

print("\n=== LONG-TERM VARIANCE ===")
for k, v in state.long_term_variance.items():
    print(f"{k}: {v}")

print("\n=== GOOD BEHAVIOR COUNTERS ===")
for k, v in state.good_behavior_counter.items():
    print(f"{k}: {v}")

print("\n=== LAST VARIANCE UPDATE TIME ===")
print(state.last_variance_update_time)

print("\n=== SESSION STATS (session-only) ===")
for k, v in state.session_stats.items():
    print(f"{k}: {v}")

# -----------------------------
# MODIFY + SAVE TEST
# -----------------------------

print("\nBefore:", state.weights["gaze"])

# Modify something
state.weights["gaze"] += 0.1

# Save changes
state.save_profile()

print("After:", state.weights["gaze"])
