from constants import *

def handleEngagement(state, current_values):
    val = "placeholder"


def updateSessionStats(state, batch):
    """
    Updates state.session_stats using batch-level statistics.
    batch = {
        "gaze": {"mean": float, "n": int, "M2": float},
        "head": {...},
        "blink": {...},
        "expression": {...}
    }
    """

    for metric, stats in batch.items():
        batch_mean = stats["mean"]
        batch_n = stats["n"]
        batch_M2 = stats["M2"]

        if batch_n == 0:
            continue  # nothing to update

        roll = state.session_stats[metric]

        # If this is the first data for this metric
        if roll["n"] == 0:
            roll["mean"] = batch_mean
            roll["n"] = batch_n
            roll["M2"] = batch_M2
            roll["std"] = (batch_M2 / batch_n) ** 0.5 if batch_n > 1 else 0
            continue

        # Merge using Welford's algorithm
        old_n = roll["n"]
        new_n = old_n + batch_n

        delta = batch_mean - roll["mean"]

        # Update mean
        new_mean = roll["mean"] + (batch_n / new_n) * delta

        # Update M2
        new_M2 = (
            roll["M2"]
            + batch_M2
            + (delta ** 2) * (old_n * batch_n / new_n)
        )

        # Save back
        roll["mean"] = new_mean
        roll["n"] = new_n
        roll["M2"] = new_M2
        roll["std"] = (new_M2 / new_n) ** 0.5 if new_n > 1 else 0




def handle_false_alarm(state, current_values):
    """
    state: State object
    current_values: dict of latest proxy values, e.g. {"gaze": 0.7, "blink": 0.3, ...}
    """

    # Compute z-scores using session baselines
    z_scores = {}
    for proxy, value in current_values.items():
        stats = state.session_stats[proxy]
        mean = stats["mean"]
        std = stats["std"]

        if mean is None or std is None:
            continue

        z_scores[proxy] = abs((value - mean) / (std + 1e-6))

    if not z_scores:
        return  # Not enough data to determine an offender

    # Identify the main offender
    main_offender = max(z_scores, key=z_scores.get)

    # Increment false alarm counter
    state.false_alarm_history[main_offender] += 1

    # If this proxy has caused enough false alarms, adjust it
    if state.false_alarm_history[main_offender] >= FALSE_ALARM_LIMIT:
        adjust_proxy(state, main_offender)
        state.false_alarm_history[main_offender] = 0  # reset counter
        state.save_profile()



def adjust_proxy(state, proxy):
    std = state.session_stats[proxy]["std"]

    if std is not None and std < LOW_VARIANCE_THRESHOLD:
        # Behavior is stable → threshold is too strict
        adjust_threshold(state, proxy)
    else:
        # Behavior is noisy → reduce weight
        adjust_weight(state, proxy)


def adjust_threshold(state, proxy):
    if proxy == "gaze":
        state.thresholds["max_offscreen_time"] += GAZE_THRESHOLD_INCREMENT
    elif proxy == "blink":
        state.thresholds["blink_rate_threshold"] += BLINK_THRESHOLD_INCREMENT
    elif proxy == "head":
        state.thresholds["head_movement_tolerance"] += HEAD_THRESHOLD_INCREMENT
    elif proxy == "expression":
        state.thresholds["expression_drop_threshold"] += EXPRESSION_THRESHOLD_INCREMENT



def adjust_weight(state, proxy):
    state.weights[proxy] = max(
        state.weights[proxy] - WEIGHT_DECREMENT,
        MIN_WEIGHT
    )

