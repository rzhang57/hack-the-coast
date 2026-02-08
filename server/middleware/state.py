import json
import os
import time

class State:
    def __init__(self, profile_path="user_profile.json"):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.profile_path = os.path.join(base_dir, profile_path)


        # Load JSON if it exists
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r") as f:
                profile = json.load(f)
        else:
            profile = {}

        # -----------------------------
        # Defaults
        # -----------------------------
        default_weights = {
            "gaze": 0.35,
            "head": 0.25,
            "blink": 0.20,
            "expression": 0.20,
        }

        default_thresholds = {
            "max_offscreen_time": 3.0,
            "blink_rate_threshold": 0.60,
            "head_movement_tolerance": 0.30,
            "expression_drop_threshold": 0.20,
        }

        default_false_alarm_history = {
            "gaze": 0,
            "head": 0,
            "blink": 0,
            "expression": 0,
        }

        default_long_term_variance = {
            "gaze": None,
            "head": None,
            "blink": None,
            "expression": None,
        }

        default_good_behavior = {
            "gaze": 0,
            "head": 0,
            "blink": 0,
            "expression": 0,
        }

        # -----------------------------
        # Initialize WEIGHTS
        # -----------------------------
        self.weights = {
            k: profile.get("weights", {}).get(k, default_weights[k])
            for k in default_weights
        }

        # -----------------------------
        # Initialize THRESHOLDS
        # -----------------------------
        self.thresholds = {
            k: profile.get("thresholds", {}).get(k, default_thresholds[k])
            for k in default_thresholds
        }

        # -----------------------------
        # False alarm counters
        # -----------------------------
        self.false_alarm_history = profile.get(
            "false_alarm_history", default_false_alarm_history
        )

        # -----------------------------
        # Good behavior counters
        # -----------------------------
        self.good_behavior_counter = profile.get(
            "good_behavior_counter", default_good_behavior
        )

        # -----------------------------
        # Long-term variance snapshots
        # -----------------------------
        self.long_term_variance = profile.get(
            "long_term_variance", default_long_term_variance
        )

        # -----------------------------
        # Timestamp for long-term variance updates
        # -----------------------------
        self.last_variance_update_time = profile.get(
            "last_variance_update_time", time.time()
        )

        # -----------------------------
        # Session-only rolling stats
        # -----------------------------
        # TODO adjust the metrics here
        self.session_stats = {
            "gaze": {"mean": None, "std": None, "n": 0, "M2": 0},
            "head": {"mean": None, "std": None, "n": 0, "M2": 0},
            "blink": {"mean": None, "std": None, "n": 0, "M2": 0},
            "expression": {"mean": None, "std": None, "n": 0, "M2": 0},
        }

        # adjust this based onn sunny TODO
        self.last_line_processed = profile.get("last_line_processed", 0)
        # adjust this based onn sunny TODO
        self.keyboard_stroke = profile.get("keyboard_stroke", true)
        self.cursor_movement = profile.get("cursor_movement", true)
        


    # Save persistent profile
    # -----------------------------
    def save_profile(self):
        profile = {
            "weights": self.weights,
            "thresholds": self.thresholds,
            "false_alarm_history": self.false_alarm_history,
            "long_term_variance": self.long_term_variance,
            "good_behavior_counter": self.good_behavior_counter,
            "last_variance_update_time": self.last_variance_update_time,
            "last_line_processed": self.last_line_processed
        }
        with open(self.profile_path, "w") as f:
            json.dump(profile, f, indent=2)
