'''#Version 2

import pandas as pd
import numpy as np
import time
import json
import os
import sys
from collections import deque

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.getcwd(), "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "threshold_feedback.json")

# Default Thresholds for Z-Score deviations
DEFAULT_CONFIG = {
    "yaw_sigma_tolerance": 2.0,    # Horizontal tolerance
    "pitch_sigma_tolerance": 3.0,  # Vertical tolerance
    "blink_sigma_tolerance": 2.5   # Blink rate tolerance
}

class NeuroAnalyzer:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.calib_mean = {}
        self.calib_std = {}
        self.max_saccade_velocity = 0.0
        
        # Buffer: 10 seconds of data @ 30 FPS = 300 frames
        # We focus on the immediate window for engagement scoring
        self.window_size = 300
        self.buffer = deque(maxlen=self.window_size)
        
        if not self.load_calibration():
            print("CRITICAL: Calibration files not found. Run calibration.py first.")
            sys.exit(1)

    def load_calibration(self):
        """Loads baseline μ and σ from calibration files."""
        conv_path = os.path.join(DATA_DIR, "calibration_convergent.csv")
        div_path = os.path.join(DATA_DIR, "calibration_divergent.csv")

        if not os.path.exists(conv_path) or not os.path.exists(div_path): return False

        try:
            # 1. Convergent Baseline (Static Focus)
            df_conv = pd.read_csv(conv_path)
            for m in ['pitch', 'yaw', 'ear', 'gaze_x', 'gaze_y']:
                self.calib_mean[m] = df_conv[m].mean()
                self.calib_std[m] = df_conv[m].std()
                if self.calib_std[m] == 0: self.calib_std[m] = 0.001

            # 2. Divergent Baseline (Max Speed)
            df_div = pd.read_csv(div_path)
            gaze_diffs = np.diff(df_div[['gaze_x', 'gaze_y']].values, axis=0)
            velocities = np.linalg.norm(gaze_diffs, axis=1)
            self.max_saccade_velocity = np.percentile(velocities, 95)
            
            print(f"✅ Calibration Loaded. Baseline Pitch: {self.calib_mean['pitch']:.2f}")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

    def check_feedback_updates(self):
        """Reads external JSON to adjust thresholds dynamically."""
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    updates = json.load(f)
                    for k, v in updates.items():
                        if k in self.config: self.config[k] = float(v)
            except: pass

    def get_gaussian_score(self, value, metric, tolerance_sigma):
        """
        Converts a raw value into an 'Engagement Probability' (0.0 to 1.0).
        Uses a Gaussian curve centered on the calibrated mean.
        
        Logic:
        - If Value == Mean, Score = 1.0 (Perfect Engagement)
        - If Value deviates by 'tolerance_sigma', Score drops significantly.
        """
        mu = self.calib_mean[metric]
        sigma = self.calib_std[metric]
        
        # Calculate Z-distance
        z = abs(value - mu) / sigma
        
        # Gaussian Decay: e^(-(z^2) / (2 * tolerance^2))
        # This creates a bell curve where 0 deviation = 1.0 score
        score = np.exp(-(z**2) / (2 * (tolerance_sigma**2)))
        return float(score)

    def calculate_window_stats(self, values):
        """
        Calculates the requested statistical format for a list of values.
        Returns: {mean, std_dev, n, M2}
        """
        arr = np.array(values)
        n = len(arr)
        if n == 0: return {"mean": 0.0, "std_dev": 0.0, "n": 0, "M2": 0.0}
        
        mean = np.mean(arr)
        # M2 is the sum of squared deviations from the mean
        m2 = np.sum((arr - mean)**2)
        std_dev = np.sqrt(m2 / n)
        
        return {
            "mean": round(float(mean), 4),
            "std_dev": round(float(std_dev), 4),
            "n": int(n),
            "M2": round(float(m2), 4)
        }

    def analyze_window(self):
        if len(self.buffer) < 10: return None

        # Convert to DataFrame
        df = pd.DataFrame(list(self.buffer), columns=['time', 'ear', 'gaze_x', 'gaze_y', 'mouth', 'pitch', 'yaw', 'keys', 'mouse'])
        
        # --- 1. CALCULATE RAW SCORES (0.0 - 1.0) PER FRAME ---
        # We apply the Gaussian scoring to every single frame in the buffer
        # to get a vector of "Momentary Engagement"
        
        # Pitch Score (Vertical Engagement)
        pitch_scores = df['pitch'].apply(lambda x: self.get_gaussian_score(x, 'pitch', self.config['pitch_sigma_tolerance']))
        
        # Yaw Score (Horizontal Engagement)
        yaw_scores = df['yaw'].apply(lambda x: self.get_gaussian_score(x, 'yaw', self.config['yaw_sigma_tolerance']))
        
        # Gaze Score (Stability)
        # For gaze, we assume the calibrated MEAN is the "center of work".
        # This is an approximation. Alternatively, we could score based on velocity (Lower velocity = Higher engagement)
        gaze_diffs = np.diff(df[['gaze_x', 'gaze_y']].values, axis=0)
        gaze_velocity = np.linalg.norm(gaze_diffs, axis=1)
        # Normalize velocity score: 1.0 if stationary, 0.0 if moving at max saccade speed
        velocity_scores = np.clip(1.0 - (gaze_velocity / (self.max_saccade_velocity + 1e-6)), 0.0, 1.0)

        # --- 2. CALCULATE STATISTICS ---
        # We now pack these score arrays into the requested format
        stats_output = {
            "pitch_engagement": self.calculate_window_stats(pitch_scores),
            "yaw_engagement": self.calculate_window_stats(yaw_scores),
            "visual_stability": self.calculate_window_stats(velocity_scores),
        }

        # --- 3. BOOLEAN FLAGS ---
        
        # Activity Check (Last 1 second)
        recent_keys = df['keys'].tail(30).sum()
        recent_mouse = df['mouse'].tail(30).sum()
        
        is_keyboard_active = bool(recent_keys > 0)
        is_mouse_active = bool(recent_mouse > 0)

        # --- 4. PHONE CHECKING MODE (The Quad-Lock) ---
        # Logic: 
        # 1. No Mouse AND No Keyboard
        # 2. Pitch is consistently LOW (Looking down) -> Low Engagement Score
        # 3. EAR is consistently LOW (Eyelids lowered) -> Low "Alertness" Score
        
        avg_pitch_score = stats_output["pitch_engagement"]["mean"]
        
        # EAR Check: Is it significantly below baseline?
        # We check if the current window mean is < (Baseline - 2*Sigma)
        current_ear_mean = df['ear'].mean()
        ear_threshold = self.calib_mean['ear'] - (2 * self.calib_std['ear'])
        is_ear_low = current_ear_mean < ear_threshold

        # Phone Trigger
        # If Pitch Engagement is LOW (meaning looking away/down) AND No Input AND Low EAR
        phone_checking_mode = (
            not is_mouse_active and 
            not is_keyboard_active and 
            (avg_pitch_score < 0.3) and # Score drops when deviating
            is_ear_low
        )

        return {
            "timestamp": time.time(),
            "metrics": stats_output,
            "flags": {
                "mouse_movement": is_mouse_active,
                "keyboard_stroke": is_keyboard_active,
                "phone_checking_mode": phone_checking_mode
            }
        }

    def run(self):
        print("🧠 NeuroAnalyzer Active. Outputting Statistical JSON...")
        
        # Find latest file
        while True:
            files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("telemetry_")])
            if files:
                current_file = os.path.join(DATA_DIR, files[-1])
                break
            time.sleep(1)

        with open(current_file, 'r') as f:
            f.readline() # Header
            while True:
                self.check_feedback_updates()
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                
                try:
                    parts = line.strip().split(',')
                    if len(parts) < 9: continue
                    self.buffer.append([float(x) for x in parts])
                    
                    # Analyze every 10 frames (approx 3 times/sec)
                    if len(self.buffer) % 10 == 0:
                        result = self.analyze_window()
                        if result:
                            print(json.dumps(result))
                            sys.stdout.flush()
                except ValueError: continue

if __name__ == "__main__":
    NeuroAnalyzer().run()




import pandas as pd
import numpy as np
import time
import json
import os
import sys
from collections import deque

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "threshold_feedback.json")

# Default Thresholds
DEFAULT_CONFIG = {
    "yaw_sigma_tolerance": 2.0,
    "pitch_sigma_tolerance": 3.0,
    "blink_sigma_tolerance": 2.5,
    "gaze_velocity_threshold": 1.5
}

class NeuroAnalyzer:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.calib_mean = {}
        self.calib_std = {}
        self.max_saccade_velocity = 0.0
        
        # Buffer: 10 seconds of data @ 30 FPS = 300 frames
        self.window_size = 300
        self.buffer = deque(maxlen=self.window_size)
        
        if not self.load_calibration():
            sys.exit(1)

    def load_calibration(self):
        conv_path = os.path.join(DATA_DIR, "calibration_convergent.csv")
        div_path = os.path.join(DATA_DIR, "calibration_divergent.csv")
        if not os.path.exists(conv_path) or not os.path.exists(div_path): return False

        try:
            df_conv = pd.read_csv(conv_path)
            for m in ['pitch', 'yaw', 'ear', 'gaze_x', 'gaze_y']:
                self.calib_mean[m] = df_conv[m].mean()
                self.calib_std[m] = max(df_conv[m].std(), 0.001)

            df_div = pd.read_csv(div_path)
            if not df_div.empty:
                gaze_diffs = np.diff(df_div[['gaze_x', 'gaze_y']].values, axis=0)
                velocities = np.linalg.norm(gaze_diffs, axis=1)
                self.max_saccade_velocity = np.percentile(velocities, 95)
            else:
                self.max_saccade_velocity = 0.5
            
            return True
        except: return False

    def check_feedback_updates(self):
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    updates = json.load(f)
                    for k, v in updates.items():
                        if k in self.config: self.config[k] = float(v)
            except: pass

    def get_gaussian_score(self, value, metric, tolerance_sigma):
        mu = self.calib_mean.get(metric, 0.5)
        sigma = self.calib_std.get(metric, 0.1)
        z = abs(value - mu) / sigma
        score = np.exp(-(z**2) / (2 * (tolerance_sigma**2)))
        return float(score)

    def calculate_window_stats(self, values):
        arr = np.array(values)
        if len(arr) == 0: return {"mean": 0.0, "std_dev": 0.0, "n": 0, "M2": 0.0}
        mean = np.mean(arr)
        m2 = np.sum((arr - mean)**2)
        std_dev = np.sqrt(m2 / len(arr))
        return {
            "mean": round(float(mean), 4),
            "std_dev": round(float(std_dev), 4),
            "n": int(len(arr)),
            "M2": round(float(m2), 4)
        }

    def analyze_window(self):
        if len(self.buffer) < 30: return None

        try:
            df = pd.DataFrame(list(self.buffer), columns=['time', 'ear', 'gaze_x', 'gaze_y', 'mouth', 'pitch', 'yaw', 'keys', 'mouse'])
            
            # 1. Engagement Scores
            pitch_scores = df['pitch'].apply(lambda x: self.get_gaussian_score(x, 'pitch', self.config['pitch_sigma_tolerance']))
            yaw_scores = df['yaw'].apply(lambda x: self.get_gaussian_score(x, 'yaw', self.config['yaw_sigma_tolerance']))
            
            gaze_diffs = np.diff(df[['gaze_x', 'gaze_y']].values, axis=0)
            gaze_velocity = np.linalg.norm(gaze_diffs, axis=1)
            velocity_scores = np.clip(1.0 - (gaze_velocity / (self.max_saccade_velocity + 1e-6)), 0.0, 1.0)

            stats_output = {
                "pitch_engagement": self.calculate_window_stats(pitch_scores),
                "yaw_engagement": self.calculate_window_stats(yaw_scores),
                "visual_stability": self.calculate_window_stats(velocity_scores),
            }

            # 2. Cognitive State
            gaze_variance = df['gaze_x'].std() + df['gaze_y'].std()
            mouth_activity = df['mouth'].mean()
            current_ear_mean = df['ear'].mean()
            ear_threshold = self.calib_mean['ear'] - (2 * self.calib_std['ear'])
            is_ear_low = current_ear_mean < ear_threshold
            
            focus_score = (1.0 / (1.0 + gaze_variance)) * 0.6 + (1.0 / (1.0 + mouth_activity)) * 0.4
            stim_score = (1.0 / (1.0 + gaze_variance)) * 0.4 + min(1.0, mouth_activity * 20) * 0.6
            stress_score = (1.0 - stats_output["visual_stability"]["mean"]) * 0.6 + (1.0 if is_ear_low else 0.0) * 0.4

            # 3. Flags
            recent_keys = df['keys'].tail(30).sum()
            recent_mouse = df['mouse'].tail(30).sum()
            avg_pitch_score = stats_output["pitch_engagement"]["mean"]
            
            phone_checking_mode = (
                recent_mouse == 0 and 
                recent_keys == 0 and 
                (avg_pitch_score < 0.3) and 
                is_ear_low
            )

            return {
                "timestamp": time.time(),
                "metrics": stats_output,
                "state_probabilities": {
                    "focus": round(focus_score, 2),
                    "stimming": round(stim_score, 2),
                    "stress": round(stress_score, 2)
                },
                "flags": {
                    "mouse_movement": bool(recent_mouse > 0),
                    "keyboard_stroke": bool(recent_keys > 0),
                    "phone_checking_mode": bool(phone_checking_mode)
                }
            }
        except: return None

    def run(self):
        # 1. Wait for file
        current_file = None
        while True:
            try:
                files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("telemetry_")])
                if files:
                    current_file = os.path.join(DATA_DIR, files[-1])
                    break
                time.sleep(1)
            except: time.sleep(1)

        # 2. Open and Seek to End (Tail Mode)
        with open(current_file, 'r') as f:
            # --- THE FIX IS HERE ---
            f.seek(0, 2) # Jump to end of file
            
            while True:
                self.check_feedback_updates()
                
                line = f.readline()
                if not line:
                    time.sleep(0.05) # Wait for new data
                    continue
                
                try:
                    parts = line.strip().split(',')
                    if len(parts) < 9: continue
                    
                    data_point = [float(x) for x in parts]
                    self.buffer.append(data_point)
                    
                    if len(self.buffer) % 10 == 0:
                        result = self.analyze_window()
                        if result:
                            print(json.dumps(result))
                            sys.stdout.flush()
                except: continue

if __name__ == "__main__":
    NeuroAnalyzer().run()





import pandas as pd
import numpy as np
import time
import json
import os
import sys
from collections import deque

# --- CONFIGURATION: ABSOLUTE PATH ANCHORING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "threshold_feedback.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "neuro_output.json")
CONV_CALIB = os.path.join(DATA_DIR, "calibration_convergent.csv")
DIV_CALIB = os.path.join(DATA_DIR, "calibration_divergent.csv")

# Default Thresholds
DEFAULT_CONFIG = {
    "yaw_sigma_tolerance": 2.0,
    "pitch_sigma_tolerance": 3.0,
    "blink_sigma_tolerance": 2.5,
    "gaze_velocity_threshold": 1.5
}

class NeuroAnalyzer:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.calib_mean = {}
        self.calib_std = {}
        self.max_saccade_velocity = 0.5 
        self.window_size = 300
        self.buffer = deque(maxlen=self.window_size)
        self.load_calibration()

    def load_calibration(self):
        if not os.path.exists(CONV_CALIB) or not os.path.exists(DIV_CALIB): return False

        try:
            df_conv = pd.read_csv(CONV_CALIB)
            if df_conv.empty: return False
            
            for m in ['pitch', 'yaw', 'ear', 'gaze_x', 'gaze_y']:
                self.calib_mean[m] = df_conv[m].mean()
                self.calib_std[m] = max(df_conv[m].std(), 0.001)

            df_div = pd.read_csv(DIV_CALIB)
            if not df_div.empty and len(df_div) > 1:
                gaze_diffs = np.diff(df_div[['gaze_x', 'gaze_y']].values, axis=0)
                velocities = np.linalg.norm(gaze_diffs, axis=1)
                self.max_saccade_velocity = np.percentile(velocities, 95)
            
            print(f"✅ [ANALYZER] Calibration Loaded. Baseline Pitch: {self.calib_mean['pitch']:.2f}")
            return True
        except: return False

    def check_feedback_updates(self):
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    updates = json.load(f)
                    for k, v in updates.items():
                        if k in self.config: self.config[k] = float(v)
            except: pass

    def get_gaussian_score(self, value, metric, tolerance_sigma):
        mu = self.calib_mean.get(metric, 0.5)
        sigma = self.calib_std.get(metric, 0.1)
        z = abs(value - mu) / sigma
        score = np.exp(-(z**2) / (2 * (tolerance_sigma**2)))
        return float(score)

    def calculate_window_stats(self, values):
        arr = np.array(values)
        if len(arr) == 0: return {"mean": 0.0, "std_dev": 0.0, "n": 0, "M2": 0.0}
        mean = np.mean(arr)
        m2 = np.sum((arr - mean)**2)
        std_dev = np.sqrt(m2 / len(arr))
        return {
            "mean": round(float(mean), 4),
            "std_dev": round(float(std_dev), 4),
            "n": int(len(arr)),
            "M2": round(float(m2), 4)
        }

    def analyze_window(self):
        if len(self.buffer) < 30: return None

        try:
            df = pd.DataFrame(list(self.buffer), columns=['time', 'ear', 'gaze_x', 'gaze_y', 'mouth', 'pitch', 'yaw', 'keys', 'mouse'])
            
            # Engagement
            pitch_scores = df['pitch'].apply(lambda x: self.get_gaussian_score(x, 'pitch', self.config['pitch_sigma_tolerance']))
            yaw_scores = df['yaw'].apply(lambda x: self.get_gaussian_score(x, 'yaw', self.config['yaw_sigma_tolerance']))
            
            gaze_diffs = np.diff(df[['gaze_x', 'gaze_y']].values, axis=0)
            gaze_velocity = np.linalg.norm(gaze_diffs, axis=1)
            velocity_scores = np.clip(1.0 - (gaze_velocity / (self.max_saccade_velocity + 1e-6)), 0.0, 1.0)

            stats_output = {
                "pitch_engagement": self.calculate_window_stats(pitch_scores),
                "yaw_engagement": self.calculate_window_stats(yaw_scores),
                "visual_stability": self.calculate_window_stats(velocity_scores),
            }

            # Cognitive State
            gaze_variance = df['gaze_x'].std() + df['gaze_y'].std()
            mouth_activity = df['mouth'].mean()
            current_ear_mean = df['ear'].mean()
            ear_threshold = self.calib_mean['ear'] - (2 * self.calib_std['ear'])
            is_ear_low = current_ear_mean < ear_threshold
            
            focus_score = (1.0 / (1.0 + gaze_variance)) * 0.6 + (1.0 / (1.0 + mouth_activity)) * 0.4
            stim_score = (1.0 / (1.0 + gaze_variance)) * 0.4 + min(1.0, mouth_activity * 20) * 0.6
            stress_score = (1.0 - stats_output["visual_stability"]["mean"]) * 0.6 + (1.0 if is_ear_low else 0.0) * 0.4

            # Flags
            recent_keys = df['keys'].tail(30).sum()
            recent_mouse = df['mouse'].tail(30).sum()
            avg_pitch_score = stats_output["pitch_engagement"]["mean"]
            
            phone_checking_mode = (
                recent_mouse == 0 and 
                recent_keys == 0 and 
                (avg_pitch_score < 0.3) and 
                is_ear_low
            )

            return {
                "timestamp": time.time(),
                "metrics": stats_output,
                "state_probabilities": {
                    "focus": round(focus_score, 2),
                    "stimming": round(stim_score, 2),
                    "stress": round(stress_score, 2)
                },
                "flags": {
                    "mouse_movement": bool(recent_mouse > 0),
                    "keyboard_stroke": bool(recent_keys > 0),
                    "phone_checking_mode": bool(phone_checking_mode)
                }
            }
        except Exception: return None

    def run(self):
        print("🧠 [ANALYZER] WAITING FOR TELEMETRY STREAM...")
        sys.stdout.flush()
        
        # 1. FIND FILE
        current_file = None
        while True:
            try:
                if not os.path.exists(DATA_DIR):
                    time.sleep(1)
                    continue
                    
                files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("telemetry_")])
                if files:
                    current_file = os.path.join(DATA_DIR, files[-1])
                    print(f"📂 [ANALYZER] TRACKING FILE: {files[-1]}")
                    sys.stdout.flush()
                    break
                time.sleep(1)
            except: time.sleep(1)

        # 2. READ LOOP
        with open(current_file, 'r') as f:
            f.seek(0, 2) # Jump to end
            
            while True:
                self.check_feedback_updates()
                line = f.readline()
                
                if not line:
                    time.sleep(0.05)
                    continue
                
                try:
                    parts = line.strip().split(',')
                    if len(parts) < 9: continue
                    
                    self.buffer.append([float(x) for x in parts])
                    
                    if len(self.buffer) >= 30 and len(self.buffer) % 10 == 0:
                        result = self.analyze_window()
                        if result:
                            # --- APPEND MODE ---
                            with open(OUTPUT_JSON, 'a') as jf:
                                json.dump(result, jf)
                                jf.write('\n')
                            
                            print(json.dumps(result))
                            sys.stdout.flush()
                            
                except ValueError: continue

if __name__ == "__main__":
    NeuroAnalyzer().run()

    




import pandas as pd
import numpy as np
import time
import json
import os
import sys
import traceback
from collections import deque

# --- CONFIGURATION: ABSOLUTE PATH ANCHORING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "threshold_feedback.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "neuro_output.json")
CONV_CALIB = os.path.join(DATA_DIR, "calibration_convergent.csv")
DIV_CALIB = os.path.join(DATA_DIR, "calibration_divergent.csv")

# Default Thresholds
DEFAULT_CONFIG = {
    "yaw_sigma_tolerance": 2.0,
    "pitch_sigma_tolerance": 3.0,
    "blink_sigma_tolerance": 2.5,
    "gaze_velocity_threshold": 1.5
}

class NeuroAnalyzer:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.calib_mean = {}
        self.calib_std = {}
        self.max_saccade_velocity = 0.5 
        self.window_size = 300
        self.buffer = deque(maxlen=self.window_size)
        self.load_calibration()

    def load_calibration(self):
        if not os.path.exists(CONV_CALIB) or not os.path.exists(DIV_CALIB): return False
        try:
            df_conv = pd.read_csv(CONV_CALIB)
            if df_conv.empty: return False
            for m in ['pitch', 'yaw', 'ear', 'gaze_x', 'gaze_y']:
                self.calib_mean[m] = df_conv[m].mean()
                self.calib_std[m] = max(df_conv[m].std(), 0.001) # Prevent ZeroDivision

            # Add explicit default for mouth if missing
            if 'mouth_dist' not in self.calib_mean:
                self.calib_mean['mouth_dist'] = 0.0
                self.calib_std['mouth_dist'] = 0.01

            df_div = pd.read_csv(DIV_CALIB)
            if not df_div.empty and len(df_div) > 1:
                gaze_diffs = np.diff(df_div[['gaze_x', 'gaze_y']].values, axis=0)
                velocities = np.linalg.norm(gaze_diffs, axis=1)
                self.max_saccade_velocity = np.percentile(velocities, 95)
            
            print(f"✅ [ANALYZER] Calibration Loaded. Baseline Pitch: {self.calib_mean.get('pitch', 0):.2f}")
            return True
        except: return False

    def check_feedback_updates(self):
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    updates = json.load(f)
                    for k, v in updates.items():
                        if k in self.config: self.config[k] = float(v)
            except: pass

    def get_gaussian_score(self, value, metric, tolerance_sigma):
        mu = self.calib_mean.get(metric, 0.5)
        sigma = self.calib_std.get(metric, 0.1)
        z = abs(value - mu) / sigma
        score = np.exp(-(z**2) / (2 * (tolerance_sigma**2)))
        return float(score)

    def calculate_window_stats(self, values):
        arr = np.array(values)
        if len(arr) == 0: return {"mean": 0.0, "std_dev": 0.0, "n": 0, "M2": 0.0}
        m = np.mean(arr)
        m2 = np.sum((arr - m)**2)
        std_dev = np.sqrt(m2 / len(arr))
        return {
            "mean": round(float(m), 4),
            "std_dev": round(float(std_dev), 4),
            "n": int(len(arr)),
            "M2": round(float(m2), 4)
        }

    def analyze_window(self):
        if len(self.buffer) < 30: return None

        try:
            # Columns match Watcher output (9 cols)
            df = pd.DataFrame(list(self.buffer), columns=['time', 'ear', 'gaze_x', 'gaze_y', 'mouth', 'pitch', 'yaw', 'keys', 'mouse'])
            
            # --- 1. NORMALIZED METRICS (0.0 to 1.0) ---
            
            # Engagement (Gatekeeper)
            pitch_score = self.get_gaussian_score(df['pitch'].mean(), 'pitch', self.config['pitch_sigma_tolerance'])
            yaw_score = self.get_gaussian_score(df['yaw'].mean(), 'yaw', self.config['yaw_sigma_tolerance'])
            engagement_gate = pitch_score * yaw_score 

            # Visual Stability
            gaze_diffs = np.diff(df[['gaze_x', 'gaze_y']].values, axis=0)
            if len(gaze_diffs) > 0:
                avg_velocity = np.linalg.norm(gaze_diffs, axis=1).mean()
                visual_stability = max(0.0, 1.0 - (avg_velocity / (self.max_saccade_velocity + 0.001)))
            else:
                visual_stability = 1.0

            # Mouth Activity (Fidgeting)
            # Safe calibration access
            calib_mouth = self.calib_mean.get('mouth_dist', 0.0) 
            mouth_z = (df['mouth'].mean() - calib_mouth) / 0.05
            mouth_activity = min(1.0, max(0.0, mouth_z))

            # Blink Stress
            avg_ear = df['ear'].mean()
            baseline_ear = self.calib_mean.get('ear', 0.3)
            blink_stress = 1.0 if avg_ear < (baseline_ear - 0.05) else 0.0

            # --- 2. STATE LOGIC ---
            
            # FOCUS: Looking + Stable Eyes + Quiet Mouth
            focus_score = engagement_gate * visual_stability * (1.0 - mouth_activity)
            
            # STIMMING: Looking + Stable Eyes + MOVING Mouth
            stimming_score = engagement_gate * visual_stability * mouth_activity * 2.0
            
            # STRESS: Chaotic Eyes OR High Blinking
            stress_score = (1.0 - visual_stability) + blink_stress + (1.0 - engagement_gate) * 0.5

            # Clip 0-1
            focus_score = min(1.0, max(0.0, focus_score))
            stimming_score = min(1.0, max(0.0, stimming_score))
            stress_score = min(1.0, max(0.0, stress_score))

            # --- 3. FLAGS & OUTPUT ---
            p_scores = df['pitch'].apply(lambda x: self.get_gaussian_score(x, 'pitch', 3.0))
            
            recent_keys = df['keys'].tail(30).sum()
            recent_mouse = df['mouse'].tail(30).sum()
            
            phone_checking_mode = (
                recent_mouse == 0 and 
                recent_keys == 0 and 
                pitch_score < 0.3 # Looking down
            )

            result = {
                "timestamp": time.time(),
                "metrics": {
                    "pitch_engagement": self.calculate_window_stats(p_scores),
                    "visual_stability": {"mean": round(visual_stability, 2)}
                },
                "state_probabilities": {
                    "focus": round(focus_score, 2),
                    "stimming": round(stimming_score, 2),
                    "stress": round(stress_score, 2)
                },
                "flags": {
                    "mouse_movement": bool(recent_mouse > 0),
                    "keyboard_stroke": bool(recent_keys > 0),
                    "phone_checking_mode": bool(phone_checking_mode)
                }
            }
            return result

        except Exception:
            # PRINT THE ERROR so we know what broke!
            print("❌ [ANALYZER MATH ERROR]:")
            traceback.print_exc()
            return None

    def run(self):
        print("🧠 [ANALYZER] WAITING FOR TELEMETRY STREAM...")
        sys.stdout.flush()
        
        current_file = None
        while True:
            try:
                if not os.path.exists(DATA_DIR):
                    time.sleep(1); continue
                files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("telemetry_")])
                if files:
                    current_file = os.path.join(DATA_DIR, files[-1])
                    print(f"📂 [ANALYZER] TRACKING FILE: {files[-1]}")
                    sys.stdout.flush()
                    break
                time.sleep(1)
            except: time.sleep(1)

        with open(current_file, 'r') as f:
            f.seek(0, 2)
            
            while True:
                self.check_feedback_updates()
                line = f.readline()
                
                if not line:
                    time.sleep(0.05); continue
                
                try:
                    parts = line.strip().split(',')
                    if len(parts) < 9: continue
                    
                    self.buffer.append([float(x) for x in parts])
                    
                    if len(self.buffer) >= 30 and len(self.buffer) % 10 == 0:
                        result = self.analyze_window()
                        if result:
                            # Append JSON
                            with open(OUTPUT_JSON, 'a') as jf:
                                json.dump(result, jf)
                                jf.write('\n')
                            
                            # Print JSON
                            print(json.dumps(result))
                            sys.stdout.flush()
                            
                except ValueError: continue

if __name__ == "__main__":
    NeuroAnalyzer().run()





import pandas as pd
import numpy as np
import time
import json
import os
import sys
import traceback
from collections import deque

# --- CONFIGURATION: ABSOLUTE PATH ANCHORING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "threshold_feedback.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "neuro_output.json")
CONV_CALIB = os.path.join(DATA_DIR, "calibration_convergent.csv")
DIV_CALIB = os.path.join(DATA_DIR, "calibration_divergent.csv")

# ==============================================================================
# 🎛️ LOGIC THRESHOLDS (Standard Deviations)
# ==============================================================================
CONFIG = {
    # ENGAGEMENT: Stricter Yaw to force head tracking influence
    "pitch_sigma_limit": 3.0,  
    "yaw_sigma_limit": 1.5, # LOWERED from 2.5 to make Yaw matter more!

    # STABILITY: How chaotic must eyes be to count as "Stress"?
    "velocity_sigma_limit": 2.0, 

    # MOUTH: How much movement counts as "Stimming"?
    "mouth_sigma_threshold": 3.0, 

    # BLINKING: How low must EAR drop to count as a "Stress Blink"?
    "blink_sigma_threshold": 2.0
}
# ==============================================================================

class NeuroAnalyzer:
    def __init__(self):
        self.calib_mean = {}
        self.calib_std = {}
        self.max_saccade_velocity = 0.5 
        self.window_size = 300
        self.buffer = deque(maxlen=self.window_size)
        self.load_calibration()

    def load_calibration(self):
        if not os.path.exists(CONV_CALIB) or not os.path.exists(DIV_CALIB): return False
        try:
            df_conv = pd.read_csv(CONV_CALIB)
            if df_conv.empty: return False
            
            for m in ['pitch', 'yaw', 'ear', 'gaze_x', 'gaze_y']:
                self.calib_mean[m] = df_conv[m].mean()
                self.calib_std[m] = max(df_conv[m].std(), 0.001)

            if 'mouth_dist' not in self.calib_mean:
                self.calib_mean['mouth_dist'] = 0.0
                self.calib_std['mouth_dist'] = 0.005

            df_div = pd.read_csv(DIV_CALIB)
            if not df_div.empty and len(df_div) > 1:
                gaze_diffs = np.diff(df_div[['gaze_x', 'gaze_y']].values, axis=0)
                velocities = np.linalg.norm(gaze_diffs, axis=1)
                self.max_saccade_velocity = np.percentile(velocities, 95)
            
            print(f"✅ [ANALYZER] Calibration Loaded. Baseline Yaw: {self.calib_mean.get('yaw', 0):.2f}")
            return True
        except: return False

    def check_feedback_updates(self):
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    updates = json.load(f)
                    for k, v in updates.items():
                        if k in CONFIG: CONFIG[k] = float(v)
            except: pass

    def get_z_score(self, value, metric):
        mu = self.calib_mean.get(metric, 0.0)
        sigma = self.calib_std.get(metric, 1.0)
        return abs(value - mu) / sigma

    def calculate_window_stats(self, values):
        arr = np.array(values)
        if len(arr) == 0: return {"mean": 0.0, "std_dev": 0.0, "n": 0, "M2": 0.0}
        m = np.mean(arr)
        m2 = np.sum((arr - m)**2)
        return {"mean": round(float(m), 4), "std_dev": round(float(np.sqrt(m2/len(arr))), 4), "n": len(arr), "M2": round(float(m2), 4)}

    def analyze_window(self):
        if len(self.buffer) < 30: return None

        try:
            df = pd.DataFrame(list(self.buffer), columns=['time', 'ear', 'gaze_x', 'gaze_y', 'mouth', 'pitch', 'yaw', 'keys', 'mouse'])
            
            # --- 1. Z-SCORES ---
            pitch_z = self.get_z_score(df['pitch'].mean(), 'pitch')
            yaw_z = self.get_z_score(df['yaw'].mean(), 'yaw')
            
            # Engagement Probability (0.0 to 1.0)
            # If Z > Limit, this drops rapidly. 
            eng_p = max(0.0, 1.0 - (pitch_z / CONFIG['pitch_sigma_limit']))
            eng_y = max(0.0, 1.0 - (yaw_z / CONFIG['yaw_sigma_limit']))
            
            # IMPORTANT: Multiplicative! If you look Left/Right (Yaw), Engagement dies.
            engagement_score = eng_p * eng_y 

            # Stability
            gaze_diffs = np.diff(df[['gaze_x', 'gaze_y']].values, axis=0)
            if len(gaze_diffs) > 0:
                avg_vel = np.linalg.norm(gaze_diffs, axis=1).mean()
                norm_vel = avg_vel / (self.max_saccade_velocity + 0.001)
                stability = max(0.0, 1.0 - (norm_vel / CONFIG['velocity_sigma_limit']))
            else:
                stability = 1.0

            # Mouth
            mouth_z = self.get_z_score(df['mouth'].mean(), 'mouth_dist')
            mouth_activity = max(0.0, min(1.0, (mouth_z - 1.0) / CONFIG['mouth_sigma_threshold']))

            # Blink
            ear_z = (self.calib_mean.get('ear', 0.3) - df['ear'].mean()) / self.calib_std.get('ear', 0.05)
            is_stressed_eyes = 1.0 if ear_z > CONFIG['blink_sigma_threshold'] else 0.0

            # --- 2. INPUT DETECTION ---
            recent_keys = df['keys'].tail(30).sum()
            recent_mouse = df['mouse'].tail(30).sum()
            is_working = (recent_keys > 0 or recent_mouse > 0)

            # --- 3. MIXING BOARD ---
            
            # A. FOCUS SCORE
            # Base = Engagement * Stability
            focus_score = engagement_score * ((stability * 0.7) + 0.3)
            
            # B. STIMMING SCORE
            stimming_score = (mouth_activity * 0.8) + (stability * 0.2)
            
            # C. STRESS SCORE
            # If you are looking away (low engagement), Stress goes UP
            stress_score = (1.0 - stability) * 0.5 + (is_stressed_eyes * 0.3) + ((1.0 - engagement_score) * 0.2)

            # --- 4. THE INPUT OVERRIDE (USER REQUEST) ---
            if is_working:
                # If typing, BOOST Focus
                focus_score = min(1.0, focus_score + 0.3)
                # If typing, REDUCE Stress/Stim (Productive output usually means less chaotic stress)
                stimming_score *= 0.5
                stress_score *= 0.5

            # If looking away (Yaw/Pitch deviations), penalize Focus heavily
            if engagement_score < 0.5:
                focus_score *= 0.2 # Massive penalty for looking away
                stress_score += 0.2 # Looking away often implies distraction/stress scanning

            # Clamp
            focus_score = round(min(1.0, max(0.0, focus_score)), 2)
            stimming_score = round(min(1.0, max(0.0, stimming_score)), 2)
            stress_score = round(min(1.0, max(0.0, stress_score)), 2)

            # --- 5. OUTPUT ---
            # Phone Check Logic
            phone_mode = (not is_working and pitch_z > CONFIG['pitch_sigma_limit'])

            result = {
                "timestamp": time.time(),
                "metrics": {
                    "engagement_score": round(engagement_score, 2),
                    "stability_score": round(stability, 2),
                    "mouth_z": round(mouth_z, 2)
                },
                "state_probabilities": {
                    "focus": focus_score,
                    "stimming": stimming_score,
                    "stress": stress_score
                },
                "flags": {
                    "mouse_movement": bool(recent_mouse > 0),
                    "keyboard_stroke": bool(recent_keys > 0),
                    "phone_checking_mode": bool(phone_mode)
                }
            }
            return result

        except Exception:
            traceback.print_exc()
            return None

    def run(self):
        print("🧠 [ANALYZER] WAITING FOR TELEMETRY STREAM...")
        sys.stdout.flush()
        
        current_file = None
        while True:
            try:
                if not os.path.exists(DATA_DIR):
                    time.sleep(1); continue
                files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("telemetry_")])
                if files:
                    current_file = os.path.join(DATA_DIR, files[-1])
                    print(f"📂 [ANALYZER] TRACKING FILE: {files[-1]}")
                    sys.stdout.flush()
                    break
                time.sleep(1)
            except: time.sleep(1)

        with open(current_file, 'r') as f:
            f.seek(0, 2)
            while True:
                self.check_feedback_updates()
                line = f.readline()
                if not line:
                    time.sleep(0.05); continue
                try:
                    parts = line.strip().split(',')
                    if len(parts) < 9: continue
                    self.buffer.append([float(x) for x in parts])
                    if len(self.buffer) >= 30 and len(self.buffer) % 10 == 0:
                        result = self.analyze_window()
                        if result:
                            with open(OUTPUT_JSON, 'a') as jf:
                                json.dump(result, jf); jf.write('\n')
                            print(json.dumps(result)); sys.stdout.flush()
                except ValueError: continue

if __name__ == "__main__":
    NeuroAnalyzer().run()


import pandas as pd
import numpy as np
import time
import json
import os
import sys
import traceback
from collections import deque

# --- CONFIGURATION: ABSOLUTE PATH ANCHORING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "threshold_feedback.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "neuro_output.json")
CONV_CALIB = os.path.join(DATA_DIR, "calibration_convergent.csv")
DIV_CALIB = os.path.join(DATA_DIR, "calibration_divergent.csv")

# ==============================================================================
# LOGIC THRESHOLDS (Standard Deviations)
# ==============================================================================
CONFIG = {
    "pitch_sigma_limit": 3.0,  
    "yaw_sigma_limit": 2.0, 
    "velocity_sigma_limit": 2.5, 
    "mouth_sigma_threshold": 3.0, 
    "blink_sigma_threshold": 2.0
}

# PHONE CHECKER SPECIFIC TUNING
PHONE_PITCH_THRESHOLD = 2.5  # Z-Score: How far down is "Down"?
PHONE_EAR_THRESHOLD = 1.5    # Z-Score: How much do eyes close when looking down?
PHONE_TRIGGER_FRAMES = 15    # ~0.5 seconds to confirm
# ==============================================================================

class NeuroAnalyzer:
    def __init__(self):
        self.calib_mean = {}
        self.calib_std = {}
        self.max_saccade_velocity = 0.5 
        self.window_size = 300
        self.buffer = deque(maxlen=self.window_size)
        self.phone_counter = 0 # Instant counter
        self.load_calibration()

    def load_calibration(self):
        if not os.path.exists(CONV_CALIB) or not os.path.exists(DIV_CALIB): return False
        try:
            df_conv = pd.read_csv(CONV_CALIB)
            if df_conv.empty: return False
            
            for m in ['pitch', 'yaw', 'ear', 'gaze_x', 'gaze_y']:
                self.calib_mean[m] = df_conv[m].mean()
                self.calib_std[m] = max(df_conv[m].std(), 0.001)

            if 'mouth_dist' not in self.calib_mean:
                self.calib_mean['mouth_dist'] = 0.0
                self.calib_std['mouth_dist'] = 0.005

            df_div = pd.read_csv(DIV_CALIB)
            if not df_div.empty and len(df_div) > 1:
                gaze_diffs = np.diff(df_div[['gaze_x', 'gaze_y']].values, axis=0)
                velocities = np.linalg.norm(gaze_diffs, axis=1)
                self.max_saccade_velocity = np.percentile(velocities, 95)
            
            print(f"✅ [ANALYZER] Calibration Loaded. Baseline Pitch: {self.calib_mean.get('pitch', 0):.2f}")
            return True
        except: return False

    def check_feedback_updates(self):
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    updates = json.load(f)
                    for k, v in updates.items():
                        if k in CONFIG: CONFIG[k] = float(v)
            except: pass

    def get_z_score(self, value, metric):
        mu = self.calib_mean.get(metric, 0.0)
        sigma = self.calib_std.get(metric, 1.0)
        return abs(value - mu) / sigma

    def calculate_window_stats(self, values):
        arr = np.array(values)
        if len(arr) == 0: return {"mean": 0.0, "std_dev": 0.0, "n": 0, "M2": 0.0}
        m = np.mean(arr)
        m2 = np.sum((arr - m)**2)
        return {"mean": round(float(m), 4), "std_dev": round(float(np.sqrt(m2/len(arr))), 4), "n": len(arr), "M2": round(float(m2), 4)}

    def analyze_window(self):
        if len(self.buffer) < 30: return None

        try:
            df = pd.DataFrame(list(self.buffer), columns=['time', 'ear', 'gaze_x', 'gaze_y', 'mouth', 'pitch', 'yaw', 'keys', 'mouse'])
            
            # --- 1. CORE METRICS (Full Window) ---
            pitch_z = self.get_z_score(df['pitch'].mean(), 'pitch')
            yaw_z = self.get_z_score(df['yaw'].mean(), 'yaw')
            
            eng_p = max(0.0, 1.0 - (pitch_z / CONFIG['pitch_sigma_limit']))
            eng_y = max(0.0, 1.0 - (yaw_z / CONFIG['yaw_sigma_limit']))
            engagement_score = eng_p * eng_y 

            gaze_diffs = np.diff(df[['gaze_x', 'gaze_y']].values, axis=0)
            avg_vel = np.linalg.norm(gaze_diffs, axis=1).mean() if len(gaze_diffs) > 0 else 0
            norm_vel = avg_vel / (self.max_saccade_velocity + 0.001)
            stability = max(0.0, 1.0 - (norm_vel / CONFIG['velocity_sigma_limit']))

            mouth_z = self.get_z_score(df['mouth'].mean(), 'mouth_dist')
            mouth_activity = max(0.0, min(1.0, (mouth_z - 1.0) / CONFIG['mouth_sigma_threshold']))

            ear_z = (self.calib_mean.get('ear', 0.3) - df['ear'].mean()) / self.calib_std.get('ear', 0.05)
            is_stressed_eyes = 1.0 if ear_z > CONFIG['blink_sigma_threshold'] else 0.0

            # --- 2. INPUT DETECTION ---
            recent_keys = df['keys'].tail(30).sum()
            recent_mouse = df['mouse'].tail(30).sum()
            is_working = (recent_keys > 0 or recent_mouse > 0)

            # --- 3. STATE LOGIC ---
            focus_score = engagement_score * ((stability * 0.7) + 0.3)
            stimming_score = (mouth_activity * 0.8) + (stability * 0.2)
            stress_score = (1.0 - stability) * 0.5 + (is_stressed_eyes * 0.3) + ((1.0 - engagement_score) * 0.2)

            if is_working:
                focus_score = min(1.0, focus_score + 0.3)
                stimming_score *= 0.5
                stress_score *= 0.5

            if engagement_score < 0.5:
                focus_score *= 0.2 
                stress_score += 0.2

            # --- 4. PHONE CHECKER LOGIC (INSTANT SNAP-BACK) ---
            # We look at the very last frame for "Current State"
            last_frame = self.buffer[-1]
            # [time, ear, gx, gy, mouth, pitch, yaw, keys, mouse]
            curr_pitch = last_frame[5]
            curr_ear = last_frame[1]
            
            # Instant Z-Scores
            curr_pitch_z = self.get_z_score(curr_pitch, 'pitch')
            # EAR Drop: (Baseline - Current) / Std
            curr_ear_drop_z = (self.calib_mean['ear'] - curr_ear) / self.calib_std['ear']
            
            # Logic: Down + Eyes Low + No Input
            looking_down = (curr_pitch_z > PHONE_PITCH_THRESHOLD)
            eyes_low = (curr_ear_drop_z > PHONE_EAR_THRESHOLD)
            
            # SNAP BACK: If looking up, reset counter immediately
            if not looking_down:
                self.phone_counter = 0
            # TRIGGER: If looking down AND eyes low AND no input
            elif looking_down and eyes_low and not is_working:
                self.phone_counter += 1
            else:
                # Ambiguous state (e.g. looking down but typing) -> Slow decay
                self.phone_counter = max(0, self.phone_counter - 1)
                
            is_phone_checking = (self.phone_counter > PHONE_TRIGGER_FRAMES)

            # --- 5. OUTPUT ---
            result = {
                "timestamp": time.time(),
                "metrics": {
                    "engagement_score": round(engagement_score, 2),
                    "stability_score": round(stability, 2),
                    "mouth_z": round(mouth_z, 2)
                },
                "state_probabilities": {
                    "focus": round(min(1.0, max(0.0, focus_score)), 2),
                    "stimming": round(min(1.0, max(0.0, stimming_score)), 2),
                    "stress": round(min(1.0, max(0.0, stress_score)), 2)
                },
                "flags": {
                    "mouse_movement": bool(recent_mouse > 0),
                    "keyboard_stroke": bool(recent_keys > 0),
                    "phone_checking_mode": bool(is_phone_checking)
                }
            }
            return result

        except Exception:
            traceback.print_exc()
            return None

    def run(self):
        print("🧠 [ANALYZER] WAITING FOR TELEMETRY STREAM...")
        sys.stdout.flush()
        
        current_file = None
        while True:
            try:
                if not os.path.exists(DATA_DIR):
                    time.sleep(1); continue
                files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("telemetry_")])
                if files:
                    current_file = os.path.join(DATA_DIR, files[-1])
                    print(f"📂 [ANALYZER] TRACKING FILE: {files[-1]}")
                    sys.stdout.flush()
                    break
                time.sleep(1)
            except: time.sleep(1)

        with open(current_file, 'r') as f:
            f.seek(0, 2)
            while True:
                self.check_feedback_updates()
                line = f.readline()
                if not line:
                    time.sleep(0.05); continue
                try:
                    parts = line.strip().split(',')
                    if len(parts) < 9: continue
                    self.buffer.append([float(x) for x in parts])
                    if len(self.buffer) >= 30 and len(self.buffer) % 10 == 0:
                        result = self.analyze_window()
                        if result:
                            with open(OUTPUT_JSON, 'a') as jf:
                                json.dump(result, jf); jf.write('\n')
                            print(json.dumps(result)); sys.stdout.flush()
                except ValueError: continue

if __name__ == "__main__":
    NeuroAnalyzer().run()

'''


import pandas as pd
import numpy as np
import time
import json
import os
import sys
import traceback
from collections import deque

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "threshold_feedback.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "neuro_output.json")
CONV_CALIB = os.path.join(DATA_DIR, "calibration_convergent.csv")
DIV_CALIB = os.path.join(DATA_DIR, "calibration_divergent.csv")

# LOGIC THRESHOLDS
CONFIG = {
    "pitch_sigma_limit": 3.0,  
    "yaw_sigma_limit": 2.0, 
    "velocity_sigma_limit": 2.5, 
    "mouth_sigma_threshold": 3.0, 
    "blink_sigma_threshold": 2.0
}

# PHONE CHECKER TUNING
PHONE_PITCH_THRESHOLD = 2.5  # Down look Z-Score
PHONE_EAR_THRESHOLD = 1.5    # Eye closing Z-Score
PHONE_YAW_THRESHOLD = 2.0    # Side look Z-Score (Left/Right)

# DURATIONS (Analyzer runs approx every 0.33s)
# 15 ticks * 0.33s = ~5 seconds for looking down
TRIGGER_TICKS_DOWN = 15 
# 23 ticks * 0.33s = ~7.5 seconds for looking side
TRIGGER_TICKS_SIDE = 23 

class NeuroAnalyzer:
    def __init__(self):
        self.calib_mean = {}
        self.calib_std = {}
        self.max_saccade_velocity = 0.5 
        self.window_size = 300
        self.buffer = deque(maxlen=self.window_size)
        
        # State Counters
        self.phone_down_counter = 0 
        self.phone_side_counter = 0
        
        self.load_calibration()

    def load_calibration(self):
        if not os.path.exists(CONV_CALIB) or not os.path.exists(DIV_CALIB): return False
        try:
            df_conv = pd.read_csv(CONV_CALIB)
            if df_conv.empty: return False
            for m in ['pitch', 'yaw', 'ear', 'gaze_x', 'gaze_y']:
                self.calib_mean[m] = df_conv[m].mean()
                self.calib_std[m] = max(df_conv[m].std(), 0.001)

            if 'mouth_dist' not in self.calib_mean:
                self.calib_mean['mouth_dist'] = 0.0
                self.calib_std['mouth_dist'] = 0.005

            df_div = pd.read_csv(DIV_CALIB)
            if not df_div.empty and len(df_div) > 1:
                gaze_diffs = np.diff(df_div[['gaze_x', 'gaze_y']].values, axis=0)
                velocities = np.linalg.norm(gaze_diffs, axis=1)
                self.max_saccade_velocity = np.percentile(velocities, 95)
            
            print(f"✅ [ANALYZER] Calibration Loaded.")
            return True
        except: return False

    def check_feedback_updates(self):
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    updates = json.load(f)
                    for k, v in updates.items():
                        if k in CONFIG: CONFIG[k] = float(v)
            except: pass

    def get_z_score(self, value, metric):
        mu = self.calib_mean.get(metric, 0.0)
        sigma = self.calib_std.get(metric, 1.0)
        return abs(value - mu) / sigma

    def calculate_window_stats(self, values):
        arr = np.array(values)
        if len(arr) == 0: return {"mean": 0.0, "std_dev": 0.0, "n": 0, "M2": 0.0}
        m = np.mean(arr)
        m2 = np.sum((arr - m)**2)
        return {"mean": round(float(m), 4), "std_dev": round(float(np.sqrt(m2/len(arr))), 4), "n": len(arr), "M2": round(float(m2), 4)}

    def analyze_window(self):
        if len(self.buffer) < 30: return None

        try:
            df = pd.DataFrame(list(self.buffer), columns=['time', 'ear', 'gaze_x', 'gaze_y', 'mouth', 'pitch', 'yaw', 'keys', 'mouse'])
            
            # --- 1. CORE METRICS ---
            pitch_z = self.get_z_score(df['pitch'].mean(), 'pitch')
            yaw_z = self.get_z_score(df['yaw'].mean(), 'yaw')
            eng_p = max(0.0, 1.0 - (pitch_z / CONFIG['pitch_sigma_limit']))
            eng_y = max(0.0, 1.0 - (yaw_z / CONFIG['yaw_sigma_limit']))
            engagement_score = eng_p * eng_y 

            gaze_diffs = np.diff(df[['gaze_x', 'gaze_y']].values, axis=0)
            avg_vel = np.linalg.norm(gaze_diffs, axis=1).mean() if len(gaze_diffs) > 0 else 0
            norm_vel = avg_vel / (self.max_saccade_velocity + 0.001)
            stability = max(0.0, 1.0 - (norm_vel / CONFIG['velocity_sigma_limit']))

            mouth_z = self.get_z_score(df['mouth'].mean(), 'mouth_dist')
            mouth_activity = max(0.0, min(1.0, (mouth_z - 1.0) / CONFIG['mouth_sigma_threshold']))
            ear_z = (self.calib_mean.get('ear', 0.3) - df['ear'].mean()) / self.calib_std.get('ear', 0.05)
            is_stressed_eyes = 1.0 if ear_z > CONFIG['blink_sigma_threshold'] else 0.0

            # --- 2. STATE LOGIC ---
            focus_score = engagement_score * ((stability * 0.7) + 0.3)
            stimming_score = (mouth_activity * 0.8) + (stability * 0.2)
            stress_score = (1.0 - stability) * 0.5 + (is_stressed_eyes * 0.3) + ((1.0 - engagement_score) * 0.2)

            recent_keys = df['keys'].tail(30).sum()
            recent_mouse = df['mouse'].tail(30).sum()
            is_working = (recent_keys > 0 or recent_mouse > 0)

            if is_working:
                focus_score = min(1.0, focus_score + 0.3)
                stimming_score *= 0.5
                stress_score *= 0.5
            if engagement_score < 0.5:
                focus_score *= 0.2 
                stress_score += 0.2

            # --- 3. PHONE CHECKER LOGIC (OR GATE) ---
            last_frame = self.buffer[-1]
            curr_pitch_z = self.get_z_score(last_frame[5], 'pitch')
            curr_yaw_z = self.get_z_score(last_frame[6], 'yaw')
            curr_ear_drop_z = (self.calib_mean['ear'] - last_frame[1]) / self.calib_std['ear']
            
            # Condition A: Looking Down
            looking_down = (curr_pitch_z > PHONE_PITCH_THRESHOLD) and (curr_ear_drop_z > PHONE_EAR_THRESHOLD)
            
            # Condition B: Looking Side (Significantly)
            looking_side = (curr_yaw_z > PHONE_YAW_THRESHOLD)

            # Counter Logic A (Down)
            if looking_down and not is_working:
                self.phone_down_counter += 1
            else:
                self.phone_down_counter = 0 # Instant reset
            
            # Counter Logic B (Side)
            if looking_side and not is_working:
                self.phone_side_counter += 1
            else:
                self.phone_side_counter = 0 # Instant reset

            # The OR Gate
            is_phone_checking = (self.phone_down_counter > TRIGGER_TICKS_DOWN) or \
                                (self.phone_side_counter > TRIGGER_TICKS_SIDE)

            # --- 4. OUTPUT ---
            result = {
                "timestamp": time.time(),
                "metrics": {
                    "engagement_score": round(engagement_score, 2),
                    "stability_score": round(stability, 2),
                    "mouth_z": round(mouth_z, 2)
                },
                "state_probabilities": {
                    "focus": round(min(1.0, max(0.0, focus_score)), 2),
                    "stimming": round(min(1.0, max(0.0, stimming_score)), 2),
                    "stress": round(min(1.0, max(0.0, stress_score)), 2)
                },
                "flags": {
                    "mouse_movement": bool(recent_mouse > 0),
                    "keyboard_stroke": bool(recent_keys > 0),
                    "phone_checking_mode": bool(is_phone_checking)
                }
            }
            return result

        except Exception:
            traceback.print_exc()
            return None

    def run(self):
        print("🧠 [ANALYZER] WAITING FOR TELEMETRY STREAM...")
        sys.stdout.flush()
        current_file = None
        while True:
            try:
                if not os.path.exists(DATA_DIR): time.sleep(1); continue
                files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("telemetry_")])
                if files:
                    current_file = os.path.join(DATA_DIR, files[-1])
                    print(f"📂 [ANALYZER] TRACKING FILE: {files[-1]}")
                    sys.stdout.flush()
                    break
                time.sleep(1)
            except: time.sleep(1)

        with open(current_file, 'r') as f:
            f.seek(0, 2)
            while True:
                self.check_feedback_updates()
                line = f.readline()
                if not line: time.sleep(0.05); continue
                try:
                    parts = line.strip().split(',')
                    if len(parts) < 9: continue
                    self.buffer.append([float(x) for x in parts])
                    if len(self.buffer) >= 30 and len(self.buffer) % 10 == 0:
                        result = self.analyze_window()
                        if result:
                            with open(OUTPUT_JSON, 'a') as jf:
                                json.dump(result, jf); jf.write('\n')
                            print(json.dumps(result)); sys.stdout.flush()
                except ValueError: continue

if __name__ == "__main__":
    NeuroAnalyzer().run()

