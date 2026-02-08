'''
import cv2
import mediapipe as mp
import time
import csv
import os
import numpy as np
import multiprocessing
from pynput import keyboard, mouse
from datetime import datetime
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.getcwd(), "data")
SHOW_VISUALS = True  # Set to False for maximum FPS
BUFFER_SECONDS = 2.0   

# ==========================================
# MODULE 1: LIVE PLOTTER (Diagnostic Tool)
# ==========================================
class LivePlotter:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.x_data = []
        self.ear_data = []
        self.gaze_x_data = []
        self.mouth_data = []

        # Setup the Figure
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(6, 8))
        self.fig.canvas.manager.set_window_title('Neuro-Sensory Telemetry')

        # Subplot 1: EAR (Blinks)
        self.ln1, = self.ax1.plot([], [], 'y-', label='EAR (Blink)')
        self.ax1.set_ylim(0, 0.5)
        self.ax1.legend(loc="upper right")

        # Subplot 2: Gaze Position
        self.ln2, = self.ax2.plot([], [], 'c-', label='Gaze X')
        self.ax2.set_ylim(0, 1.0)
        self.ax2.legend(loc="upper right")

        # Subplot 3: Mouth Fidgeting
        self.ln3, = self.ax3.plot([], [], 'm-', label='Mouth Dist')
        self.ax3.set_ylim(0, 0.2)
        self.ax3.legend(loc="upper right")

        plt.tight_layout()

    def update_and_draw(self, rel_time, ear, gaze_x, mouth_dist):
        """Updates data buffers and refreshes the plot."""
        self.x_data.append(rel_time)
        self.ear_data.append(ear)
        self.gaze_x_data.append(gaze_x)
        self.mouth_data.append(mouth_dist)

        # Keep rolling window fixed size
        if len(self.x_data) > self.window_size:
            self.x_data.pop(0)
            self.ear_data.pop(0)
            self.gaze_x_data.pop(0)
            self.mouth_data.pop(0)

        # Update lines
        self.ln1.set_data(self.x_data, self.ear_data)
        self.ax1.set_xlim(self.x_data[0], self.x_data[-1])
        
        self.ln2.set_data(self.x_data, self.gaze_x_data)
        self.ax2.set_xlim(self.x_data[0], self.x_data[-1])
        
        self.ln3.set_data(self.x_data, self.mouth_data)
        self.ax3.set_xlim(self.x_data[0], self.x_data[-1])
        
        plt.pause(0.001)

# ==========================================
# MODULE 2: INPUT TRACKER (Separate Process)
# ==========================================
def input_worker(q):
    """Runs in a dedicated CPU core."""
    state = {'keys': 0, 'mouse_px': 0.0, 'last_pos': None}
    
    def on_press(key): state['keys'] += 1
    def on_move(x, y):
        if state['last_pos']:
            dx, dy = x - state['last_pos'][0], y - state['last_pos'][1]
            state['mouse_px'] += (dx**2 + dy**2)**0.5
        state['last_pos'] = (x, y)

    kb = keyboard.Listener(on_press=on_press)
    m = mouse.Listener(on_move=on_move)
    kb.start()
    m.start()

    while True:
        time.sleep(0.01) # 100Hz updates
        if state['keys'] > 0 or state['mouse_px'] > 0:
            q.put({'keys': state['keys'], 'mouse_px': int(state['mouse_px'])})
            state['keys'], state['mouse_px'] = 0, 0.0

# ==========================================
# MODULE 3: BIOMETRIC UTILS
# ==========================================
class BiometricUtils:
    L_EYE = [33, 160, 158, 133, 153, 144]
    R_EYE = [362, 385, 387, 263, 373, 380]
    IRIS_L_IDX = 473
    IRIS_R_IDX = 468
    MOUTH_UPPER_IDX = 13
    MOUTH_LOWER_IDX = 14

    @staticmethod
    def calculate_ear(landmarks):
        def get_single_ear(indices):
            v1 = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) - 
                                np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
            v2 = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) - 
                                np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
            h = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) - 
                               np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
            return (v1 + v2) / (2.0 * h)
        return (get_single_ear(BiometricUtils.L_EYE) + get_single_ear(BiometricUtils.R_EYE)) / 2.0
    
    @staticmethod
    def get_gaze_coords(landmarks):
        l_pt = landmarks[BiometricUtils.IRIS_L_IDX]
        r_pt = landmarks[BiometricUtils.IRIS_R_IDX]
        avg_x = (l_pt.x + r_pt.x) / 2.0
        avg_y = (l_pt.y + r_pt.y) / 2.0
        return avg_x, avg_y

    # UPDATED: Outer Lip Landmarks for Closed-Mouth Fidgeting
    # 0 = Top of the top lip (Center)
    # 17 = Bottom of the bottom lip (Center)
    MOUTH_TOP_OUTER = 0
    MOUTH_BOTTOM_OUTER = 17

    @staticmethod
    def get_mouth_dist(landmarks):
        """Calculates vertical distance of the entire mouth structure (outer lips)."""
        upper = landmarks[BiometricUtils.MOUTH_TOP_OUTER]
        lower = landmarks[BiometricUtils.MOUTH_BOTTOM_OUTER]
        
        # We calculate the vertical distance (y-axis)
        # This will now change even if the mouth is closed!
        return np.abs(upper.y - lower.y)    

# ==========================================
# MAIN SCAFFOLD: THE PRODUCER
# ==========================================
class NeuroWatcher:
    def __init__(self):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(DATA_DIR, f"telemetry_{timestamp_str}.csv")
        self.start_time = time.time()
        
        self.cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        self.input_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=input_worker, args=(self.input_queue,))
        self.process.daemon = True
        self.process.start()

        self.plotter = LivePlotter(window_size=100)

        self.buffer = []
        self.last_flush = time.time()
        self._init_csv()

    def _init_csv(self):
        # FIXED: Added missing comma between mouth_dist and keys
        with open(self.csv_path, mode='w', newline='') as f:
            csv.writer(f).writerow(['rel_time', 'ear', 'gaze_x', 'gaze_y', 'mouth_dist', 'keys', 'mouse_px'])

    def run(self):
        print(f"🚀 Watcher Active. Data: {self.csv_path}")
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success: continue

                # FIXED: Calculate time FIRST so it is available for plotting
                rel_time = time.time() - self.start_time

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                
                # Default values
                ear = 0.0
                gaze_x, gaze_y = 0.0, 0.0 
                mouth_dist = 0.0
                
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    ear = BiometricUtils.calculate_ear(lm)
                    gaze_x, gaze_y = BiometricUtils.get_gaze_coords(lm)
                    mouth_dist = BiometricUtils.get_mouth_dist(lm)

                    # Update Live Plot
                    self.plotter.update_and_draw(rel_time, ear, gaze_x, mouth_dist)
                    
                    if SHOW_VISUALS:
                        mp.solutions.drawing_utils.draw_landmarks(
                            image=frame, landmark_list=results.multi_face_landmarks[0],
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                        )

                # Get Inputs
                total_keys = 0
                total_mouse = 0
                while not self.input_queue.empty():
                    data = self.input_queue.get()
                    if isinstance(data, dict):
                        total_keys += data.get('keys', 0)
                        total_mouse += data.get('mouse_px', 0)
                
                self.buffer.append([rel_time, ear, gaze_x, gaze_y, mouth_dist, total_keys, total_mouse])

                if time.time() - self.last_flush > BUFFER_SECONDS:
                    with open(self.csv_path, mode='a', newline='') as f:
                        csv.writer(f).writerows(self.buffer)
                    print(f"💾 Flushed {len(self.buffer)} frames. Time: {rel_time:.1f}s")
                    self.buffer = []
                    self.last_flush = time.time()

                if SHOW_VISUALS:
                    cv2.putText(frame, f"EAR: {ear:.2f} | Keys: {total_keys}", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Neuro-Watcher', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.process.terminate()

if __name__ == "__main__":
    NeuroWatcher().run()

    #Iteration 2...
import cv2
import mediapipe as mp
import time
import csv
import os
import numpy as np
import multiprocessing
from pynput import keyboard, mouse
from datetime import datetime
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.getcwd(), "data")
SHOW_VISUALS = True  
BUFFER_SECONDS = 2.0   
SMOOTHING_ALPHA = 0.2 

# ==========================================
# MODULE 1: LIVE PLOTTER (Diagnostic Tool)
# ==========================================
class LivePlotter:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.x_data = []
        self.ear_data, self.gaze_x_data = [], []
        self.mouth_data, self.pitch_data, self.yaw_data = [], [], []

        plt.style.use('dark_background')
        # FIXED: Explicitly creating 5 axes
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5) = plt.subplots(5, 1, figsize=(6, 12))
        self.fig.canvas.manager.set_window_title('Neuro-Sensory Telemetry')

        self.ln1, = self.ax1.plot([], [], 'y-', label='EAR')
        self.ax1.set_ylim(0, 0.5)
        self.ax1.legend(loc="upper right")

        self.ln2, = self.ax2.plot([], [], 'c-', label='Gaze X')
        self.ax2.set_ylim(0, 1.0)
        self.ax2.legend(loc="upper right")

        self.ln3, = self.ax3.plot([], [], 'm-', label='Mouth Dist')
        self.ax3.set_ylim(0, 0.2)
        self.ax3.legend(loc="upper right")

        self.ln4, = self.ax4.plot([], [], 'r-', label='Head Pitch')
        self.ax4.set_ylim(0.0, 1.0) 
        self.ax4.legend(loc="upper right")

        self.ln5, = self.ax5.plot([], [], 'g-', label='Head Yaw')
        self.ax5.set_ylim(-50, 50) 
        self.ax5.legend(loc="upper right")

        plt.tight_layout()

    def update_and_draw(self, rel_time, ear, gaze_x, mouth_dist, pitch, yaw):
        self.x_data.append(rel_time)
        self.ear_data.append(ear)
        self.gaze_x_data.append(gaze_x)
        self.mouth_data.append(mouth_dist)
        self.pitch_data.append(pitch)
        self.yaw_data.append(yaw)

        if len(self.x_data) > self.window_size:
            for d in [self.x_data, self.ear_data, self.gaze_x_data, self.mouth_data, self.pitch_data, self.yaw_data]:
                d.pop(0)

        self.ln1.set_data(self.x_data, self.ear_data)
        self.ln2.set_data(self.x_data, self.gaze_x_data)
        self.ln3.set_data(self.x_data, self.mouth_data)
        self.ln4.set_data(self.x_data, self.pitch_data)
        self.ln5.set_data(self.x_data, self.yaw_data)
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.set_xlim(self.x_data[0], self.x_data[-1])
        
        plt.pause(0.001)

# ==========================================
# MODULE 2: INPUT TRACKER
# ==========================================
def input_worker(q):
    state = {'keys': 0, 'mouse_px': 0.0, 'last_pos': None}
    def on_press(key): state['keys'] += 1
    def on_move(x, y):
        if state['last_pos']:
            dx, dy = x - state['last_pos'][0], y - state['last_pos'][1]
            state['mouse_px'] += (dx**2 + dy**2)**0.5
        state['last_pos'] = (x, y)

    kb = keyboard.Listener(on_press=on_press)
    m = mouse.Listener(on_move=on_move)
    kb.start(); m.start()

    while True:
        time.sleep(0.01)
        if state['keys'] > 0 or state['mouse_px'] > 0:
            q.put({'keys': state['keys'], 'mouse_px': int(state['mouse_px'])})
            state['keys'], state['mouse_px'] = 0, 0.0

# ==========================================
# MODULE 3: BIOMETRIC UTILS
# ==========================================
class BiometricUtils:
    L_EYE = [33, 160, 158, 133, 153, 144]
    R_EYE = [362, 385, 387, 263, 373, 380]
    IRIS_L_IDX = 473
    IRIS_R_IDX = 468
    MOUTH_TOP_OUTER = 0
    MOUTH_BOTTOM_OUTER = 17
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    @staticmethod
    def calculate_ear(landmarks):
        def get_single_ear(indices):
            v1 = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) - np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
            v2 = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) - np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
            h = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) - np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
            return (v1 + v2) / (2.0 * h)
        return (get_single_ear(BiometricUtils.L_EYE) + get_single_ear(BiometricUtils.R_EYE)) / 2.0
    
    @staticmethod
    def get_gaze_coords(landmarks):
        l_pt, r_pt = landmarks[BiometricUtils.IRIS_L_IDX], landmarks[BiometricUtils.IRIS_R_IDX]
        return (l_pt.x + r_pt.x) / 2.0, (l_pt.y + r_pt.y) / 2.0

    @staticmethod
    def get_mouth_dist(landmarks):
        return np.abs(landmarks[BiometricUtils.MOUTH_TOP_OUTER].y - landmarks[BiometricUtils.MOUTH_BOTTOM_OUTER].y)

    @staticmethod
    def get_head_pose(landmarks, img_w, img_h):
        nose = landmarks[BiometricUtils.NOSE_TIP]
        chin = landmarks[BiometricUtils.CHIN]
        l_eye = landmarks[BiometricUtils.LEFT_EYE_OUTER]
        r_eye = landmarks[BiometricUtils.RIGHT_EYE_OUTER]
        
        eyes_mid_y = (l_eye.y + r_eye.y) / 2.0
        face_height = np.abs(chin.y - eyes_mid_y)
        nose_to_chin = np.abs(chin.y - nose.y)
        pitch_ratio = nose_to_chin / (face_height + 1e-6)
        
        nose_x = nose.x * img_w
        l_eye_x = l_eye.x * img_w
        r_eye_x = r_eye.x * img_w
        yaw_diff = np.abs(nose_x - l_eye_x) - np.abs(nose_x - r_eye_x)
        
        return pitch_ratio, yaw_diff

# ==========================================
# MAIN SCAFFOLD: THE PRODUCER
# ==========================================
class NeuroWatcher:
    def __init__(self):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(DATA_DIR, f"telemetry_{timestamp_str}.csv")
        self.start_time = time.time()
        
        self.cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.input_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=input_worker, args=(self.input_queue,))
        self.process.daemon = True; self.process.start()
        
        self.plotter = LivePlotter(window_size=100)
        self.buffer, self.last_flush = [], time.time()
        self.prev_pitch, self.prev_yaw = 0.5, 0.0
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, mode='w', newline='') as f:
            csv.writer(f).writerow(['rel_time', 'ear', 'gaze_x', 'gaze_y', 'mouth_dist', 'pitch', 'yaw', 'keys', 'mouse_px'])

    def run(self):
        print(f"🚀 Watcher Active. Data: {self.csv_path}")
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success: continue
                rel_time = time.time() - self.start_time
                img_h, img_w, _ = frame.shape
                results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                ear, gaze_x, gaze_y, mouth_dist = 0.0, 0.0, 0.0, 0.0
                pitch, yaw = self.prev_pitch, self.prev_yaw
                
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    ear = BiometricUtils.calculate_ear(lm)
                    gaze_x, gaze_y = BiometricUtils.get_gaze_coords(lm)
                    mouth_dist = BiometricUtils.get_mouth_dist(lm)
                    raw_pitch, raw_yaw = BiometricUtils.get_head_pose(lm, img_w, img_h)
                    
                    pitch = (SMOOTHING_ALPHA * raw_pitch) + ((1 - SMOOTHING_ALPHA) * self.prev_pitch)
                    yaw = (SMOOTHING_ALPHA * raw_yaw) + ((1 - SMOOTHING_ALPHA) * self.prev_yaw)
                    self.prev_pitch, self.prev_yaw = pitch, yaw

                    self.plotter.update_and_draw(rel_time, ear, gaze_x, mouth_dist, pitch, yaw)
                    
                    if SHOW_VISUALS:
                        mp.solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=results.multi_face_landmarks[0], connections=mp.solutions.face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

                total_keys, total_mouse = 0, 0
                while not self.input_queue.empty():
                    data = self.input_queue.get()
                    if isinstance(data, dict):
                        total_keys += data.get('keys', 0); total_mouse += data.get('mouse_px', 0)
                
                self.buffer.append([rel_time, ear, gaze_x, gaze_y, mouth_dist, pitch, yaw, total_keys, total_mouse])

                if time.time() - self.last_flush > BUFFER_SECONDS:
                    with open(self.csv_path, mode='a', newline='') as f:
                        csv.writer(f).writerows(self.buffer)
                    print(f"💾 Flushed {len(self.buffer)} frames. Time: {rel_time:.1f}s")
                    self.buffer, self.last_flush = [], time.time()

                if SHOW_VISUALS:
                    cv2.putText(frame, f"P: {pitch:.2f} | Y: {yaw:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Neuro-Watcher', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.cap.release(); cv2.destroyAllWindows(); self.process.terminate()

if __name__ == "__main__":
    NeuroWatcher().run()


#Final iteration 

import cv2
import mediapipe as mp
import time
import csv
import os
import numpy as np
import multiprocessing
from pynput import keyboard, mouse
from datetime import datetime
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.getcwd(), "data")
SHOW_CAMERA_VIEW = False   # Set False to hide the webcam window
SHOW_PLOT_VIEW = False     # Set False to hide the live matplotlib graphs
BUFFER_SECONDS = 2.0      
SMOOTHING_ALPHA = 0.2     # Lower = Smoother, Higher = More Responsive

# ==========================================
# MODULE 1: LIVE PLOTTER (Diagnostic Tool)
# ==========================================
class LivePlotter:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.x_data = []
        self.ear_data, self.gaze_x_data = [], []
        self.mouth_data, self.pitch_data, self.yaw_data = [], [], []

        # Setup the Figure
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5) = plt.subplots(5, 1, figsize=(6, 12))
        self.fig.canvas.manager.set_window_title('Neuro-Sensory Telemetry')

        # Subplot 1: EAR (Blinks)
        self.ln1, = self.ax1.plot([], [], 'y-', label='EAR')
        self.ax1.set_ylim(0, 0.5)
        self.ax1.legend(loc="upper right")

        # Subplot 2: Gaze Position
        self.ln2, = self.ax2.plot([], [], 'c-', label='Gaze X')
        self.ax2.set_ylim(0, 1.0)
        self.ax2.legend(loc="upper right")

        # Subplot 3: Mouth Fidgeting
        self.ln3, = self.ax3.plot([], [], 'm-', label='Mouth Dist')
        self.ax3.set_ylim(0, 0.2)
        self.ax3.legend(loc="upper right")

        # Subplot 4: Head Pitch (Smartphone Detection)
        self.ln4, = self.ax4.plot([], [], 'r-', label='Head Pitch (Ratio)')
        self.ax4.set_ylim(0.0, 1.0) # Normalized Ratio
        self.ax4.legend(loc="upper right")

        # Subplot 5: Head Yaw (Left/Right)
        self.ln5, = self.ax5.plot([], [], 'g-', label='Head Yaw (Asymmetry)')
        self.ax5.set_ylim(-100, 100) # Expanded to -100/+100 as requested
        self.ax5.legend(loc="upper right")

        plt.tight_layout()

    def update_and_draw(self, rel_time, ear, gaze_x, mouth_dist, pitch, yaw):
        self.x_data.append(rel_time)
        self.ear_data.append(ear)
        self.gaze_x_data.append(gaze_x)
        self.mouth_data.append(mouth_dist)
        self.pitch_data.append(pitch)
        self.yaw_data.append(yaw)

        # Rolling window logic
        if len(self.x_data) > self.window_size:
            # Pop one item from EVERY list
            for d in [self.x_data, self.ear_data, self.gaze_x_data, self.mouth_data, self.pitch_data, self.yaw_data]:
                d.pop(0)

        self.ln1.set_data(self.x_data, self.ear_data)
        self.ln2.set_data(self.x_data, self.gaze_x_data)
        self.ln3.set_data(self.x_data, self.mouth_data)
        self.ln4.set_data(self.x_data, self.pitch_data)
        self.ln5.set_data(self.x_data, self.yaw_data)
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.set_xlim(self.x_data[0], self.x_data[-1])
        
        plt.pause(0.001)

# ==========================================
# MODULE 2: INPUT TRACKER
# ==========================================
def input_worker(q):
    state = {'keys': 0, 'mouse_px': 0.0, 'last_pos': None}
    def on_press(key): state['keys'] += 1
    def on_move(x, y):
        if state['last_pos']:
            dx, dy = x - state['last_pos'][0], y - state['last_pos'][1]
            state['mouse_px'] += (dx**2 + dy**2)**0.5
        state['last_pos'] = (x, y)

    kb = keyboard.Listener(on_press=on_press)
    m = mouse.Listener(on_move=on_move)
    kb.start(); m.start()

    while True:
        time.sleep(0.01)
        if state['keys'] > 0 or state['mouse_px'] > 0:
            q.put({'keys': state['keys'], 'mouse_px': int(state['mouse_px'])})
            state['keys'], state['mouse_px'] = 0, 0.0

# ==========================================
# MODULE 3: BIOMETRIC UTILS (STABILIZED 2D)
# ==========================================
class BiometricUtils:
    L_EYE = [33, 160, 158, 133, 153, 144]
    R_EYE = [362, 385, 387, 263, 373, 380]
    IRIS_L_IDX = 473
    IRIS_R_IDX = 468
    MOUTH_TOP_OUTER = 0
    MOUTH_BOTTOM_OUTER = 17
    
    # Landmarks for 2D Pitch/Yaw
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    @staticmethod
    def calculate_ear(landmarks):
        def get_single_ear(indices):
            v1 = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) - np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
            v2 = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) - np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
            h = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) - np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
            return (v1 + v2) / (2.0 * h)
        return (get_single_ear(BiometricUtils.L_EYE) + get_single_ear(BiometricUtils.R_EYE)) / 2.0
    
    @staticmethod
    def get_gaze_coords(landmarks):
        l_pt, r_pt = landmarks[BiometricUtils.IRIS_L_IDX], landmarks[BiometricUtils.IRIS_R_IDX]
        return (l_pt.x + r_pt.x) / 2.0, (l_pt.y + r_pt.y) / 2.0

    @staticmethod
    def get_mouth_dist(landmarks):
        return np.abs(landmarks[BiometricUtils.MOUTH_TOP_OUTER].y - landmarks[BiometricUtils.MOUTH_BOTTOM_OUTER].y)

    @staticmethod
    def get_head_pose(landmarks, img_w, img_h):
        """
        Calculates 2D Ratios. Returns Pitch (0-1) and Yaw (Pixel Diff).
        """
        nose = landmarks[BiometricUtils.NOSE_TIP]
        chin = landmarks[BiometricUtils.CHIN]
        l_eye = landmarks[BiometricUtils.LEFT_EYE_OUTER]
        r_eye = landmarks[BiometricUtils.RIGHT_EYE_OUTER]
        
        # Pitch: Ratio of Nose-to-Chin vs Face Height
        eyes_mid_y = (l_eye.y + r_eye.y) / 2.0
        face_height = np.abs(chin.y - eyes_mid_y)
        nose_to_chin = np.abs(chin.y - nose.y)
        pitch_ratio = nose_to_chin / (face_height + 1e-6)
        
        # Yaw: Horizontal Asymmetry (Scaled by Image Width)
        nose_x = nose.x * img_w
        l_eye_x = l_eye.x * img_w
        r_eye_x = r_eye.x * img_w
        yaw_diff = np.abs(nose_x - l_eye_x) - np.abs(nose_x - r_eye_x)
        
        return pitch_ratio, yaw_diff

# ==========================================
# MAIN SCAFFOLD: THE PRODUCER
# ==========================================
class NeuroWatcher:
    def __init__(self):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(DATA_DIR, f"telemetry_{timestamp_str}.csv")
        self.start_time = time.time()
        
        # Camera Setup
        self.cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        # MediaPipe Setup
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Multiprocessing
        self.input_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=input_worker, args=(self.input_queue,))
        self.process.daemon = True; self.process.start()
        
        # Visualization (Conditional Init)
        self.plotter = None
        if SHOW_PLOT_VIEW:
            self.plotter = LivePlotter(window_size=100)
            
        self.buffer, self.last_flush = [], time.time()
        
        # State for Smoothing
        self.prev_pitch = 0.5
        self.prev_yaw = 0.0
        
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, mode='w', newline='') as f:
            csv.writer(f).writerow(['rel_time', 'ear', 'gaze_x', 'gaze_y', 'mouth_dist', 'pitch', 'yaw', 'keys', 'mouse_px'])

    def run(self):
        print(f"🚀 Watcher Active. Data: {self.csv_path}")
        print(f"   Camera View: {SHOW_CAMERA_VIEW} | Plot View: {SHOW_PLOT_VIEW}")
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success: continue
                rel_time = time.time() - self.start_time
                img_h, img_w, _ = frame.shape
                
                # Convert for MediaPipe
                results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Input Gathering (Drain queue regardless of face detection)
                total_keys, total_mouse = 0, 0
                while not self.input_queue.empty():
                    data = self.input_queue.get()
                    if isinstance(data, dict):
                        total_keys += data.get('keys', 0); total_mouse += data.get('mouse_px', 0)
                
                # Only process and log if a face is detected
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    ear = BiometricUtils.calculate_ear(lm)
                    gaze_x, gaze_y = BiometricUtils.get_gaze_coords(lm)
                    mouth_dist = BiometricUtils.get_mouth_dist(lm)
                    
                    # Calculate 2D Pose
                    raw_pitch, raw_yaw = BiometricUtils.get_head_pose(lm, img_w, img_h)
                    
                    # Apply Smoothing (EMA)
                    pitch = (SMOOTHING_ALPHA * raw_pitch) + ((1 - SMOOTHING_ALPHA) * self.prev_pitch)
                    yaw = (SMOOTHING_ALPHA * raw_yaw) + ((1 - SMOOTHING_ALPHA) * self.prev_yaw)
                    
                    # Update State
                    self.prev_pitch, self.prev_yaw = pitch, yaw

                    # Update Plotter (Only if enabled)
                    if SHOW_PLOT_VIEW and self.plotter:
                        self.plotter.update_and_draw(rel_time, ear, gaze_x, mouth_dist, pitch, yaw)
                    
                    # Log Data (MOVED INSIDE IF BLOCK - "Zero-Drop" Fix)
                    self.buffer.append([rel_time, ear, gaze_x, gaze_y, mouth_dist, pitch, yaw, total_keys, total_mouse])

                    # Camera Visualization (Only if enabled)
                    if SHOW_CAMERA_VIEW:
                        mp.solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=results.multi_face_landmarks[0], connections=mp.solutions.face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                        cv2.putText(frame, f"P: {pitch:.2f} | Y: {yaw:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('Neuro-Watcher', frame)
                
                # Flush to Disk Logic
                if time.time() - self.last_flush > BUFFER_SECONDS:
                    if self.buffer: # Only write if there is data
                        with open(self.csv_path, mode='a', newline='') as f:
                            csv.writer(f).writerows(self.buffer)
                        print(f"💾 Flushed {len(self.buffer)} frames. Time: {rel_time:.1f}s")
                        self.buffer = []
                    self.last_flush = time.time()

                if SHOW_CAMERA_VIEW:
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    
        finally:
            self.cap.release(); cv2.destroyAllWindows(); self.process.terminate()

if __name__ == "__main__":
    NeuroWatcher().run()






import cv2
import mediapipe as mp
import time
import csv
import os
import numpy as np
import multiprocessing
from pynput import keyboard, mouse
from datetime import datetime
import matplotlib.pyplot as plt
from utils import BiometricUtils

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.getcwd(), "data")
SHOW_CAMERA_VIEW = True   # Set False to hide the webcam window
SHOW_PLOT_VIEW = True     # Set False to hide the live matplotlib graphs
BUFFER_SECONDS = 2.0      
SMOOTHING_ALPHA = 0.2     # Lower = Smoother, Higher = More Responsive

# ==========================================
# MODULE 1: LIVE PLOTTER (Diagnostic Tool)
# ==========================================
class LivePlotter:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.x_data = []
        self.ear_data, self.gaze_x_data = [], []
        self.mouth_data, self.pitch_data, self.yaw_data = [], [], []

        # Setup the Figure
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5) = plt.subplots(5, 1, figsize=(6, 12))
        self.fig.canvas.manager.set_window_title('Neuro-Sensory Telemetry')

        # Subplot 1: EAR (Blinks)
        self.ln1, = self.ax1.plot([], [], 'y-', label='EAR')
        self.ax1.set_ylim(0, 0.5)
        self.ax1.legend(loc="upper right")

        # Subplot 2: Gaze Position
        self.ln2, = self.ax2.plot([], [], 'c-', label='Gaze X')
        self.ax2.set_ylim(0, 1.0)
        self.ax2.legend(loc="upper right")

        # Subplot 3: Mouth Fidgeting
        self.ln3, = self.ax3.plot([], [], 'm-', label='Mouth Dist')
        self.ax3.set_ylim(0, 0.2)
        self.ax3.legend(loc="upper right")

        # Subplot 4: Head Pitch (Smartphone Detection)
        self.ln4, = self.ax4.plot([], [], 'r-', label='Head Pitch (Ratio)')
        self.ax4.set_ylim(0.0, 1.0) # Normalized Ratio
        self.ax4.legend(loc="upper right")

        # Subplot 5: Head Yaw (Left/Right)
        self.ln5, = self.ax5.plot([], [], 'g-', label='Head Yaw (Asymmetry)')
        self.ax5.set_ylim(-100, 100) # Expanded to -100/+100 as requested
        self.ax5.legend(loc="upper right")

        plt.tight_layout()

    def update_and_draw(self, rel_time, ear, gaze_x, mouth_dist, pitch, yaw):
        self.x_data.append(rel_time)
        self.ear_data.append(ear)
        self.gaze_x_data.append(gaze_x)
        self.mouth_data.append(mouth_dist)
        self.pitch_data.append(pitch)
        self.yaw_data.append(yaw)

        # Rolling window logic
        if len(self.x_data) > self.window_size:
            # Pop one item from EVERY list
            for d in [self.x_data, self.ear_data, self.gaze_x_data, self.mouth_data, self.pitch_data, self.yaw_data]:
                d.pop(0)

        self.ln1.set_data(self.x_data, self.ear_data)
        self.ln2.set_data(self.x_data, self.gaze_x_data)
        self.ln3.set_data(self.x_data, self.mouth_data)
        self.ln4.set_data(self.x_data, self.pitch_data)
        self.ln5.set_data(self.x_data, self.yaw_data)
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.set_xlim(self.x_data[0], self.x_data[-1])
        
        plt.pause(0.001)

# ==========================================
# MODULE 2: INPUT TRACKER
# ==========================================
def input_worker(q):
    state = {'keys': 0, 'mouse_px': 0.0, 'last_pos': None}
    def on_press(key): state['keys'] += 1
    def on_move(x, y):
        if state['last_pos']:
            dx, dy = x - state['last_pos'][0], y - state['last_pos'][1]
            state['mouse_px'] += (dx**2 + dy**2)**0.5
        state['last_pos'] = (x, y)

    kb = keyboard.Listener(on_press=on_press)
    m = mouse.Listener(on_move=on_move)
    kb.start(); m.start()

    while True:
        time.sleep(0.01)
        if state['keys'] > 0 or state['mouse_px'] > 0:
            q.put({'keys': state['keys'], 'mouse_px': int(state['mouse_px'])})
            state['keys'], state['mouse_px'] = 0, 0.0


# ==========================================
# MAIN SCAFFOLD: THE PRODUCER
# ==========================================
class NeuroWatcher:
    def __init__(self):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(DATA_DIR, f"telemetry_{timestamp_str}.csv")
        self.start_time = time.time()
        
        # Camera Setup
        self.cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        # MediaPipe Setup
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Multiprocessing
        self.input_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=input_worker, args=(self.input_queue,))
        self.process.daemon = True; self.process.start()
        
        # Visualization (Conditional Init)
        self.plotter = None
        if SHOW_PLOT_VIEW:
            self.plotter = LivePlotter(window_size=100)
            
        self.buffer, self.last_flush = [], time.time()
        
        # State for Smoothing
        self.prev_pitch = 0.5
        self.prev_yaw = 0.0
        
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, mode='w', newline='') as f:
            csv.writer(f).writerow(['rel_time', 'ear', 'gaze_x', 'gaze_y', 'mouth_dist', 'pitch', 'yaw', 'keys', 'mouse_px'])

    def run(self):
        print(f"🚀 Watcher Active. Data: {self.csv_path}")
        print(f"   Camera View: {SHOW_CAMERA_VIEW} | Plot View: {SHOW_PLOT_VIEW}")
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success: continue
                rel_time = time.time() - self.start_time
                img_h, img_w, _ = frame.shape
                
                # Convert for MediaPipe
                results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Input Gathering (Drain queue regardless of face detection)
                total_keys, total_mouse = 0, 0
                while not self.input_queue.empty():
                    data = self.input_queue.get()
                    if isinstance(data, dict):
                        total_keys += data.get('keys', 0); total_mouse += data.get('mouse_px', 0)
                
                # Only process and log if a face is detected
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    ear = BiometricUtils.calculate_ear(lm)
                    gaze_x, gaze_y = BiometricUtils.get_gaze_coords(lm)
                    mouth_dist = BiometricUtils.get_mouth_dist(lm)
                    
                    # Calculate 2D Pose
                    raw_pitch, raw_yaw = BiometricUtils.get_head_pose(lm, img_w, img_h)
                    
                    # Apply Smoothing (EMA)
                    pitch = (SMOOTHING_ALPHA * raw_pitch) + ((1 - SMOOTHING_ALPHA) * self.prev_pitch)
                    yaw = (SMOOTHING_ALPHA * raw_yaw) + ((1 - SMOOTHING_ALPHA) * self.prev_yaw)
                    
                    # Update State
                    self.prev_pitch, self.prev_yaw = pitch, yaw

                    # Update Plotter (Only if enabled)
                    if SHOW_PLOT_VIEW and self.plotter:
                        self.plotter.update_and_draw(rel_time, ear, gaze_x, mouth_dist, pitch, yaw)
                    
                    # Log Data (MOVED INSIDE IF BLOCK - "Zero-Drop" Fix)
                    self.buffer.append([rel_time, ear, gaze_x, gaze_y, mouth_dist, pitch, yaw, total_keys, total_mouse])

                    # Camera Visualization (Only if enabled)
                    if SHOW_CAMERA_VIEW:
                        mp.solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=results.multi_face_landmarks[0], connections=mp.solutions.face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                        cv2.putText(frame, f"P: {pitch:.2f} | Y: {yaw:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('Neuro-Watcher', frame)
                
                # Flush to Disk Logic
                if time.time() - self.last_flush > BUFFER_SECONDS:
                    if self.buffer: # Only write if there is data
                        with open(self.csv_path, mode='a', newline='') as f:
                            csv.writer(f).writerows(self.buffer)
                        print(f"💾 Flushed {len(self.buffer)} frames. Time: {rel_time:.1f}s")
                        self.buffer = []
                    self.last_flush = time.time()

                if SHOW_CAMERA_VIEW:
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    
        finally:
            self.cap.release(); cv2.destroyAllWindows(); self.process.terminate()

if __name__ == "__main__":
    NeuroWatcher().run()
'''
    

import cv2
import mediapipe as mp
import time
import csv
import os
import numpy as np
import multiprocessing
from pynput import keyboard, mouse
from datetime import datetime
import matplotlib.pyplot as plt
from utils import BiometricUtils

# --- CONFIGURATION: ABSOLUTE PATH ANCHORING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

SHOW_CAMERA_VIEW = True
SHOW_PLOT_VIEW = True
BUFFER_SECONDS = 0.5 
SMOOTHING_ALPHA = 0.2

# ==========================================
# MODULE 1: LIVE PLOTTER
# ==========================================
class LivePlotter:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.x_data = []
        self.ear_data, self.gaze_x_data = [], []
        self.mouth_data, self.pitch_data, self.yaw_data = [], [], []

        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5) = plt.subplots(5, 1, figsize=(6, 12))
        self.fig.canvas.manager.set_window_title('Neuro-Sensory Telemetry')

        self.ln1, = self.ax1.plot([], [], 'y-', label='EAR')
        self.ax1.set_ylim(0, 0.5); self.ax1.legend(loc="upper right")

        self.ln2, = self.ax2.plot([], [], 'c-', label='Gaze X')
        self.ax2.set_ylim(0, 1.0); self.ax2.legend(loc="upper right")

        self.ln3, = self.ax3.plot([], [], 'm-', label='Mouth Dist')
        self.ax3.set_ylim(0, 0.2); self.ax3.legend(loc="upper right")

        self.ln4, = self.ax4.plot([], [], 'r-', label='Head Pitch')
        self.ax4.set_ylim(0.0, 1.0); self.ax4.legend(loc="upper right")

        self.ln5, = self.ax5.plot([], [], 'g-', label='Head Yaw')
        self.ax5.set_ylim(-100, 100); self.ax5.legend(loc="upper right")

        plt.tight_layout()

    def update_and_draw(self, rel_time, ear, gaze_x, mouth_dist, pitch, yaw):
        self.x_data.append(rel_time)
        self.ear_data.append(ear)
        self.gaze_x_data.append(gaze_x)
        self.mouth_data.append(mouth_dist)
        self.pitch_data.append(pitch)
        self.yaw_data.append(yaw)

        if len(self.x_data) > self.window_size:
            for d in [self.x_data, self.ear_data, self.gaze_x_data, self.mouth_data, self.pitch_data, self.yaw_data]:
                d.pop(0)

        self.ln1.set_data(self.x_data, self.ear_data)
        self.ln2.set_data(self.x_data, self.gaze_x_data)
        self.ln3.set_data(self.x_data, self.mouth_data)
        self.ln4.set_data(self.x_data, self.pitch_data)
        self.ln5.set_data(self.x_data, self.yaw_data)
        
        if len(self.x_data) > 1:
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
                ax.set_xlim(self.x_data[0], self.x_data[-1])
        
        plt.pause(0.001)

# ==========================================
# MODULE 2: INPUT TRACKER
# ==========================================
def input_worker(q):
    state = {'keys': 0, 'mouse_px': 0.0, 'last_pos': None}
    def on_press(key): state['keys'] += 1
    def on_move(x, y):
        if state['last_pos']:
            dx, dy = x - state['last_pos'][0], y - state['last_pos'][1]
            state['mouse_px'] += (dx**2 + dy**2)**0.5
        state['last_pos'] = (x, y)

    try:
        kb = keyboard.Listener(on_press=on_press)
        m = mouse.Listener(on_move=on_move)
        kb.start(); m.start()
        
        while True:
            time.sleep(0.01)
            if state['keys'] > 0 or state['mouse_px'] > 0:
                q.put({'keys': state['keys'], 'mouse_px': int(state['mouse_px'])})
                state['keys'], state['mouse_px'] = 0, 0.0
    except Exception:
        pass 

# ==========================================
# MODULE 3: THE WATCHER (PRODUCER)
# ==========================================
class NeuroWatcher:
    def __init__(self):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(DATA_DIR, f"telemetry_{timestamp_str}.csv")
        self.start_time = time.time()
        
        self.cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.input_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=input_worker, args=(self.input_queue,))
        self.process.daemon = True; self.process.start()
        
        self.plotter = None
        if SHOW_PLOT_VIEW:
            self.plotter = LivePlotter(window_size=100)
            
        self.buffer, self.last_flush = [], time.time()
        self.prev_pitch = 0.5
        self.prev_yaw = 0.0
        
        self._init_csv()

    def _init_csv(self):
        print(f"📝 [WATCHER] Creating File: {self.csv_path}")
        with open(self.csv_path, mode='w', newline='') as f:
            csv.writer(f).writerow(['rel_time', 'ear', 'gaze_x', 'gaze_y', 'mouth_dist', 'pitch', 'yaw', 'keys', 'mouse_px'])
            f.flush()
            os.fsync(f.fileno())

    def run(self):
        print(f"🚀 [WATCHER] CAMERA ACTIVE. Writing to: {self.csv_path}")
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success: continue
                
                rel_time = time.time() - self.start_time
                results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Get Input Data
                total_keys, total_mouse = 0, 0
                while not self.input_queue.empty():
                    data = self.input_queue.get()
                    if isinstance(data, dict):
                        total_keys += data.get('keys', 0); total_mouse += data.get('mouse_px', 0)
                
                # Process Face
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    ear = BiometricUtils.calculate_ear(lm)
                    gaze_x, gaze_y = BiometricUtils.get_gaze_coords(lm)
                    mouth_dist = BiometricUtils.get_mouth_dist(lm)
                    raw_pitch, raw_yaw = BiometricUtils.get_head_pose(lm, frame.shape[1], frame.shape[0])
                    
                    # Smoothing
                    pitch = (SMOOTHING_ALPHA * raw_pitch) + ((1 - SMOOTHING_ALPHA) * self.prev_pitch)
                    yaw = (SMOOTHING_ALPHA * raw_yaw) + ((1 - SMOOTHING_ALPHA) * self.prev_yaw)
                    self.prev_pitch, self.prev_yaw = pitch, yaw

                    if self.plotter: 
                        self.plotter.update_and_draw(rel_time, ear, gaze_x, mouth_dist, pitch, yaw)
                    
                    self.buffer.append([rel_time, ear, gaze_x, gaze_y, mouth_dist, pitch, yaw, total_keys, total_mouse])

                    if SHOW_CAMERA_VIEW:
                        mp.solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=results.multi_face_landmarks[0], connections=mp.solutions.face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                        cv2.imshow('Neuro-Watcher', frame)
                
                # Flush to Disk
                if time.time() - self.last_flush > BUFFER_SECONDS:
                    if self.buffer:
                        with open(self.csv_path, mode='a', newline='') as f:
                            csv.writer(f).writerows(self.buffer)
                            f.flush()
                            os.fsync(f.fileno())
                        self.buffer = []
                    self.last_flush = time.time()

                if SHOW_CAMERA_VIEW:
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    
        finally:
            self.cap.release(); cv2.destroyAllWindows(); self.process.terminate()

if __name__ == "__main__":
    NeuroWatcher().run()