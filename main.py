import tkinter as tk
from tkinter import ttk, font
import threading
import time
import queue
import RPi.GPIO as GPIO
from hardware.ultrasonic import UltrasonicSensor
import cv2
from PIL import Image, ImageTk
import torch
import sounddevice as sd
import torchaudio.transforms as T
import numpy as np
from software.face_recognition import FaceRecognition

# Global variables for system state
system_state = {
    "current_mode": "IDLE",
    "person_detected": False,
    "face_recognized": False,
    "registered_users": {},
    "current_user": "",
    "reset_timer": None,
    "lock_timer": None
}

# Communication queue between threads
event_queue = queue.Queue()

# Hardware constants
SOLENOID_PIN = 14
TRIGGER_PIN = 17
ECHO_PIN = 27
PRESENCE_THRESHOLD = 100  # 1 meter in cm

# Audio constants
SAMPLE_RATE = 16000
DURATION = 1.0
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
THRESHOLD = 0.78

class DoorLockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Door Lock System")
        self.root.geometry("800x480")
        self.root.attributes('-fullscreen', True)
        
        # Initialize components
        self.setup_styles()
        self.create_ui()
        self.init_hardware()
        self.init_models()
        
        # Start system threads
        self.start_threads()

    def setup_styles(self):
        """Configure UI styles"""
        style = ttk.Style()
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=12)
        
        style.configure("Main.TFrame", background="#f0f0f0")
        style.configure("Status.TLabel", font=("Helvetica", 24, "bold"), background="#f0f0f0")
        style.configure("Info.TLabel", font=("Helvetica", 18), background="#f0f0f0")

    def create_ui(self):
        """Create main UI components"""
        self.main_frame = ttk.Frame(self.root, style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_label = ttk.Label(self.main_frame, text="SYSTEM IDLE", style="Status.TLabel")
        self.status_label.pack(pady=(50, 20))
        
        self.info_label = ttk.Label(self.main_frame, text="Waiting for presence detection...", style="Info.TLabel")
        self.info_label.pack(pady=20)
        
        # Camera feed label
        self.camera_label = ttk.Label(self.main_frame)
        self.camera_label.pack()

    def init_hardware(self):
        """Initialize hardware components"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SOLENOID_PIN, GPIO.OUT)
        GPIO.output(SOLENOID_PIN, GPIO.LOW)
        
        self.ultrasonic = UltrasonicSensor(TRIGGER_PIN, ECHO_PIN)
        self.face_recognition = FaceRecognition()
        self.cap = None

    def init_models(self):
        """Initialize AI models"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wake_word_model = self.load_wake_word_model()
        self.audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=64, 
            n_fft=400, hop_length=160
        ).to(self.device)

    def load_wake_word_model(self):
        """Load wake word detection model"""
        model = torch.jit.load("476998.pth", map_location=self.device)
        model.eval()
        return model

    def start_threads(self):
        """Start system threads"""
        threading.Thread(target=self.monitor_presence, daemon=True).start()
        threading.Thread(target=self.process_events, daemon=True).start()

    def monitor_presence(self):
        """Ultrasonic presence detection thread"""
        while True:
            if system_state["current_mode"] == "IDLE":
                distance = self.ultrasonic.get_distance()
                if distance <= PRESENCE_THRESHOLD:
                    event_queue.put(("PRESENCE_DETECTED", distance))
                    time.sleep(1)  # Prevent multiple triggers
                time.sleep(0.1)

    def process_events(self):
        """Process system events from queue"""
        while True:
            try:
                event, data = event_queue.get(timeout=0.1)
                self.root.after(0, self.handle_event, event, data)
            except queue.Empty:
                continue

    def handle_event(self, event, data):
        """Handle different system events"""
        handlers = {
            "PRESENCE_DETECTED": self.handle_presence,
            "WAKE_WORD_DETECTED": self.handle_wake_word,
            "FACE_RECOGNIZED": self.handle_face_recognized,
            "FACE_NOT_RECOGNIZED": self.handle_face_not_recognized
        }
        handlers.get(event, lambda x: None)(data)

    def handle_presence(self, distance):
        """Handle presence detection"""
        if system_state["current_mode"] == "IDLE":
            system_state["current_mode"] = "WAKEUP_LISTENING"
            self.status_label.config(text="PRESENCE DETECTED")
            self.info_label.config(text="Say wake word to activate")
            self.start_audio_stream()

            # Start timeout timer
            system_state["reset_timer"] = self.root.after(180000, self.timeout_reset)

    def start_audio_stream(self):
        """Start listening for wake word"""
        self.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback
        )
        self.audio_stream.start()

    def audio_callback(self, indata, frames, time, status):
        """Audio processing callback"""
        self.audio_buffer[:-frames] = self.audio_buffer[frames:]
        self.audio_buffer[-frames:] = indata[:, 0]
        
        # Wake word detection
        audio_tensor = torch.from_numpy(self.audio_buffer).to(self.device)
        mel = self.transform(audio_tensor)
        prediction = torch.sigmoid(self.wake_word_model(mel.unsqueeze(0)))
        
        if prediction.item() > THRESHOLD:
            self.audio_stream.stop()
            event_queue.put(("WAKE_WORD_DETECTED", None))

    def handle_wake_word(self, _):
        """Handle wake word detection"""
        system_state["current_mode"] = "FACE_RECOGNITION"
        self.status_label.config(text="FACE RECOGNITION")
        self.info_label.config(text="Looking at camera...")
        self.start_camera()
        self.root.after(3000, self.perform_face_recognition)

    def start_camera(self):
        """Initialize camera feed"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.update_camera_feed()

    def update_camera_feed(self):
        """Update live camera preview"""
        if self.cap and system_state["current_mode"] == "FACE_RECOGNITION":
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)
            self.root.after(10, self.update_camera_feed)

    def perform_face_recognition(self):
        """Perform face recognition check"""
        ret, frame = self.cap.read()
        if ret:
            frame, recognized = self.face_recognition.recognize_face(frame)
            if recognized:
                event_queue.put(("FACE_RECOGNIZED", None))
            else:
                event_queue.put(("FACE_NOT_RECOGNIZED", None))
        self.cleanup_camera()

    def handle_face_recognized(self, _):
        """Handle successful face recognition"""
        system_state["current_mode"] = "UNLOCKED"
        self.status_label.config(text="ACCESS GRANTED")
        self.info_label.config(text="Door unlocked for 10 seconds")
        GPIO.output(SOLENOID_PIN, GPIO.HIGH)
        system_state["lock_timer"] = self.root.after(10000, self.lock_door)

    def handle_face_not_recognized(self, _):
        """Handle failed face recognition"""
        self.status_label.config(text="ACCESS DENIED")
        self.info_label.config(text="Face not recognized")
        self.root.after(5000, self.reset_to_idle)

    def lock_door(self):
        """Lock the door and reset system"""
        GPIO.output(SOLENOID_PIN, GPIO.LOW)
        self.reset_to_idle()

    def timeout_reset(self):
        """Reset system after inactivity"""
        if system_state["current_mode"] == "WAKEUP_LISTENING":
            self.info_label.config(text="No wake word detected")
            self.root.after(2000, self.reset_to_idle)

    def reset_to_idle(self):
        """Reset system to initial state"""
        system_state["current_mode"] = "IDLE"
        self.status_label.config(text="SYSTEM IDLE")
        self.info_label.config(text="Waiting for presence detection...")
        self.cleanup_camera()
        
        if self.audio_stream:
            self.audio_stream.stop()

    def cleanup_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        self.camera_label.config(image='')

if __name__ == "__main__":
    root = tk.Tk()
    app = DoorLockApp(root)
    root.mainloop()
    GPIO.cleanup()