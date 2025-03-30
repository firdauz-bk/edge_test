import tkinter as tk
from tkinter import ttk, font
import threading
import time
import queue
import RPi.GPIO as GPIO
import cv2
from PIL import Image, ImageTk
import torch
import sounddevice as sd
import torchaudio.transforms as T
import numpy as np
import os

from software.face_recognition import FaceRecognition
from hardware.door_lock import DoorLock
from hardware.ultrasonic import UltrasonicSensor

# Global variables for system state
system_state = {
    "current_mode": "IDLE",  # IDLE, WAKEUP_LISTENING, FACE_RECOGNITION, UNLOCKED
    "person_detected": False,
    "face_recognized": False,
    "registered_users": {},  # Will store face encodings with names
    "current_user": "",
    "reset_timer": None,
    "lock_timer": None
}

# Communication queue between threads
event_queue = queue.Queue()

# GPIO pins for door lock solenoid
SOLENOID_PIN = 14
TRIGGER_PIN = 17
ECHO_PIN = 27

# Constants
PRESENCE_THRESHOLD = 100  # 1 meter in cm
WAKEUP_TIMEOUT = 180  # 3 minutes in seconds
DOOR_UNLOCK_TIME = 10  # 10 seconds

# Audio constants
SAMPLE_RATE = 16000
DURATION = 1.0
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
THRESHOLD = 0.78

class DoorLockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Door Lock System")
        self.root.geometry("800x480")  # Common 7" display resolution for RPi
        self.root.attributes('-fullscreen', True)  # Set fullscreen for embedded display
        
        # Set up styles
        self.setup_styles()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = ttk.Label(
            self.main_frame, 
            text="SYSTEM IDLE", 
            style="Status.TLabel",
            anchor="center"
        )
        self.status_label.pack(pady=(50, 20))
        
        # Info display
        self.info_label = ttk.Label(
            self.main_frame, 
            text="Waiting for presence detection...", 
            style="Info.TLabel",
            anchor="center"
        )
        self.info_label.pack(pady=20)
        
        # Camera feed label
        self.camera_label = ttk.Label(self.main_frame)
        self.camera_label.pack(pady=10)
        
        # Welcome message (initially hidden)
        self.welcome_label = ttk.Label(
            self.main_frame, 
            text="", 
            style="Welcome.TLabel",
            anchor="center"
        )
        self.welcome_label.pack(pady=20)
        self.welcome_label.pack_forget()  # Hide initially
        
        # Countdown label (initially hidden)
        self.countdown_label = ttk.Label(
            self.main_frame, 
            text="", 
            style="Countdown.TLabel",
            anchor="center"
        )
        self.countdown_label.pack(pady=20)
        self.countdown_label.pack_forget()  # Hide initially
        
        # Button frame for admin functions
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.BOTTOM, pady=40)
        
        # Admin button to register faces
        self.register_button = ttk.Button(
            self.button_frame,
            text="Register New Face",
            command=self.start_registration,
            style="Button.TButton"
        )
        self.register_button.pack(side=tk.LEFT, padx=20)
        
        # Exit button (for development)
        self.exit_button = ttk.Button(
            self.button_frame,
            text="Exit",
            command=self.exit_application,
            style="Button.TButton"
        )
        self.exit_button.pack(side=tk.RIGHT, padx=20)
        
        # Initialize hardware components
        self.init_hardware()
        
        # Initialize AI models
        self.init_models()
        
        # Start system threads
        self.start_threads()
        
    def setup_styles(self):
        """Set up styles for the UI elements"""
        style = ttk.Style()
        
        # Define custom fonts
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=12)
        
        title_font = font.Font(family="Helvetica", size=24, weight="bold")
        info_font = font.Font(family="Helvetica", size=18)
        welcome_font = font.Font(family="Helvetica", size=36, weight="bold")
        countdown_font = font.Font(family="Helvetica", size=72, weight="bold")
        button_font = font.Font(family="Helvetica", size=14, weight="bold")
        
        # Configure styles with fonts
        style.configure("Main.TFrame", background="#f0f0f0")
        style.configure("Status.TLabel", font=title_font, background="#f0f0f0", foreground="#333333")
        style.configure("Info.TLabel", font=info_font, background="#f0f0f0", foreground="#555555")
        style.configure("Welcome.TLabel", font=welcome_font, background="#f0f0f0", foreground="#009900")
        style.configure("Countdown.TLabel", font=countdown_font, background="#f0f0f0", foreground="#ff0000")
        style.configure("Button.TButton", font=button_font)
        
    def init_hardware(self):
        """Initialize hardware components"""
        # Set up GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SOLENOID_PIN, GPIO.OUT)
        GPIO.output(SOLENOID_PIN, GPIO.LOW)  # Ensure door is locked initially
        
        # Initialize ultrasonic sensor
        self.ultrasonic = UltrasonicSensor(TRIGGER_PIN, ECHO_PIN)
        
        # Initialize door lock
        self.door_lock = DoorLock(SOLENOID_PIN)
        
        # Initialize face recognition
        self.face_recognition = FaceRecognition()
        
        # Initialize camera
        self.cap = None
        
    def init_models(self):
        """Initialize AI models"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wake_word_model = self.load_wake_word_model()
        self.audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, 
            n_mels=64, 
            n_fft=400, 
            hop_length=160
        ).to(self.device)
        self.audio_stream = None
        
    def load_wake_word_model(self):
        """Load wake word detection model"""
        try:
            model = torch.jit.load("476998.pth", map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading wake word model: {e}")
            # Create a dummy model for testing if needed
            return None
        
    def start_threads(self):
        """Start system threads"""
        # Start ultrasonic sensor thread
        self.ultrasonic_thread = threading.Thread(target=self.monitor_presence, daemon=True)
        self.ultrasonic_thread.start()
        
        # Start event processing thread
        self.update_thread = threading.Thread(target=self.process_events, daemon=True)
        self.update_thread.start()
    
    def monitor_presence(self):
        """Thread function to monitor presence using ultrasonic sensor"""
        while True:
            if system_state["current_mode"] == "IDLE":
                distance = self.ultrasonic.get_distance()
                
                # Check if someone is within detection range
                if distance <= PRESENCE_THRESHOLD:
                    event_queue.put(("PRESENCE_DETECTED", distance))
                    time.sleep(1)  # Prevent multiple triggers
                
                # Small sleep to prevent CPU overuse
                time.sleep(0.1)
            else:
                # If not in IDLE mode, just wait and check periodically
                time.sleep(0.5)
    
    def process_events(self):
        """Process events from the queue and update UI accordingly"""
        while True:
            try:
                event, data = event_queue.get(block=True, timeout=0.1)
                
                # Use after() to update UI from the main thread
                self.root.after(0, self.handle_event, event, data)
                
            except queue.Empty:
                # Queue is empty, continue the loop
                continue
                
    def handle_event(self, event, data):
        """Handle different system events"""
        handlers = {
            "PRESENCE_DETECTED": self.handle_presence_detected,
            "WAKE_WORD_DETECTED": self.handle_wakeup_detected,
            "FACE_RECOGNIZED": self.handle_face_recognized,
            "FACE_NOT_RECOGNIZED": self.handle_face_not_recognized,
            "REGISTRATION_COMPLETE": self.handle_registration_complete,
            "UPDATE_COUNTDOWN": self.update_countdown,
            "RESET_TO_IDLE": self.reset_to_idle
        }
        handler = handlers.get(event)
        if handler:
            handler(data)
    
    def handle_presence_detected(self, distance):
        """Handle presence detection event"""
        if system_state["current_mode"] == "IDLE":
            system_state["current_mode"] = "WAKEUP_LISTENING"
            system_state["person_detected"] = True
            
            self.status_label.config(text="PRESENCE DETECTED")
            self.info_label.config(text=f"Say wake word to activate face recognition\nDistance: {distance:.1f} cm")
            
            # Start the wakeup word listener
            self.start_audio_stream()
            
            # Start timeout timer for reset
            if system_state["reset_timer"]:
                self.root.after_cancel(system_state["reset_timer"])
            system_state["reset_timer"] = self.root.after(WAKEUP_TIMEOUT * 1000, self.timeout_reset)
    
    def start_audio_stream(self):
        """Start the wake word detection"""
        try:
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback
            )
            self.audio_stream.start()
            print("Listening for wake word...")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            # Fall back to manual trigger for testing
            print("Audio device not available, using manual trigger ('w' key)")
    
    def audio_callback(self, indata, frames, time, status):
        """Process incoming audio data"""
        if system_state["current_mode"] != "WAKEUP_LISTENING":
            return
            
        self.audio_buffer[:-frames] = self.audio_buffer[frames:]
        self.audio_buffer[-frames:] = indata[:, 0]
        
        # Wake word detection
        if self.wake_word_model:
            audio_tensor = torch.from_numpy(self.audio_buffer).to(self.device)
            mel = self.transform(audio_tensor)
            prediction = torch.sigmoid(self.wake_word_model(mel.unsqueeze(0)))
            
            if prediction.item() > THRESHOLD:
                # Stop the audio stream to conserve resources
                self.audio_stream.stop()
                event_queue.put(("WAKE_WORD_DETECTED", None))
    
    def handle_wakeup_detected(self, _):
        """Handle wake word detection"""
        if system_state["current_mode"] == "WAKEUP_LISTENING":
            system_state["current_mode"] = "FACE_RECOGNITION"
            
            self.status_label.config(text="WAKE WORD DETECTED")
            self.info_label.config(text="Looking for face...")
            
            # Cancel the reset timer
            if system_state["reset_timer"]:
                self.root.after_cancel(system_state["reset_timer"])
            
            # Start the face recognition
            self.start_camera()
            self.root.after(3000, self.perform_face_recognition)
    
    def start_camera(self):
        """Initialize camera feed"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.update_camera_feed()
        except Exception as e:
            print(f"Error initializing camera: {e}")
            # If camera fails, proceed directly to face recognition simulation
            self.root.after(1000, self.perform_face_recognition)
    
    def update_camera_feed(self):
        """Update live camera preview"""
        if self.cap and system_state["current_mode"] == "FACE_RECOGNITION":
            ret, frame = self.cap.read()
            if ret:
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)
            self.root.after(10, self.update_camera_feed)
    
    def perform_face_recognition(self):
        """Perform face recognition check"""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame, recognized = self.face_recognition.recognize_face(frame)
                if recognized:
                    event_queue.put(("FACE_RECOGNIZED", "User"))
                else:
                    event_queue.put(("FACE_NOT_RECOGNIZED", None))
            else:
                # Camera read failed
                event_queue.put(("FACE_NOT_RECOGNIZED", None))
        else:
            # No camera available
            event_queue.put(("FACE_NOT_RECOGNIZED", None))
            
        # Clean up camera resources
        self.cleanup_camera()
    
    def handle_face_recognized(self, user_name):
        """Handle successful face recognition"""
        system_state["current_mode"] = "UNLOCKED"
        system_state["current_user"] = user_name
        
        self.status_label.config(text="ACCESS GRANTED")
        
        # Display welcome message
        self.welcome_label.config(text=f"Welcome, {user_name}")
        self.welcome_label.pack(pady=20)
        
        # Update info label
        self.info_label.config(text="Door unlocked for 10 seconds")
        
        # Unlock the door
        self.door_lock.unlock()
        
        # Set timer to lock after specified duration
        system_state["lock_timer"] = self.root.after(DOOR_UNLOCK_TIME * 1000, self.lock_door)
    
    def handle_face_not_recognized(self, _):
        """Handle failed face recognition"""
        system_state["current_mode"] = "FACE_NOT_RECOGNIZED"
        
        self.status_label.config(text="ACCESS DENIED")
        self.info_label.config(text="Face Not Recognised")
        
        # Show countdown before reset
        self.countdown_label.pack(pady=20)
        self.start_countdown(5)  # 5 second countdown before reset
    
    def lock_door(self):
        """De-activate the solenoid to lock the door"""
        self.door_lock.lock()
        self.info_label.config(text="Door Locked")
        
        # Hide welcome message
        self.welcome_label.pack_forget()
        
        # Reset to idle after lock
        self.root.after(1000, self.reset_to_idle)
    
    def start_countdown(self, seconds):
        """Start a countdown timer"""
        self.countdown_seconds = seconds
        self.update_countdown(seconds)
    
    def update_countdown(self, seconds):
        """Update the countdown display"""
        if seconds > 0:
            self.countdown_label.config(text=str(seconds))
            self.root.after(1000, lambda: event_queue.put(("UPDATE_COUNTDOWN", seconds - 1)))
        else:
            self.countdown_label.pack_forget()
            event_queue.put(("RESET_TO_IDLE", None))
    
    def timeout_reset(self):
        """Handle timeout for wake word detection"""
        if system_state["current_mode"] == "WAKEUP_LISTENING":
            self.info_label.config(text="Timeout - No wake word detected")
            self.root.after(2000, self.reset_to_idle)
    
    def reset_to_idle(self, _=None):
        """Reset the system to idle state"""
        system_state["current_mode"] = "IDLE"
        system_state["person_detected"] = False
        system_state["face_recognized"] = False
        system_state["current_user"] = ""
        
        # Reset UI
        self.status_label.config(text="SYSTEM IDLE")
        self.info_label.config(text="Waiting for presence detection...")
        self.welcome_label.pack_forget()
        self.countdown_label.pack_forget()
        
        # Stop any active streams
        self.cleanup_camera()
        self.stop_audio_stream()
        
        # Cancel any active timers
        if system_state["reset_timer"]:
            self.root.after_cancel(system_state["reset_timer"])
            system_state["reset_timer"] = None
            
        if system_state["lock_timer"]:
            self.root.after_cancel(system_state["lock_timer"])
            system_state["lock_timer"] = None
    
    def cleanup_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_label.config(image='')
        
    def stop_audio_stream(self):
        """Stop the audio stream"""
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                print("Audio stream stopped")
            except:
                pass
    
    def start_registration(self):
        """Start the face registration process"""
        # This would be implemented with actual face registration logic
        # For now, just show a message
        self.info_label.config(text="Registration feature not implemented")
        
    def handle_registration_complete(self, name):
        """Handle completed face registration"""
        self.info_label.config(text=f"Registration complete for {name}")
        self.root.after(2000, self.reset_to_idle)
    
    def exit_application(self):
        """Clean up and exit application"""
        self.cleanup_camera()
        self.stop_audio_stream()
        self.door_lock.cleanup()
        self.root.destroy()


# Simulated triggers for testing (to be used when hardware is unavailable)
def simulate_wakeup_word():
    """Simulate wake word detection (for testing)"""
    if system_state["current_mode"] == "WAKEUP_LISTENING":
        event_queue.put(("WAKE_WORD_DETECTED", None))

def simulate_face_recognized():
    """Simulate face recognition (for testing)"""
    if system_state["current_mode"] == "FACE_RECOGNITION":
        event_queue.put(("FACE_RECOGNIZED", "Test User"))

def simulate_face_not_recognized():
    """Simulate failed face recognition (for testing)"""
    if system_state["current_mode"] == "FACE_RECOGNITION":
        event_queue.put(("FACE_NOT_RECOGNIZED", None))


if __name__ == "__main__":
    # Set up the root window
    root = tk.Tk()
    app = DoorLockApp(root)
    
    # For development/testing only - bind keys to simulate events
    root.bind('w', lambda e: simulate_wakeup_word())
    root.bind('f', lambda e: simulate_face_recognized())
    root.bind('n', lambda e: simulate_face_not_recognized())
    
    # Start the Tkinter event loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        # Clean up resources
        GPIO.cleanup()