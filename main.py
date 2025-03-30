import tkinter as tk
from tkinter import ttk, font, simpledialog
import threading
import time
import queue
import RPi.GPIO as GPIO
import os
import cv2
from PIL import Image, ImageTk


from hardware.ultrasonic import UltrasonicSensor
from software.wake_word import WakeWordDetector
from software.face_recognition import FaceRecognitionSystem
from hardware.door_lock import DoorLock

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

# Constants
PRESENCE_THRESHOLD = 100  # 1 meter in cm
WAKEUP_TIMEOUT = 180  # 3 minutes in seconds
DOOR_UNLOCK_TIME = 10  # 10 seconds

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

        # Camera frame (initially hidden)
        self.camera_frame = ttk.Frame(self.main_frame)
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack()
        self.camera_active = False
        self.camera_update_id = None

        # Initialize hardware
        self.init_hardware()
        
        # Start the UI update thread
        self.update_thread = threading.Thread(target=self.process_events, daemon=True)
        self.update_thread.start()
        
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
        # Configure GPIO mode first
        GPIO.setmode(GPIO.BCM)

        # Initialize door lock
        self.door_lock = DoorLock(solenoid_pin=SOLENOID_PIN)
        
        # Initialize ultrasonic sensor
        self.ultrasonic = UltrasonicSensor(trigger_pin=17, echo_pin=27)
        
        # Initialize wake word detector
        self.wake_word_detector = WakeWordDetector(
            model_path="476998.pth",  # Adjust path as needed
            threshold=0.78,
            event_queue=event_queue
        )
        
        # Initialize face recognition system
        self.face_recognition = FaceRecognitionSystem(
            saved_faces_dir="saved_faces",
            recognition_threshold=0.3,
            event_queue=event_queue
        )
        
        # Start ultrasonic sensor thread
        self.ultrasonic_thread = threading.Thread(target=self.monitor_presence, daemon=True)
        self.ultrasonic_thread.start()

    def monitor_presence(self):
        """Thread function to monitor presence using ultrasonic sensor"""
        presence_detected = False
        
        while True:
            if system_state["current_mode"] == "IDLE":
                distance = self.ultrasonic.get_distance()
                
                # Check if someone is within detection range
                if distance <= PRESENCE_THRESHOLD and not presence_detected:
                    presence_detected = True
                    event_queue.put(("PRESENCE_DETECTED", distance))
                
                # Reset detection state if person moves away
                elif distance > PRESENCE_THRESHOLD + 20 and presence_detected:
                    presence_detected = False
                
                # Small sleep to prevent CPU overuse
                time.sleep(0.1)
            else:
                # If not in IDLE mode, just wait and check periodically
                presence_detected = False
                time.sleep(0.5)
    
    def process_events(self):
        """Process events from the queue and update UI accordingly"""
        while True:
            try:
                event, data = event_queue.get(block=True, timeout=0.1)
                
                # Use after() to update UI from the main thread
                if event == "PRESENCE_DETECTED":
                    self.root.after(0, self.handle_presence_detected, data)
                elif event == "WAKEUP_WORD_DETECTED":
                    self.root.after(0, self.handle_wakeup_detected)
                elif event == "FACE_RECOGNIZED":
                    self.root.after(0, self.handle_face_recognized, data)
                elif event == "FACE_NOT_RECOGNIZED":
                    self.root.after(0, self.handle_face_not_recognized)
                elif event == "REGISTRATION_COMPLETE":
                    self.root.after(0, self.handle_registration_complete, data)
                elif event == "UPDATE_COUNTDOWN":
                    self.root.after(0, self.update_countdown, data)
                elif event == "RESET_TO_IDLE":
                    self.root.after(0, self.reset_to_idle)
                
            except queue.Empty:
                # Queue is empty, continue the loop
                continue
    
    def handle_presence_detected(self, distance):
        """Handle presence detection event"""
        if system_state["current_mode"] == "IDLE":
            system_state["current_mode"] = "WAKEUP_LISTENING"
            system_state["person_detected"] = True
            
            self.status_label.config(text="PRESENCE DETECTED")
            self.info_label.config(text=f"Say wake word to activate face recognition\nDistance: {distance:.1f} cm")
            
            # Start the wakeup word listener (to be implemented)
            self.wake_word_detector.start_listening()
            
            # Start timeout timer for reset
            if system_state["reset_timer"]:
                self.root.after_cancel(system_state["reset_timer"])
            system_state["reset_timer"] = self.root.after(WAKEUP_TIMEOUT * 1000, self.timeout_reset)
    
    def start_wakeup_listener(self):
        """Start the wake word detection thread"""
        # This will be implemented later
        # For now, we'll simulate this with a button press
        
        # In a real implementation, this would start microphone listening
        pass
    
    def handle_wakeup_detected(self):
        """Handle wake word detection"""
        if system_state["current_mode"] == "WAKEUP_LISTENING":
            system_state["current_mode"] = "FACE_RECOGNITION"
            
            self.status_label.config(text="WAKE WORD DETECTED")
            self.info_label.config(text="Looking for face...")
            
            # Cancel the reset timer
            if system_state["reset_timer"]:
                self.root.after_cancel(system_state["reset_timer"])
            
            # Start the face recognition (to be implemented)
            self.start_face_recognition()
    
    def start_face_recognition(self):
        """Start the face recognition process"""
        # Activate the camera display
        self.camera_frame.pack(pady=20)
        self.camera_active = True
        
        # Start face recognition system
        self.face_recognition.start_camera()
        
        # Start camera feed updates
        self.update_camera_feed()

    def update_camera_feed(self):
        """Update the camera feed on the UI"""
        if not self.camera_active:
            return
            
        # Get frame from face recognition system (non-blocking)
        if hasattr(self.face_recognition, 'cap') and self.face_recognition.cap:
            cap = self.face_recognition.cap
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Convert to RGB for display
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image
                    img = Image.fromarray(frame)
                    # Convert to PhotoImage
                    imgtk = ImageTk.PhotoImage(image=img)
                    # Update label
                    self.camera_label.imgtk = imgtk
                    self.camera_label.configure(image=imgtk)
        
        # Schedule next update if still active
        if self.camera_active:
            self.camera_update_id = self.root.after(33, self.update_camera_feed)  # ~30fps
    
    def stop_camera_feed(self):
        """Stop the camera feed"""
        self.camera_active = False
        if self.camera_update_id:
            self.root.after_cancel(self.camera_update_id)
            self.camera_update_id = None
        self.camera_frame.pack_forget()
    
    def handle_face_recognized(self, user_name):
        """Handle successful face recognition"""
        system_state["current_mode"] = "UNLOCKED"
        system_state["current_user"] = user_name
        
        self.status_label.config(text="ACCESS GRANTED")
        
        # Display welcome message
        self.welcome_label.config(text=f"Welcome, {user_name}")
        self.welcome_label.pack(pady=20)
        
        # Unlock the door
        self.unlock_door()
        
        # Set timer to lock after specified duration
        system_state["lock_timer"] = self.root.after(DOOR_UNLOCK_TIME * 1000, self.lock_door)
    
    def handle_face_not_recognized(self):
        """Handle failed face recognition"""
        system_state["current_mode"] = "FACE_NOT_RECOGNIZED"
        
        self.stop_camera_feed()
        
        self.status_label.config(text="ACCESS DENIED")
        self.info_label.config(text="Face Not Recognized")
        
        # Show countdown before reset
        self.countdown_label.pack(pady=20)
        self.start_countdown(5)  # 5 second countdown
    
    def unlock_door(self):
        """Activate the solenoid to unlock the door"""
        GPIO.output(SOLENOID_PIN, GPIO.HIGH)
        self.info_label.config(text="Door Unlocked")
    
    def lock_door(self):
        """De-activate the solenoid to lock the door"""
        GPIO.output(SOLENOID_PIN, GPIO.LOW)
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
            event_queue.put(("UPDATE_COUNTDOWN", seconds - 1))
            self.root.after(1000, lambda: event_queue.put(("UPDATE_COUNTDOWN", seconds - 1)))
        else:
            self.countdown_label.pack_forget()
            event_queue.put(("RESET_TO_IDLE", None))
    
    def timeout_reset(self):
        """Handle timeout for wake word detection"""
        if system_state["current_mode"] == "WAKEUP_LISTENING":
            self.info_label.config(text="Timeout - No wake word detected")
            self.root.after(2000, self.reset_to_idle)
    
    def reset_to_idle(self):
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
        
        # Cancel any active timers
        if system_state["reset_timer"]:
            self.root.after_cancel(system_state["reset_timer"])
            system_state["reset_timer"] = None
            
        if system_state["lock_timer"]:
            self.root.after_cancel(system_state["lock_timer"])
            system_state["lock_timer"] = None
    
    def start_registration(self):
        """Start the face registration process"""
        registration_window = tk.Toplevel(self.root)
        registration_window.title("Face Registration")
        registration_window.geometry("600x400")
        
        # Add registration UI elements
        label = ttk.Label(
            registration_window,
            text="Face Registration Mode",
            font=font.Font(family="Helvetica", size=18, weight="bold")
        )
        label.pack(pady=20)
        
        # Name entry
        name_frame = ttk.Frame(registration_window)
        name_frame.pack(pady=20)
        
        ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT, padx=5)
        name_entry = ttk.Entry(name_frame, font=font.Font(family="Helvetica", size=14))
        name_entry.pack(side=tk.LEFT, padx=5)
        
        # Camera preview
        camera_frame = ttk.Frame(registration_window)
        camera_frame.pack(pady=10)
        camera_label = ttk.Label(camera_frame)
        camera_label.pack()
        
        # Start camera for preview
        cap = cv2.VideoCapture(0)
        
        def update_preview():
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Convert to RGB for display
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image
                    img = Image.fromarray(frame)
                    # Resize for preview
                    img = img.resize((320, 240))
                    # Convert to PhotoImage
                    imgtk = ImageTk.PhotoImage(image=img)
                    # Update label
                    camera_label.imgtk = imgtk
                    camera_label.configure(image=imgtk)
                    # Schedule next update
                    camera_label.after(33, update_preview)  # ~30fps
        
        # Start preview
        update_preview()
        
        # Status message
        status_label = ttk.Label(
            registration_window,
            text="Enter name and click 'Capture Face'",
            font=font.Font(family="Helvetica", size=12)
        )
        status_label.pack(pady=10)
        
        # Buttons
        button_frame = ttk.Frame(registration_window)
        button_frame.pack(pady=20)
        
        def capture_face():
            name = name_entry.get().strip()
            if not name:
                status_label.config(text="Please enter a name first!")
                return
                
            # Register face using face recognition system
            success = self.face_recognition.register_face(name)
            
            if success:
                status_label.config(text=f"Successfully registered {name}!")
                # Update UI in main window
                if self.event_queue:
                    self.event_queue.put(("REGISTRATION_COMPLETE", name))
                # Close window after short delay
                registration_window.after(2000, registration_window.destroy)
            else:
                status_label.config(text="Registration failed! Please try again.")
        
        capture_button = ttk.Button(
            button_frame,
            text="Capture Face",
            command=capture_face
        )
        capture_button.pack(side=tk.LEFT, padx=10)
        
        def cancel_registration():
            cap.release()
            registration_window.destroy()
        
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=cancel_registration
        )
        cancel_button.pack(side=tk.LEFT, padx=10)
        
        # Clean up on window close
        registration_window.protocol("WM_DELETE_WINDOW", cancel_registration)
    
    def handle_registration_complete(self, name):
        """Handle completed face registration"""
        # Update registered users (will be implemented with facial recognition)
        self.info_label.config(text=f"Registration complete for {name}")
        self.root.after(2000, self.reset_to_idle)
    
    def exit_application(self):
        """Clean up and exit application"""
        # Stop all threads and cleanup resources
        self.stop_camera_feed()
        
        if hasattr(self, 'wake_word_detector'):
            self.wake_word_detector.cleanup()
            
        if hasattr(self, 'face_recognition'):
            self.face_recognition.cleanup()
            
        if hasattr(self, 'door_lock'):
            self.door_lock.cleanup()
            
        if hasattr(self, 'ultrasonic'):
            self.ultrasonic.cleanup()
            
        # Clean up remaining GPIO
        try:
            GPIO.cleanup()
        except:
            pass
            
        self.root.destroy()

if __name__ == "__main__":
    # Set up the root window
    root = tk.Tk()
    app = DoorLockApp(root)

    # Start the Tkinter event loop
    root.mainloop()