import tkinter as tk
from tkinter import ttk, font
import threading
import time
import queue
import RPi.GPIO as GPIO
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
SOLENOID_PIN = 18

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
        # Set up GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SOLENOID_PIN, GPIO.OUT)
        GPIO.output(SOLENOID_PIN, GPIO.LOW)  # Ensure door is locked initially
        
        # Initialize ultrasonic sensor
        self.ultrasonic = UltrasonicSensor(trigger_pin=17, echo_pin=27)
        
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
            self.start_wakeup_listener()
            
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
        # This will be implemented later
        # For now, we'll simulate this with a button press
        pass
    
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
        # This will be implemented later with the face recognition module
        registration_window = tk.Toplevel(self.root)
        registration_window.title("Face Registration")
        registration_window.geometry("600x400")
        
        # Add registration UI elements here
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
        
        # Buttons
        button_frame = ttk.Frame(registration_window)
        button_frame.pack(pady=20)
        
        capture_button = ttk.Button(
            button_frame,
            text="Capture Face",
            command=lambda: None  # Will be implemented with facial recognition
        )
        capture_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=registration_window.destroy
        )
        cancel_button.pack(side=tk.LEFT, padx=10)
    
    def handle_registration_complete(self, name):
        """Handle completed face registration"""
        # Update registered users (will be implemented with facial recognition)
        self.info_label.config(text=f"Registration complete for {name}")
        self.root.after(2000, self.reset_to_idle)
    
    def exit_application(self):
        """Clean up and exit application"""
        # Clean up GPIO
        GPIO.cleanup()
        self.root.destroy()


# Simulated triggers for testing (to be replaced with actual wake word and face recognition)
def simulate_wakeup_word():
    """Simulate wake word detection (for testing)"""
    if system_state["current_mode"] == "WAKEUP_LISTENING":
        event_queue.put(("WAKEUP_WORD_DETECTED", None))

def simulate_face_recognized():
    """Simulate face recognition (for testing)"""
    if system_state["current_mode"] == "FACE_RECOGNITION":
        event_queue.put(("FACE_RECOGNIZED", "John Doe"))

def simulate_face_not_recognized():
    """Simulate failed face recognition (for testing)"""
    if system_state["current_mode"] == "FACE_RECOGNITION":
        event_queue.put(("FACE_NOT_RECOGNIZED", None))


if __name__ == "__main__":
    # Set up the root window
    root = tk.Tk()
    app = DoorLockApp(root)
    
    # For development/testing only
    # Bind keys to simulate events
    root.bind('w', lambda e: simulate_wakeup_word())
    root.bind('f', lambda e: simulate_face_recognized())
    root.bind('n', lambda e: simulate_face_not_recognized())
    
    # Start the Tkinter event loop
    root.mainloop()