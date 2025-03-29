import tkinter as tk
from threading import Thread
import time
import os

from software.wake_word import WakeWordDetector
from software.face_recognition import FaceRecognition
from hardware.door_lock import DoorLock
from hardware.ultrasonic import UltrasonicSensor

# Create directory for saved faces
if not os.path.exists("saved_faces"):
    os.makedirs("saved_faces")

class SmartDoorSystem:
    def __init__(self):
        # Initialize components
        self.door_lock = DoorLock()
        self.ultrasonic = UltrasonicSensor(trigger_pin=17, echo_pin=27)
        self.wake_word_detector = WakeWordDetector(callback=self.wake_word_callback)
        self.face_recognition = FaceRecognition(callback=self.face_recognized_callback)
        
        # System state
        self.wake_word_detected = False
        self.system_active = True
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Smart Door Lock System")
        self.root.geometry("640x580")
        
        # Status label
        self.status_label = tk.Label(self.root, text="System in idle mode. Waiting for approach...")
        self.status_label.pack(pady=10)
        
        # Camera feed label
        self.camera_label = tk.Label(self.root)
        self.camera_label.pack()
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        
        register_button = tk.Button(button_frame, text="Register Face", 
                                   command=self.register_face)
        register_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        display_button = tk.Button(button_frame, text="Display Saved Faces", 
                                  command=self.display_saved_faces)
        display_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        test_button = tk.Button(button_frame, text="Test Door Lock", 
                               command=self.door_lock.test_cycle)
        test_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    def start(self):
        # Start monitoring thread
        self.monitor_thread = Thread(target=self.monitor_approach, daemon=True)
        self.monitor_thread.start()
        
        # Start GUI main loop
        try:
            self.root.mainloop()
        finally:
            self.cleanup()
    
    def monitor_approach(self):
        """Thread that continuously monitors for approaching people using ultrasonic sensor"""
        while self.system_active:
            distance = self.ultrasonic.get_distance()
            
            if distance <= 100:  # Within 1 meter
                self.status_label.config(text=f"Person detected at {distance}cm. Listening for wake word...")
                # Start listening for wake word
                if not self.wake_word_detector.is_running:
                    self.wake_word_detector.start()
            else:
                # Stop wake word detection when no one is near
                if self.wake_word_detector.is_running and not self.wake_word_detected:
                    self.wake_word_detector.stop()
                    self.status_label.config(text="System in idle mode. Waiting for approach...")
            
            time.sleep(0.5)  # Check every half second
    
    def wake_word_callback(self):
        """Called when wake word is detected"""
        self.wake_word_detected = True
        self.status_label.config(text="Wake word detected! Starting face recognition...")
        
        # Add a small delay before stopping the detector
        self.root.after(100, self.wake_word_detector.stop)
        
        # Start camera and face recognition with a longer delay
        self.face_recognition.start_camera()
        self.root.after(3000, self.face_recognition.recognize_face)
    
    def face_recognized_callback(self, recognized):
        """Called after face recognition is complete"""
        if recognized:
            self.status_label.config(text="Face recognized! Door unlocked.")
            self.door_lock.unlock()
            self.root.after(10000, self.door_lock.lock)  # Lock after 10 seconds
        else:
            self.status_label.config(text="Face not recognized. Access denied.")
        
        # Reset system
        self.wake_word_detected = False
        self.face_recognition.stop_camera()
    
    def register_face(self):
        """Open face registration interface"""
        # Stop detection temporarily
        self.wake_word_detector.stop()
        self.face_recognition.register_face(self.root, self.registration_complete)
    
    def registration_complete(self):
        """Called when face registration is complete"""
        # Restart detection if needed
        if not self.wake_word_detected:
            self.wake_word_detector.start()
    
    def display_saved_faces(self):
        """Show saved face images"""
        self.face_recognition.display_saved_faces(self.root)
    
    def cleanup(self):
        """Clean up resources when program exits"""
        self.system_active = False
        self.wake_word_detector.stop()
        self.face_recognition.stop_camera()
        self.door_lock.cleanup()
        self.ultrasonic.cleanup()

if __name__ == "__main__":
    system = SmartDoorSystem()
    system.start()