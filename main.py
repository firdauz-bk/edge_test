import tkinter as tk
import threading
import os
import time
from PIL import Image, ImageTk

# Import our modules
from ultrasonic import UltrasonicSensor
from wake_word import WakeWordDetector
from face_recognition import FaceRecognition
from door_lock import DoorLock

# GPIO Configuration
ULTRASONIC_TRIGGER_PIN = 17
ULTRASONIC_ECHO_PIN = 27
RELAY_PIN = 14
DETECTION_THRESHOLD_CM = 100  # 1 meter

# Timeout configuration
IDLE_TIMEOUT = 180  # 3 minutes in seconds

class DoorLockApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Door Lock System")
        self.root.geometry("640x580")
        
        # Create directory for saved faces if it doesn't exist
        if not os.path.exists("saved_faces"):
            os.makedirs("saved_faces")
            
        # Initialize components
        self.door_lock = DoorLock(RELAY_PIN)
        self.ultrasonic = UltrasonicSensor(ULTRASONIC_TRIGGER_PIN, ULTRASONIC_ECHO_PIN)
        self.wake_word_detector = WakeWordDetector()
        self.face_recognition = FaceRecognition()
        
        # State variables
        self.system_state = "idle"  # idle, listening, face_recognition
        self.wake_word_detected = False
        self.idle_timer = None
        
        # Create GUI elements
        self.create_gui_elements()
        
        # Start the ultrasonic monitoring in a separate thread
        self.start_ultrasonic_thread()
    
    def create_gui_elements(self):
        # Status display
        self.status_label = tk.Label(self.root, text="No Presence Detected (Idle Mode)", 
                                    font=('Helvetica', 14), fg='red')
        self.status_label.pack(pady=10)
        
        # Camera feed display
        self.camera_label = tk.Label(self.root)
        self.camera_label.pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.register_button = tk.Button(button_frame, text="Register Face", 
                                        command=self.open_face_registration)
        self.register_button.pack(side=tk.LEFT, padx=10)
        
        self.view_faces_button = tk.Button(button_frame, text="View Registered Faces", 
                                          command=self.display_saved_faces)
        self.view_faces_button.pack(side=tk.LEFT, padx=10)
    
    def start_ultrasonic_thread(self):
        """Start the ultrasonic sensor monitoring thread"""
        self.ultrasonic_thread = threading.Thread(target=self.monitor_presence)
        self.ultrasonic_thread.daemon = True
        self.ultrasonic_thread.start()
    
    def monitor_presence(self):
        """Monitor for presence using ultrasonic sensor"""
        while True:
            distance = self.ultrasonic.get_distance()
            # If someone is detected within threshold distance
            if distance <= DETECTION_THRESHOLD_CM and distance > 0:
                if self.system_state == "idle":
                    self.system_state = "listening"
                    self.update_status("Presence Detected - Listening for wake word...", 'green')
                    self.start_wake_word_detection()
                    # Start idle timer
                    if self.idle_timer is not None:
                        self.root.after_cancel(self.idle_timer)
                    self.idle_timer = self.root.after(IDLE_TIMEOUT * 1000, self.reset_to_idle)
            time.sleep(0.5)  # Check every half second
    
    def start_wake_word_detection(self):
        """Start listening for wake word"""
        self.wake_word_detector.start(callback=self.on_wake_word_detected)
    
    def stop_wake_word_detection(self):
        """Stop listening for wake word"""
        self.wake_word_detector.stop()
    
    def on_wake_word_detected(self):
        """Called when wake word is detected"""
        self.wake_word_detected = True
        self.system_state = "face_recognition"
        self.update_status("Wake Word Detected! Starting Face Recognition...", 'blue')
        
        # Stop wake word detection and start camera
        self.stop_wake_word_detection()
        self.start_face_recognition()
    
    def start_face_recognition(self):
        """Start the face recognition process"""
        self.face_recognition.start_camera()
        # Wait a moment for camera to start
        self.root.after(1000, self.perform_face_recognition)
    
    def perform_face_recognition(self):
        """Perform face recognition and unlock door if successful"""
        frame, name, recognized = self.face_recognition.recognize_person()
        
        # Display the frame
        if frame is not None:
            frame_rgb = self.face_recognition.convert_to_rgb(frame)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        
        # Handle recognition result
        if recognized:
            # Unlock the door
            self.door_lock.unlock()
            welcome_msg = f"Welcome, {name}!" if name else "Welcome!"
            self.update_status(welcome_msg, 'green')
            
            # Schedule door lock after 10 seconds
            self.root.after(10000, self.lock_door_and_reset)
        else:
            # Show countdown for 3 seconds before resetting
            self.update_status("Face Not Recognized - Resetting in 3...", 'red')
            self.root.after(1000, lambda: self.update_status("Face Not Recognized - Resetting in 2...", 'red'))
            self.root.after(2000, lambda: self.update_status("Face Not Recognized - Resetting in 1...", 'red'))
            self.root.after(3000, self.reset_to_idle)
        
        # Stop camera
        self.face_recognition.stop_camera()
    
    def lock_door_and_reset(self):
        """Lock the door and reset to idle state"""
        self.door_lock.lock()
        self.reset_to_idle()
    
    def reset_to_idle(self):
        """Reset the system to idle state"""
        # Cancel any pending resets
        if self.idle_timer is not None:
            self.root.after_cancel(self.idle_timer)
            self.idle_timer = None
        
        # Reset system state
        self.system_state = "idle"
        self.wake_word_detected = False
        
        # Stop any active components
        self.stop_wake_word_detection()
        self.face_recognition.stop_camera()
        
        # Clear camera display
        self.camera_label.configure(image='')
        
        # Update status
        self.update_status("No Presence Detected (Idle Mode)", 'red')
    
    def update_status(self, message, color):
        """Update the status display"""
        self.status_label.config(text=message, fg=color)
    
    def open_face_registration(self):
        """Open the face registration window"""
        # Stop any ongoing detection
        current_state = self.system_state
        self.stop_wake_word_detection()
        self.face_recognition.stop_camera()
        
        # Create registration window
        reg_window = tk.Toplevel(self.root)
        reg_window.title("Face Registration")
        reg_window.geometry("640x520")
        
        # Create fields for name
        name_frame = tk.Frame(reg_window)
        name_frame.pack(pady=10)
        
        tk.Label(name_frame, text="Name:").pack(side=tk.LEFT, padx=5)
        name_entry = tk.Entry(name_frame, width=30)
        name_entry.pack(side=tk.LEFT, padx=5)
        
        # Camera display
        camera_label = tk.Label(reg_window)
        camera_label.pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(reg_window)
        button_frame.pack(pady=10)
        
        def start_registration_camera():
            """Start camera feed for registration"""
            self.face_recognition.start_camera()
            update_registration_feed()
        
        def update_registration_feed():
            """Update camera feed in registration window"""
            if reg_window.winfo_exists():
                frame = self.face_recognition.get_frame()
                if frame is not None:
                    frame_rgb = self.face_recognition.convert_to_rgb(frame)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    camera_label.imgtk = imgtk
                    camera_label.configure(image=imgtk)
                    reg_window.after(10, update_registration_feed)
        
        def save_registered_face():
            """Save the current face with a name"""
            name = name_entry.get().strip()
            if not name:
                tk.messagebox.showerror("Error", "Please enter a name")
                return
                
            frame = self.face_recognition.get_frame()
            if frame is not None:
                success = self.face_recognition.register_face(frame, name)
                if success:
                    tk.messagebox.showinfo("Success", f"Face for {name} registered successfully!")
                    close_registration()
                else:
                    tk.messagebox.showerror("Error", "No face detected in the frame")
        
        def close_registration():
            """Close registration and restore previous state"""
            self.face_recognition.stop_camera()
            reg_window.destroy()
            
            # Restore previous state
            if current_state == "listening":
                self.start_wake_word_detection()
        
        save_button = tk.Button(button_frame, text="Save Face", command=save_registered_face)
        save_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = tk.Button(button_frame, text="Cancel", command=close_registration)
        cancel_button.pack(side=tk.LEFT, padx=10)
        
        # Start camera
        start_registration_camera()
        
        # Handle window close event
        reg_window.protocol("WM_DELETE_WINDOW", close_registration)
    
    def display_saved_faces(self):
        """Display all saved faces"""
        faces = self.face_recognition.get_registered_faces()
        
        if not faces:
            tk.messagebox.showinfo("No Faces", "No faces have been registered yet.")
            return
        
        # Create display window
        faces_window = tk.Toplevel(self.root)
        faces_window.title("Registered Faces")
        
        # Create a grid of faces
        row, col = 0, 0
        max_cols = 3
        
        for face_info in faces:
            name = face_info['name']
            image_path = face_info['path']
            
            # Create a frame for each face
            face_frame = tk.Frame(faces_window, padx=10, pady=10)
            face_frame.grid(row=row, column=col, padx=10, pady=10)
            
            # Load and display image
            img = Image.open(image_path)
            img = img.resize((150, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            img_label = tk.Label(face_frame, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack()
            
            # Display name
            name_label = tk.Label(face_frame, text=name)
            name_label.pack()
            
            # Move to next grid position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def cleanup(self):
        """Clean up resources on application exit"""
        self.door_lock.cleanup()
        self.face_recognition.cleanup()

# Application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = DoorLockApplication(root)
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        app.cleanup()