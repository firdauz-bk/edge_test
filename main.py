import tkinter as tk
import threading
from threading import Thread, Event
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import torch
import sounddevice as sd
import torchaudio.transforms as T
import time
import socket
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

from software.wake_word import WakeWordModel, start_audio_stream, stop_audio_stream, set_callback, detect_wake_word, pause_audio_stream, resume_audio_stream
from software.face_recognition import FaceRecognition
from hardware.door_lock import DoorLock
from hardware.ultrasonic import UltrasonicSensor

# Global variables
wake_word_detected = False
presence_detected = False
cap = None
face_window = None
audio_stream = None
ultrasonic_thread_running = False
ultrasonic_stop_event = Event()
reset_timer = None
audio_system_busy = False

# Dashboard Stuff
dashboard_server = None
system_status = {
    "presence_detected": False,
    "wake_word_detected": False,
    "face_recognized": False,
    "door_locked": True,
    "system_mode": "Idle",
    "last_event": "",
    "last_event_time": "",
}

# Set up audio buffer
SAMPLE_RATE = 16000
DURATION = 1.0
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
THRESHOLD = 0.78

# Ultrasonic sensor settings
TRIGGER_PIN = 17
ECHO_PIN = 27
PRESENCE_DISTANCE = 50  # Distance in cm (1m)
PRESENCE_TIMEOUT = 120   # 2 minutes timeout in seconds

# Load wake word model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wake_word_model = WakeWordModel().to(device)
wake_word_model.load_state_dict(torch.load("476998.pth", map_location=device))
wake_word_model.eval()

# Initialize transform for mel spectrogram (separate from model to match original)
transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64, n_fft=400, hop_length=160).to(device)

# Initialize face recognition
face_recognition = FaceRecognition()

# Initialize lock system
lock_system = DoorLock()

# Initialize ultrasonic sensor
ultrasonic_sensor = UltrasonicSensor(TRIGGER_PIN, ECHO_PIN)

# Camera settings
webcam_resolution = (640, 480)

class FaceRecognition:
    def __init__(self):
        # Get absolute path to the saved_faces directory
        self.faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_faces")
        print(f"FaceRecognition initialized. Looking for faces in: {self.faces_dir}")
        # Load saved faces from absolute path
        self.load_known_faces()

# --- Dashboard Thread ---
# Create a simple HTTP server for the dashboard
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global system_status
        
        # Parse the path
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # Handle API requests
        if path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')  # Allow cross-origin requests
            self.end_headers()
            
            # Add current time to status
            system_status["current_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            self.wfile.write(json.dumps(system_status).encode())
            return
        
        # Handle commands
        elif path == '/api/command':
            query = urllib.parse.parse_qs(parsed_path.query)
            
            if 'action' in query:
                action = query['action'][0]
                
                if action == 'reset':
                    # Schedule reset on the main thread
                    root.after(0, reset_to_idle_mode)
                    response = {"status": "success", "message": "System reset initiated"}
                elif action == 'register_face':
                    # Schedule face registration on the main thread
                    root.after(0, register_face)
                    response = {"status": "success", "message": "Face registration initiated"}
                elif action == 'unlock':
                    # Schedule door unlock on the main thread
                    root.after(0, lambda: unlock_door_manually())
                    response = {"status": "success", "message": "Door unlock initiated"}
                else:
                    response = {"status": "error", "message": "Unknown action"}
            else:
                response = {"status": "error", "message": "No action specified"}
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Serve the dashboard HTML
        elif path == '/' or path == '':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Simple HTML page that redirects to the dashboard URL
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Smart Door Lock System</title>
                <meta http-equiv="refresh" content="0;url=http://localhost:5000">
            </head>
            <body>
                <p>Redirecting to dashboard...</p>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode())
            return
        
        # Default 404 response for any other path
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        # Silence HTTP server logs to avoid cluttering console
        return

# Function to start the dashboard server
def start_dashboard_server():
    global dashboard_server
    
    try:
        # Use port 8080 for the API
        server_address = ('', 8080)
        dashboard_server = HTTPServer(server_address, DashboardHandler)
        
        print(f"Starting dashboard server on http://localhost:8080")
        dashboard_thread = threading.Thread(target=dashboard_server.serve_forever, daemon=True)
        dashboard_thread.start()
    except Exception as e:
        print(f"Error starting dashboard server: {e}")

# Function to stop the dashboard server
def stop_dashboard_server():
    global dashboard_server
    if dashboard_server:
        dashboard_server.shutdown()
        print("Dashboard server stopped")

# Function to update the system status
def update_system_status(key, value, event_desc=None):
    global system_status
    system_status[key] = value
    
    if event_desc:
        system_status["last_event"] = event_desc
        system_status["last_event_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

# Add this to unlock the door manually (for dashboard control)
def unlock_door_manually():
    lock_system.unlock()
    update_system_status("door_locked", False, "Door manually unlocked from dashboard")
    status_label.config(text="Door manually unlocked from dashboard")
    
    # Set timer to lock door after 10 seconds
    root.after(10000, lock_door_and_reset)

# Modify existing functions to update system_status

# --- Ultrasonic Sensor Thread ---
def ultrasonic_thread():
    global presence_detected, ultrasonic_thread_running
    ultrasonic_thread_running = True
    ultrasonic_stop_event.clear()
    
    print("Starting ultrasonic sensor thread - low power mode active")
    status_label.config(text="Idle mode: Waiting for presence detection")
    
    try:
        while not ultrasonic_stop_event.is_set():
            distance = ultrasonic_sensor.get_distance()
            print(f"Distance: {distance} cm")
            
            if distance <= PRESENCE_DISTANCE and not presence_detected:
                presence_detected = True
                print("Presence detected within 50cm!")
                root.after(0, presence_detected_callback)
                break
                
            time.sleep(0.5)  # Reduced polling for power efficiency
    except Exception as e:
        print(f"Ultrasonic sensor error: {e}")
    finally:
        ultrasonic_thread_running = False
        print("Ultrasonic sensor thread stopped")

def start_ultrasonic_detection():
    global ultrasonic_thread_running
    if not ultrasonic_thread_running:
        Thread(target=ultrasonic_thread, daemon=True).start()

def stop_ultrasonic_detection():
    global ultrasonic_stop_event
    ultrasonic_stop_event.set()

def presence_detected_callback():
    global presence_detected, audio_system_busy
    print("Presence detected! Starting wake word detection...")
    status_label.config(text="Presence detected - Listening for wake word")
    
    # Update system status
    update_system_status("presence_detected", True, "Presence detected")
    update_system_status("system_mode", "Listening for wake word")

    # Check if audio system is busy
    if audio_system_busy:
        print("Audio system busy, waiting before starting...")
        root.after(1000, presence_detected_callback)
        return
    
    # Start audio detection for wake word
    audio_system_busy = True
    set_callback(audio_callback)
    
    # Use a separate thread to start audio to prevent UI freezing
    Thread(target=lambda: safe_start_audio_stream(), daemon=True).start()
    
    # Set timeout for wake word detection
    schedule_reset_timer()

def safe_start_audio_stream():
    global audio_system_busy
    try:
        # More thorough cleanup process
        stop_audio_stream()  # Make sure any existing stream is stopped
        
        # Force reset sounddevice before starting
        sd._terminate()
        time.sleep(1.0)  # Longer sleep
        sd._initialize()
        time.sleep(1.0)  # Longer sleep
        
        # Reset defaults
        sd.default.device = 0
        sd.default.channels = 1
        sd.default.samplerate = 16000
        
        # Start the stream
        start_audio_stream()
        audio_system_busy = False
        print("Audio stream started successfully")
    except Exception as e:
        print(f"Error in safe_start_audio_stream: {e}")
        audio_system_busy = False
        # If can't start audio, reset to idle mode
        root.after(2000, reset_to_idle_mode)  # longer delay

def schedule_reset_timer():
    global reset_timer
    # Cancel any existing timer
    if reset_timer is not None:
        root.after_cancel(reset_timer)
    
    # Set a new timer to reset system after 3 minutes if no wake word
    reset_timer = root.after(PRESENCE_TIMEOUT * 1000, reset_to_idle_mode)

def reset_to_idle_mode():
    global presence_detected, wake_word_detected, cap, reset_timer, audio_system_busy
    
    print("Resetting to idle mode...")
    status_label.config(text="Resetting to idle mode")
    
    # Update system status
    update_system_status("presence_detected", False)
    update_system_status("wake_word_detected", False)
    update_system_status("face_recognized", False)
    update_system_status("system_mode", "Idle")
    update_system_status("last_event", "System reset to idle")
    update_system_status("last_event_time", time.strftime("%Y-%m-%d %H:%M:%S"))

    # Reinitialize face recognition to ensure fresh state
    face_recognition = FaceRecognition()

    # Clear states
    presence_detected = False
    wake_word_detected = False
    audio_system_busy = True  # Mark as busy during reset
    
    # Cancel timer
    if reset_timer is not None:
        root.after_cancel(reset_timer)
        reset_timer = None
    
    # Stop any active processes
    stop_audio_stream()
    
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
    
    # Reset UI
    camera_label.config(image='')
    
    # Give system time to release resources
    def delayed_restart():
        global audio_system_busy
        audio_system_busy = False
        status_label.config(text="Idle mode: Waiting for presence detection")
        start_ultrasonic_detection()
    
    # Delay restart to allow resources to be released
    root.after(2000, delayed_restart)

# Replace stop_audio_stream() with pause_audio_stream() in the audio_callback function:
def audio_callback(indata, frames, time, status):
    global wake_word_detected, audio_buffer
    if status:
        print(f"Audio callback error: {status}")
    
    audio_buffer[:-frames] = audio_buffer[frames:]
    audio_buffer[-frames:] = indata[:, 0]
    
    # Use the model to detect wake word
    prediction = detect_wake_word(wake_word_model, audio_buffer, device)
    print(f"Wake word probability: {prediction}")

    if prediction > THRESHOLD:
        wake_word_detected = True
        print("Wake word detected! Scanning face...")
        status_label.config(text="Wake word detected - Starting face scan")
        pause_audio_stream()  # Just pause instead of stopping
        
        # Update system status
        update_system_status("wake_word_detected", True, "Wake word detected")
        update_system_status("system_mode", "Scanning face")
        

        # Cancel reset timer
        global reset_timer
        if reset_timer is not None:
            root.after_cancel(reset_timer)
            reset_timer = None
            
        root.after(0, start_camera)
        root.after(3000, perform_face_recognition)  # 3 second delay

# --- Camera handling functions ---
def start_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_resolution[1])
        update_camera_feed()

def update_camera_feed():
    global cap
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.configure(image=imgtk)
        root.after(10, update_camera_feed)

def open_face_camera():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_resolution[1])

# --- Face registration functions ---
def register_face():
    global cap, face_window, wake_word_detected, presence_detected
    print(f"Current state - Wake word: {wake_word_detected}, Presence: {presence_detected}")
    
    # Stop any active detection processes
    stop_audio_stream()
    stop_ultrasonic_detection()
    
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None  

    face_window = tk.Toplevel(root)
    face_window.title("Face Registration")
    face_window.geometry("640x580")
    # Make the window modal to prevent interactions with the main window
    face_window.transient(root)
    face_window.grab_set()

    face_label = tk.Label(face_window)
    face_label.pack()

    save_button = tk.Button(face_window, text="Save Face", command=save_face)
    save_button.pack(pady=10)

    close_button = tk.Button(face_window, text="Close", command=close_face_registration)
    close_button.pack(pady=10)

    # Run camera feed in a separate thread
    thread = Thread(target=open_face_camera)
    thread.daemon = True
    thread.start()

    update_face_feed(face_label)

def close_face_registration():
    global cap, face_window, wake_word_detected, presence_detected
    print(f"Current state - Wake word: {wake_word_detected}, Presence: {presence_detected}")
    
    if face_window.winfo_exists():
        face_window.grab_release()  # Release the grab before destroying
        face_window.destroy()

    if cap is not None:
        cap.release()
        cap = None  # Avoid using a released camera
    
    # Reset to idle mode regardless of current state
    reset_to_idle_mode()

def update_face_feed(face_label):
    if face_window.winfo_exists():  # Ensure window still exists
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                face_label.imgtk = imgtk
                face_label.configure(image=imgtk)

        if face_window.winfo_exists():  # Double-check before scheduling the next update
            face_window.after(10, lambda: update_face_feed(face_label))

def save_face():
    global cap, face_window
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Ensure directory exists
            if not os.path.exists("saved_faces"):
                os.makedirs("saved_faces")
                
            face_filename = f"saved_faces/face_1.png"
            cv2.imwrite(face_filename, frame)
            print(f"Face saved as {face_filename}")
            cap.release()

    # Ensure the face_window is properly closed
    if face_window:
        face_window.grab_release()  # Release the grab before destroying
        face_window.destroy()  # Close the registration window
    
    # Reset to idle mode
    reset_to_idle_mode()

# --- Face Recognition ---
def perform_face_recognition():
    global cap, wake_word_detected
    
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame, recognized = face_recognition.recognize_face(frame)
            result_text = "Face Recognized!" if recognized else "Face Not Recognized"
            
            # Update system status
            update_system_status("face_recognized", recognized, 
                              "Face recognized" if recognized else "Face not recognized")

            # Draw result on frame
            cv2.putText(frame, result_text, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0) if recognized else (0, 0, 255), 2)        

            # If recognized, unlock door
            if recognized:
                status_label.config(text="Face recognized - Door unlocked")
                lock_system.unlock()

                # Update system status
                update_system_status("door_locked", False, "Door unlocked")
                update_system_status("system_mode", "Door unlocked")

                # Lock after 10 seconds as specified in requirements
                root.after(10000, lock_door_and_reset)
            else:
                # Show countdown for 5 seconds before resetting
                status_label.config(text="Face not recognized - Resetting in 5 seconds")
                
                # Update system status
                update_system_status("system_mode", "Face not recognized - Resetting")

                countdown_seconds = 5
                start_countdown(countdown_seconds)
            
            # Update camera feed with recognition result
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.configure(image=imgtk)
    
    wake_word_detected = False  # Reset wake word flag

def start_countdown(seconds):
    if seconds > 0:
        status_label.config(text=f"Face not recognized - Resetting in {seconds} seconds")
        root.after(1000, lambda: start_countdown(seconds - 1))
    else:
        reset_to_idle_mode()

def lock_door_and_reset():
    lock_system.lock()

    # Update system status
    update_system_status("door_locked", True, "Door locked")
    
    status_label.config(text="Door locked - Resetting system")
    root.after(1000, reset_to_idle_mode)

def display_saved_faces():
    face_window = tk.Toplevel(root)
    face_window.title("Saved Faces")
    
    if os.path.exists("saved_faces"):
        for i, filename in enumerate(os.listdir("saved_faces")):
            face_path = os.path.join("saved_faces", filename)
            img = Image.open(face_path)
            img = img.resize((100, 100))
            imgtk = ImageTk.PhotoImage(img)
            label = tk.Label(face_window, image=imgtk)
            label.image = imgtk
            label.grid(row=i // 5, column=i % 5, padx=5, pady=5)

# --- Reset audio system completely ---
def reset_audio():
    global audio_system_busy
    
    status_label.config(text="Resetting audio system...")
    audio_system_busy = True
    
    # Stop any existing audio stream
    stop_audio_stream()
    
    # Force reset sounddevice
    try:
        sd._terminate()
        time.sleep(1)
        sd._initialize()
        time.sleep(0.5)
        
        # Reset defaults
        sd.default.device = 0
        sd.default.channels = 1
        sd.default.samplerate = 16000
        
        print("Audio system reset complete")
        status_label.config(text="Audio system reset complete")
        audio_system_busy = False
        
        # Reset to idle mode
        root.after(1000, reset_to_idle_mode)
    except Exception as e:
        print(f"Error during audio reset: {e}")
        status_label.config(text=f"Audio reset error: {e}")
        audio_system_busy = False
        root.after(1000, reset_to_idle_mode)

def on_closing():
    print("Application closing, cleaning up resources...")
    stop_ultrasonic_detection()
    stop_audio_stream()
    stop_dashboard_server()  # Stop dashboard server

    if cap is not None and cap.isOpened():
        cap.release()
    lock_system.cleanup()
    sd._terminate()  # Make sure to terminate sounddevice
    root.destroy()



# --- Main application ---
if __name__ == "__main__":
    # Set sounddevice defaults
    sd.default.device = 0
    sd.default.channels = 1
    sd.default.samplerate = 16000
    
    # Set up Tkinter UI
    root = tk.Tk()
    root.title("Smart Door Lock System")
    root.geometry("640x680")
    
    # Create status label
    status_label = tk.Label(root, text="Initializing system...", font=("Arial", 14))
    status_label.pack(pady=10)
    
    # Camera feed label
    camera_label = tk.Label(root)
    camera_label.pack()
    
    # Button frame
    button_frame = tk.Frame(root)
    button_frame.pack()
    
    # Register face button
    register_button = tk.Button(button_frame, text="Register Face", command=register_face)
    register_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Display saved faces button
    display_button = tk.Button(button_frame, text="Display Saved Faces", command=display_saved_faces)
    display_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Manual reset button
    reset_button = tk.Button(button_frame, text="Reset System", command=reset_to_idle_mode)
    reset_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Audio reset button - Special button to force audio system reset
    audio_reset_button = tk.Button(root, text="Reset Audio System", command=reset_audio, bg='#ffcccc')
    audio_reset_button.pack(pady=10)
    
    # Status and debug info
    audio_status = tk.Label(root, text="Audio device info:")
    audio_status.pack(pady=5)
    
    # Start the dashboard server
    start_dashboard_server()

    try:
        devices_info = f"Audio devices: {sd.query_devices()}"
        device_label = tk.Label(root, text=devices_info[:80] + "...", font=("Courier", 8))
        device_label.pack()
    except:
        device_label = tk.Label(root, text="Could not query audio devices", font=("Courier", 8))
        device_label.pack()
    
    # Set callback for wake word detection
    set_callback(audio_callback)
    
    # Force reset of audio system before starting
    sd._terminate()
    time.sleep(0.5)
    sd._initialize()
    time.sleep(0.5)
    
    # Start the app in idle mode using the ultrasonic sensor
    root.after(1000, start_ultrasonic_detection)
    root.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        root.mainloop()
    finally:
        # Cleanup
        stop_ultrasonic_detection()
        stop_audio_stream()
        if cap is not None and cap.isOpened():
            cap.release()
        lock_system.cleanup()