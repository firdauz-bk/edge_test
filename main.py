import tkinter as tk
from threading import Thread
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import torch
import sounddevice as sd
import torchaudio.transforms as T

from software.wake_word import WakeWordModel, start_audio_stream, stop_audio_stream, set_callback, detect_wake_word
from software.face_recognition import FaceRecognition
from hardware.door_lock import DoorLock

# Global variables
wake_word_detected = False
cap = None
face_window = None
audio_stream = None

# Set up audio buffer
SAMPLE_RATE = 16000
DURATION = 1.0
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
THRESHOLD = 0.78

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

# Camera settings
webcam_resolution = (640, 480)

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
        stop_audio_stream()
        root.after(0, start_camera)
        print("Scanning face...")
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
    global cap, face_window, wake_word_detected
    print(wake_word_detected)
    if not wake_word_detected:
        stop_audio_stream()  # Pause audio thread
        
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
    global cap, face_window, wake_word_detected
    print(wake_word_detected)
    if face_window.winfo_exists():
        face_window.grab_release()  # Release the grab before destroying
        face_window.destroy()

    if cap is not None:
        cap.release()
        cap = None  # Avoid using a released camera
    
    if not wake_word_detected:
        # Make sure to restart the audio stream
        root.after(100, start_audio_stream)  # Small delay to ensure cleanup is complete
    else:
        root.after(100, start_camera)  # Small delay to ensure cleanup is complete

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
    global cap, face_window, wake_word_detected
    print(wake_word_detected)
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
    
    if not wake_word_detected:
        # Make sure to restart the audio stream
        root.after(100, start_audio_stream)  # Small delay to ensure cleanup is complete
    else:
        root.after(100, start_camera)  # Small delay to ensure cleanup is complete

# --- Face Recognition ---
def perform_face_recognition():
    global cap, wake_word_detected
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame, recognized = face_recognition.recognize_face(frame)
            if recognized:
                lock_system.unlock()
                root.after(5000, lock_system.lock)  # Lock after 5 seconds
            
            # Show the result in a new window
            face_result_window = tk.Toplevel(root)
            face_result_window.title("Face Recognition Result")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            result_label = tk.Label(face_result_window, image=imgtk)
            result_label.imgtk = imgtk
            result_label.configure(image=imgtk)
            result_label.pack()
            
    wake_word_detected = False  # Reset wake word flag

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

# --- Main application ---
if __name__ == "__main__":
    # Set sounddevice defaults
    sd.default.device = 0
    
    # Set up Tkinter UI
    root = tk.Tk()
    root.title("Wake Word Detection with Camera Feed")
    root.geometry("640x580")
    
    # Create main label
    label = tk.Label(root, text="Waiting for wake word...")
    label.pack(pady=10)
    
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
    
    # Set up wake word detection
    set_callback(audio_callback)
    
    # Start audio stream in a separate thread
    Thread(target=start_audio_stream, daemon=True).start()
    
    try:
        root.mainloop()
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        lock_system.cleanup()