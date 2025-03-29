import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
from threading import Thread
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

class FaceRecognition:
    def __init__(self, callback=None):
        # Initialize face recognition model
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Buffalo l model loaded successfully!")
        
        # Camera settings
        self.webcam_resolution = (640, 480)
        self.cap = None
        
        # Callback function to be called after recognition
        self.callback = callback
        
        # Face registration window
        self.face_window = None
    
    def start_camera(self):
        """Start the camera capture"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_resolution[1])
            return True
        return False
    
    def stop_camera(self):
        """Stop the camera capture"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            cv2.destroyAllWindows()
    
    def recognize_face(self):
        """Capture a frame and perform face recognition"""
        if self.cap is None or not self.cap.isOpened():
            print("Camera not available")
            if self.callback:
                self.callback(False)
            return
        
        # Read frame from camera
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image")
            if self.callback:
                self.callback(False)
            return
        
        # Try to load saved face
        saved_face_path = "saved_faces/face_1.png"
        if not os.path.exists(saved_face_path):
            print("No saved face found")
            if self.callback:
                self.callback(False)
            return
        
        saved_face_img = cv2.imread(saved_face_path)
        if saved_face_img is None:
            print("Could not load saved face")
            if self.callback:
                self.callback(False)
            return
        
        # Get face embedding from saved face
        saved_faces = self.app.get(saved_face_img)
        if not saved_faces:
            print("No face found in saved image")
            if self.callback:
                self.callback(False)
            return
        
        saved_embedding = saved_faces[0]['embedding']
        
        # Get face embeddings from current frame
        faces = self.app.get(frame)
        recognized = False
        
        if faces:
            for face in faces:
                embedding = face['embedding']
                dist = cosine(embedding, saved_embedding)
                
                # Draw rectangle around face
                bbox = face['bbox'].astype(int)
                color = (0, 255, 0) if dist < 0.3 else (0, 0, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Add distance text
                text = f"Match: {(1 - dist) * 100:.1f}%"
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if dist < 0.3:  # Threshold for recognition
                    print("Face recognized!")
                    recognized = True
                    break
        
        # Display recognition result
        self.show_recognition_result(frame)
        
        # Call the callback with recognition result
        if self.callback:
            self.callback(recognized)
    
    def show_recognition_result(self, frame):
        """Show the face recognition result in a popup window"""
        result_window = tk.Toplevel()
        result_window.title("Face Recognition Result")
        
        # Convert frame to display in tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Create and display label
        result_label = tk.Label(result_window, image=imgtk)
        result_label.imgtk = imgtk
        result_label.pack()
        
        # Add close button
        close_button = tk.Button(result_window, text="Close", 
                               command=result_window.destroy)
        close_button.pack(pady=10)
    
    def register_face(self, parent, completion_callback=None):
        """Open interface for face registration"""
        # Stop camera if running
        self.stop_camera()
        
        # Hide parent window
        parent.withdraw()
        
        # Create registration window
        self.face_window = tk.Toplevel(parent)
        self.face_window.title("Face Registration")
        
        # Create label for camera feed
        face_label = tk.Label(self.face_window)
        face_label.pack()
        
        # Add buttons
        save_button = tk.Button(self.face_window, text="Save Face", 
                              command=lambda: self.save_face(parent, completion_callback))
        save_button.pack(pady=10)
        
        close_button = tk.Button(self.face_window, text="Close", 
                               command=lambda: self.close_registration(parent, completion_callback))
        close_button.pack(pady=10)
        
        # Start camera in a separate thread
        self.start_camera()
        
        # Update camera feed
        self.update_face_feed(face_label)
    
    def update_face_feed(self, face_label):
        """Update camera feed in registration window"""
        if self.face_window is None or not self.face_window.winfo_exists():
            return
            
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to display in tkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update label
                face_label.imgtk = imgtk
                face_label.configure(image=imgtk)
        
        # Schedule next update
        if self.face_window.winfo_exists():
            self.face_window.after(10, lambda: self.update_face_feed(face_label))
    
    def save_face(self, parent, completion_callback=None):
        """Save the current face image"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Save face image
                face_filename = "saved_faces/face_1.png"
                cv2.imwrite(face_filename, frame)
                print(f"Face saved as {face_filename}")
        
        # Close registration window
        self.close_registration(parent, completion_callback)
    
    def close_registration(self, parent, completion_callback=None):
        """Close the face registration window"""
        # Clean up
        self.stop_camera()
        
        if self.face_window and self.face_window.winfo_exists():
            self.face_window.destroy()
            self.face_window = None
        
        # Show parent window
        parent.deiconify()
        
        # Call completion callback if provided
        if completion_callback:
            completion_callback()
    
    def display_saved_faces(self, parent):
        """Display all saved faces in a window"""
        face_window = tk.Toplevel(parent)
        face_window.title("Saved Faces")
        
        if not os.path.exists("saved_faces") or not os.listdir("saved_faces"):
            tk.Label(face_window, text="No saved faces found").pack(pady=20)
            return
            
        for i, filename in enumerate(os.listdir("saved_faces")):
            face_path = os.path.join("saved_faces", filename)
            img = Image.open(face_path)
            img = img.resize((100, 100))
            imgtk = ImageTk.PhotoImage(img)
            
            frame = tk.Frame(face_window)
            frame.grid(row=i // 5, column=i % 5, padx=5, pady=5)
            
            label = tk.Label(frame, image=imgtk)
            label.image = imgtk
            label.pack()
            
            name_label = tk.Label(frame, text=filename)
            name_label.pack()