import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import threading
import queue
import time

class FaceRecognitionSystem:
    def __init__(self, saved_faces_dir="saved_faces", recognition_threshold=0.3, event_queue=None):
        """
        Initialize the face recognition system
        
        Args:
            saved_faces_dir (str): Directory to store registered face images
            recognition_threshold (float): Threshold for face recognition (lower is stricter)
            event_queue (queue.Queue): Queue for sending events to main thread
        """
        self.saved_faces_dir = saved_faces_dir
        self.recognition_threshold = recognition_threshold
        self.event_queue = event_queue
        self.face_database = {}  # Will store name -> embedding pairs
        
        # Create directory for saved faces if it doesn't exist
        if not os.path.exists(saved_faces_dir):
            os.makedirs(saved_faces_dir)
            
        # Initialize InsightFace
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Face recognition model loaded successfully!")
        
        # Camera settings
        self.camera_resolution = (640, 480)
        self.cap = None
        self.is_running = False
        self.camera_thread = None
        
        # Load saved faces
        self.load_face_database()
        
    def load_face_database(self):
        """Load and process all saved faces from the directory"""
        self.face_database = {}
        
        if not os.path.exists(self.saved_faces_dir):
            return
            
        for filename in os.listdir(self.saved_faces_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                # Extract name from filename (face_name.png)
                name = os.path.splitext(filename)[0]
                if '_' in name:
                    name = name.split('_', 1)[1]  # Remove 'face_' prefix if present
                    
                # Load and process face
                face_path = os.path.join(self.saved_faces_dir, filename)
                try:
                    face_img = cv2.imread(face_path)
                    faces = self.app.get(face_img)
                    
                    if faces:
                        # Store the face embedding
                        self.face_database[name] = faces[0]['embedding']
                        print(f"Loaded face for {name}")
                    else:
                        print(f"No face found in {filename}")
                except Exception as e:
                    print(f"Error loading face {filename}: {e}")
        
        print(f"Loaded {len(self.face_database)} faces into database")
        
    def start_camera(self):
        """Start the camera for face recognition"""
        if self.is_running:
            return
            
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
        
        if not self.cap.isOpened():
            print("Error: Could not open camera!")
            return
            
        self.is_running = True
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
    def stop_camera(self):
        """Stop the camera"""
        self.is_running = False
        
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def camera_loop(self):
        """Main camera processing loop"""
        recognition_frames = 0
        max_recognition_frames = 50  # Try recognition for ~5 seconds at 10fps
        recognized_name = None
        
        while self.is_running and self.cap and self.cap.isOpened():
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame!")
                break
                
            # Process for face recognition
            processed_frame, name = self.recognize_face(frame)
            
            # If face recognized, count it
            if name:
                if recognized_name is None:
                    recognized_name = name
                    print(f"Face recognized: {name}")
                elif recognized_name == name:
                    # Same person recognized consistently
                    recognition_frames += 1
                    
                    # If recognized consistently, trigger success event
                    if recognition_frames >= 3:  # Require 3 consistent recognitions
                        if self.event_queue:
                            self.event_queue.put(("FACE_RECOGNIZED", name))
                        self.is_running = False
                        break
            else:
                recognition_frames = 0
                
            # Check if we've exceeded max frames without consistent recognition
            if max_recognition_frames <= 0:
                if self.event_queue:
                    self.event_queue.put(("FACE_NOT_RECOGNIZED", None))
                self.is_running = False
                break
                
            max_recognition_frames -= 1
            time.sleep(0.1)  # Limit to ~10fps to reduce CPU usage
            
        self.stop_camera()
        
    def recognize_face(self, frame):
        """
        Recognize faces in a frame
        
        Args:
            frame: Camera frame to process
            
        Returns:
            tuple: (processed frame with annotations, recognized name or None)
        """
        if not self.face_database:
            print("No faces in database!")
            return frame, None
            
        # Get faces from frame
        faces = self.app.get(frame)
        
        recognized_name = None
        
        if faces:
            # Process each detected face
            for face in faces:
                embedding = face['embedding']
                bbox = face['bbox'].astype(int)
                
                # Compare with saved faces
                best_match = None
                lowest_distance = 1.0  # Initialize with maximum possible cosine distance
                
                for name, saved_embedding in self.face_database.items():
                    distance = cosine(embedding, saved_embedding)
                    
                    if distance < lowest_distance:
                        lowest_distance = distance
                        best_match = name
                
                # Draw bounding box
                if lowest_distance < self.recognition_threshold:
                    # Face recognized
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{best_match}: {lowest_distance:.2f}", 
                               (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
                    recognized_name = best_match
                else:
                    # Face not recognized
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.putText(frame, f"Unknown: {lowest_distance:.2f}", 
                               (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 0, 255), 2)
        
        return frame, recognized_name
        
    def register_face(self, name):
        """
        Register a new face with the given name
        
        Args:
            name (str): Name for the new face
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
            
        if not self.cap.isOpened():
            print("Error: Could not open camera for registration!")
            return False
            
        # Wait for camera to stabilize
        time.sleep(0.5)
        
        # Capture multiple frames (to get a good one)
        success = False
        for _ in range(10):
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Check for faces
            faces = self.app.get(frame)
            
            if faces:
                # Save the face image
                face_filename = f"{self.saved_faces_dir}/face_{name}.png"
                cv2.imwrite(face_filename, frame)
                
                # Add to database
                self.face_database[name] = faces[0]['embedding']
                print(f"Face registered: {name}")
                success = True
                break
                
            time.sleep(0.1)
            
        # Release camera
        self.cap.release()
        self.cap = None
        
        return success
        
    def register_face_from_frame(self, name, frame):
        """
        Register a new face from an already captured frame
        
        Args:
            name (str): Name for the new face
            frame: Already captured camera frame
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            # Check for faces
            faces = self.app.get(frame)
            
            if faces:
                # Save the face image
                face_filename = f"{self.saved_faces_dir}/face_{name}.png"
                cv2.imwrite(face_filename, frame)
                
                # Add to database
                self.face_database[name] = faces[0]['embedding']
                print(f"Face registered: {name}")
                return True
            else:
                print("No face found in frame")
                return False
        except Exception as e:
            print(f"Error registering face: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_camera()