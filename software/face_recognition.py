import cv2
import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import json
import time

class FaceRecognition:
    def __init__(self, faces_dir="saved_faces"):
        self.faces_dir = faces_dir
        self.registry_file = os.path.join(faces_dir, "registry.json")
        
        # Create faces directory if it doesn't exist
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            
        # Initialize face analysis model
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Face recognition model loaded successfully")
        
        # Initialize camera
        self.camera = None
        self.resolution = (640, 480)
        
        # Load registered faces
        self.load_registry()
    
    def load_registry(self):
        """Load the registry of saved faces"""
        self.face_registry = []
        
        # Check if registry file exists
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    self.face_registry = json.load(f)
            except Exception as e:
                print(f"Error loading face registry: {e}")
        
        # For backward compatibility, scan the directory for any unregistered faces
        for filename in os.listdir(self.faces_dir):
            if filename.endswith(('.jpg', '.png')) and not filename.startswith('.'):
                face_path = os.path.join(self.faces_dir, filename)
                
                # Check if this face is already in the registry
                if not any(entry['path'] == face_path for entry in self.face_registry):
                    # Extract name from filename (remove extension)
                    name = os.path.splitext(filename)[0]
                    if name.startswith("face_"):
                        name = f"Person {name.split('_')[1]}"
                    
                    # Add to registry
                    self.face_registry.append({
                        'name': name,
                        'path': face_path
                    })
            
        # Save updated registry
        self.save_registry()
        
        print(f"Loaded {len(self.face_registry)} registered faces")
    
    def save_registry(self):
        """Save the face registry to file"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.face_registry, f, indent=2)
        except Exception as e:
            print(f"Error saving face registry: {e}")
    
    def start_camera(self):
        """Robust camera initialization"""
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Explicitly use V4L2 backend
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            # Wait for camera to initialize
            for _ in range(10):  # Attempt 10 frames to warm up
                self.camera.read()
    
    def stop_camera(self):
        """Stop the camera"""
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            self.camera = None
    
    def get_frame(self):
        """Get a valid frame from the camera"""
        if self.camera is None or not self.camera.isOpened():
            self.start_camera()  # Ensure camera is always active
        
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            return frame if ret else None
        return None
    
    def convert_to_rgb(self, frame):
        """Convert BGR frame to RGB"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def register_face(self, frame, name):
        """Register a new face"""
        # Detect faces in the frame
        faces = self.app.get(frame)
        if not faces:
            print("No face detected in the frame")
            return False
        
        # Use only the first face if multiple are detected
        face = faces[0]
        
        # Create a unique filename
        face_id = len(self.face_registry) + 1
        filename = f"{name.replace(' ', '_')}_{face_id}.jpg"
        face_path = os.path.join(self.faces_dir, filename)
        
        # Save the face image
        cv2.imwrite(face_path, frame)
        
        # Add to registry
        self.face_registry.append({
            'name': name,
            'path': face_path
        })
        
        # Save updated registry
        self.save_registry()
        
        print(f"Registered new face: {name}")
        return True
    
    def get_registered_faces(self):
        """Get list of registered faces"""
        return self.face_registry
    
    def recognize_person(self):
        """Recognize a person from camera feed"""
        frame = self.get_frame()
        if frame is None:
            return None, None, False
        
        # If no faces are registered, can't recognize anyone
        if not self.face_registry:
            print("No faces registered in the system")
            return frame, None, False
        
        # Detect face in current frame
        faces = self.app.get(frame)
        if not faces:
            print("No face detected in camera feed")
            return frame, None, False
        
        # Get embedding for detected face
        current_face = faces[0]
        current_embedding = current_face['embedding']
        
        # Draw bounding box around detected face
        bbox = current_face['bbox'].astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        
        # Check against registered faces
        best_match = None
        best_score = 1.0  # Lower is better for cosine distance
        threshold = 0.3  # Threshold for face recognition
        
        for face_entry in self.face_registry:
            # Load the registered face
            face_img = cv2.imread(face_entry['path'])
            if face_img is None:
                continue
                
            # Detect face in the registered image
            saved_faces = self.app.get(face_img)
            if not saved_faces:
                continue
                
            # Get embedding for registered face
            saved_embedding = saved_faces[0]['embedding']
            
            # Calculate similarity
            similarity = cosine(current_embedding, saved_embedding)
            print(f"Similarity with {face_entry['name']}: {similarity:.4f}")
            
            # Update best match
            if similarity < best_score:
                best_score = similarity
                best_match = face_entry
        
        # If a match is found
        if best_match and best_score < threshold:
            # Draw green bounding box for recognized face
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Add name label
            cv2.putText(frame, best_match['name'], (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            print(f"Recognized: {best_match['name']} (score: {best_score:.4f})")
            return frame, best_match['name'], True
        else:
            # Add "Unknown" label
            cv2.putText(frame, "Unknown", (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            print(f"Face not recognized. Best score: {best_score:.4f}")
            return frame, None, False
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_camera()

# Test function
def main():
    face_recognition = FaceRecognition()
    
    # Start camera
    face_recognition.start_camera()
    
    try:
        # Display camera feed with face recognition
        while True:
            frame, name, recognized = face_recognition.recognize_person()
            if frame is not None:
                # Display result
                cv2.imshow("Face Recognition", frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        face_recognition.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()