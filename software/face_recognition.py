import cv2
import os
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

class FaceRecognition:
    def __init__(self):
        # Initialize InsightFace
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Buffalo I model loaded successfully!")
        
        # Create directory for saved faces if it doesn't exist
        if not os.path.exists("saved_faces"):
            os.makedirs("saved_faces")
    
    def recognize_face(self, frame):
        saved_face_img = cv2.imread("saved_faces/face_1.png")
        recognized = False  # Initialize the recognition flag
        
        if saved_face_img is None:
            print("Could not load saved face.")
            return frame, recognized

        saved_face = self.app.get(saved_face_img)
        if not saved_face:
            print("No face found in saved image.")
            return frame, recognized

        saved_embedding = saved_face[0]['embedding']

        faces = self.app.get(frame)
        if faces:
            for face in faces:
                embedding = face['embedding']
                dist = cosine(embedding, saved_embedding)
                if dist < 0.3:  # threshold, modify as needed
                    print("Face Recognized!")
                    recognized = True  # Set flag when match found
                    bbox = face['bbox'].astype(int)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    break
                else:
                    print("Face not recognized.")
        return frame, recognized