import cv2
import os
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model

class FatigueDetection:
    def __init__(self, path):
        self.path = path
        self.frames = []
        self.faces = []
        self.eyes_image = []
        self.mouth_image = []
        self.detector = MTCNN()
        self.model = load_model('./fatigue_model.h5')  # Load pre-trained model

    def generate_frame(self):
        cap = cv2.VideoCapture(self.path)
        self.frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Optional: Enhance the frame for better face detection in poor lighting conditions
                frame = self.adjust_gamma(frame, gamma=1.5)
                self.frames.append(frame)
                count += 10  # Skip 10 frames to speed up processing
                cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            else:
                cap.release()
                break

    # Optional: Preprocessing for poor lighting
    def adjust_gamma(self, image, gamma=1.5):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def extract_faces(self):
        for img in self.frames:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            if len(faces) > 0:
                for i, (x, y, w, h) in enumerate(faces):
                    self.faces.append(img[y:y+h, x:x+w])
                    if i > 1:
                        break
            else:
                print("No face detected in this frame. Skipping...")

    def eyeRegionExtraction(self):
        for face in self.faces:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            feature = self.detector.detect_faces(gray)
            if len(feature) > 0:  # Check if MTCNN has detected faces
                keys = feature[0]['keypoints']
                left_eye = keys['left_eye']
                right_eye = keys['right_eye']
                s = int(left_eye[1] // 2)
                a = int(right_eye[1] // 3)
                self.eyes_image.append(face[left_eye[0]-s:right_eye[1]+a, left_eye[1]-s:right_eye[0]+a])
            else:
                print("No facial keypoints detected. Skipping eye region extraction.")

    def mouthRegionExtraction(self):
        for face in self.faces:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            feature = self.detector.detect_faces(gray)
            if len(feature) > 0:  # Check if MTCNN has detected faces
                keys = feature[0]['keypoints']
                left_mouth = keys['mouth_left']
                right_mouth = keys['mouth_right']
                x1, y1 = left_mouth
                x2, y2 = right_mouth
                d = int(np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2)))
                h = d // 2
                self.mouth_image.append(face[y1-h:y2+h, x1:x2])
            else:
                print("No facial keypoints detected. Skipping mouth region extraction.")

    def predict_status(self):
        pre = []
        images = self.eyes_image + self.mouth_image
        for img in images:
            new_img = cv2.resize(img, (175, 175))
            x = (new_img / 255).reshape(1, 175, 175, 3)
            pre.append(np.argmax(self.model.predict(x)))
        self.count = {'eye_open': pre.count(0), 'eye_close': pre.count(1), 'mouth_open': pre.count(2), 'mouth_close': pre.count(3)}

    def PERCLOS(self):
        total_eye_frames = self.count['eye_open'] + self.count['eye_close']
        if total_eye_frames == 0:  # No eyes detected in any frames
            return 0  # Return 0 to indicate no eye closure detected
        return self.count['eye_close'] / total_eye_frames

    def POM(self):
        total_mouth_frames = self.count['mouth_open'] + self.count['mouth_close']
        if total_mouth_frames == 0:  # No mouth detected in any frames
            return 0  # Return 0 to indicate no mouth opening detected
        return self.count['mouth_open'] / total_mouth_frames

    def calculate(self):
        self.generate_frame()
        self.extract_faces()
        self.eyeRegionExtraction()
        self.mouthRegionExtraction()
        self.predict_status()
        return (self.PERCLOS() > 0.5) or (self.POM() > 0.5)
