import cv2
import mediapipe as mp
from time import time
import numpy as np
from collections import deque

class MorseCodeBlinkDetector:
    def __init__(self):
        # Mediapipe Face Mesh initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils

        # Eye landmark indices in Mediapipe's 468 face landmarks
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]

        # Blink detection parameters
        self.blink_threshold = 0.25  # EAR threshold for detecting eye closure
        self.blink_start_time = None
        self.morse_code = []
        self.ear_history = deque(maxlen=60)  # To calculate blink duration
        self.blink_detected = False
        self.current_symbol = ""

    def eye_aspect_ratio(self, eye_landmarks):
        # Calculate the EAR (Eye Aspect Ratio)
        A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_blinks(self, frame):
        # Convert the image to RGB as required by Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye landmarks for both eyes
                left_eye_landmarks = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                                       int(face_landmarks.landmark[i].y * frame.shape[0])) for i in self.left_eye_idx]
                right_eye_landmarks = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                                        int(face_landmarks.landmark[i].y * frame.shape[0])) for i in self.right_eye_idx]

                # Calculate the Eye Aspect Ratio (EAR)
                left_ear = self.eye_aspect_ratio(left_eye_landmarks)
                right_ear = self.eye_aspect_ratio(right_eye_landmarks)
                ear = (left_ear + right_ear) / 2.0

                # Draw the eye contours
                cv2.polylines(frame, [np.array(left_eye_landmarks, dtype=np.int32)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(right_eye_landmarks, dtype=np.int32)], True, (0, 255, 0), 1)

                self.ear_history.append(ear)

                # Blink detection logic
                if ear < self.blink_threshold and not self.blink_detected:
                    self.blink_detected = True
                    self.blink_start_time = time()  # Start the blink timer
                elif ear >= self.blink_threshold and self.blink_detected:
                    blink_duration = time() - self.blink_start_time  # Calculate blink duration

                    if blink_duration < 1.0:
                        self.morse_code.append(".")
                        self.current_symbol = "."
                        print("Detected short blink: .")
                    elif blink_duration > 3.0:
                        self.morse_code.append("-")
                        self.current_symbol = "-"
                        print("Detected long blink: -")

                    self.blink_detected = False
                    self.blink_start_time = None

        return frame

    def render_morse_code(self, frame):
        # Render the morse code dynamically on the frame
        morse_text = ''.join(self.morse_code)
        cv2.putText(frame, f"Morse Code: {morse_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    detector = MorseCodeBlinkDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect blinks and get the updated frame
        frame = detector.detect_blinks(frame)

        # Render Morse Code on the video feed
        frame = detector.render_morse_code(frame)

        # Display the video feed with the detected blinks and morse code
        cv2.imshow("Blink to Morse Code", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
