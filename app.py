import cv2
import mediapipe as mp
import time
from flask import Flask, render_template, Response
import math 
app = Flask(__name__)

# Mediapipe face detection and landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# State variables
last_blink_time = 0
morse_code = ""
current_state = "waiting"
blink_start_time = 0
eye_closed = False
stare_start_time = 0

# Constants
DOT_DURATION = 1      # Blink < 1 sec
DASH_DURATION = 2      # Blink 2-3 sec
BREAK_DURATION = 3     # Staring for more than 3 sec
import math
# Morse Code Dictionary
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', ' ': ' '
}

# Function to convert Morse code to text
def morse_to_text(morse_code):
    morse_code = morse_code.strip()
    words = morse_code.split('   ')  # Split words by 3 spaces
    decoded_message = []

    for word in words:
        letters = word.split(' ')  # Split letters by 1 space
        decoded_word = ''.join([key for key, value in MORSE_CODE_DICT.items() if value == letter] for letter in letters)
        decoded_message.append(decoded_word)

    return ' '.join(decoded_message)


def calculate_ear(eye_landmarks):
    # Vertical distances
    vertical_1 = math.dist([eye_landmarks[1].x, eye_landmarks[1].y], [eye_landmarks[5].x, eye_landmarks[5].y])
    vertical_2 = math.dist([eye_landmarks[2].x, eye_landmarks[2].y], [eye_landmarks[4].x, eye_landmarks[4].y])

    # Horizontal distance
    horizontal = math.dist([eye_landmarks[0].x, eye_landmarks[0].y], [eye_landmarks[3].x, eye_landmarks[3].y])

    # Calculate EAR: (vertical_1 + vertical_2) / (2 * horizontal)
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# Blink detection function
def detect_blinks(frame):
    global last_blink_time, morse_code, current_state, blink_start_time, eye_closed, stare_start_time

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for the left eye
            left_eye_landmarks = [face_landmarks.landmark[i] for i in [133, 173, 144, 145, 153, 154]]
            ear = calculate_ear(left_eye_landmarks)

            if ear < 0.2:  # Eye is considered closed
                if not eye_closed:
                    eye_closed = True
                    blink_start_time = time.time()
                else:
                    if current_state == "waiting":
                        if time.time() - blink_start_time >= 5:
                            current_state = "active"
                            print("Morse code input started!")
                    elif current_state == "active":
                        blink_duration = time.time() - blink_start_time
                        if blink_duration >= DASH_DURATION:
                            morse_code += "-"
                            print("Dash added")
                        elif blink_duration >= DOT_DURATION:
                            morse_code += "."
                            print("Dot added")
            else:  # Eye is open
                if eye_closed:
                    eye_closed = False
                    blink_end_time = time.time()
                    blink_duration = blink_end_time - blink_start_time

                    if current_state == "active":
                        if blink_duration >= DOT_DURATION and blink_duration <= DASH_DURATION:
                            morse_code += "."
                            print("Dot added")
                        elif blink_duration > DASH_DURATION:
                            morse_code += "-"
                            print("Dash added")
                        blink_start_time = 0

                if current_state == "active":
                    if not stare_start_time:
                        stare_start_time = time.time()
                    elif time.time() - stare_start_time >= BREAK_DURATION:
                        morse_code += " "
                        print("Space added")
                        stare_start_time = 0

    return frame


'''def detect_blinks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Use eye landmarks for left eye
            left_eye_landmarks = [face_landmarks.landmark[i] for i in [133, 173, 144, 145, 153, 154]]
            ear = calculate_ear(left_eye_landmarks)
            
            # If EAR is below threshold, eye is considered closed
            if ear < 0.2:
                print("Blink detected")
                # Add Morse code detection logic here
                
    return frame
'''


# Flask route for home page
@app.route('/')
def index():
    return render_template('index.html', morse_code=morse_code)


# Generate video frames for the camera feed
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_blinks(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Route for video stream
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
