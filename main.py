import cv2
import mediapipe as mp
from collections import deque
import threading
import queue
import time

print("Imports done")

# Settings 
SHAKE_WINDOW = 15
SHAKE_THRESHOLD = 0.15
TONGUE_THRESHOLD = 0.013
MIN_MOUTH_OPEN = 0.05
TRIGGER_COOLDOWN = 60

# Video paths
#VIDEO_TONGUE_SHAKE = r"C:\Users\naifn\Downloads\"
VIDEO_HEAD_SHAKE = r"C:\Users\naifn\Downloads\cat-shaking-head.mp4"
VIDEO_TONGUE = r"C:\Users\naifn\Downloads\rigby-freaky.mp4"
VIDEO_HEAD_TILT = r"C:\Users\naifn\Downloads\orange-cat-hunchback.mp4" 

# Setup MediaPipe
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())

# Variables
nose_positions = deque(maxlen=SHAKE_WINDOW)
cooldown = 0
current_video_path = None
video_queue = queue.Queue(maxsize=1)
play_video_flag = threading.Event()

# Background video reader thread 
def video_reader():
    global current_video_path
    print("Video reader thread started")
    while True:
        play_video_flag.wait()
        if not current_video_path:
            play_video_flag.clear()
            continue

        vid = cv2.VideoCapture(current_video_path)
        if not vid.isOpened():
            print("Could not open video:", current_video_path)
            play_video_flag.clear()
            continue

        print(f"Playing: {current_video_path}")
        while vid.isOpened() and play_video_flag.is_set():
            ret, frame = vid.read()
            if not ret:
                break
            if not video_queue.full():
                video_queue.put(frame)
            time.sleep(0.01)

        vid.release()
        play_video_flag.clear()

# Start the thread
threading.Thread(target=video_reader, daemon=True).start()

# Gesture detection functions 
def detect_tongue(landmarks):
    upper_lip = landmarks[13].y
    lower_lip = landmarks[14].y
    tongue_tip = landmarks[16].y
    mouth_height = lower_lip - upper_lip
    if mouth_height < MIN_MOUTH_OPEN:
        return False
    return (tongue_tip - lower_lip) > TONGUE_THRESHOLD

def detect_head_shake():
    if len(nose_positions) < SHAKE_WINDOW:
        return False
    motion = max(nose_positions) - min(nose_positions)
    return motion > SHAKE_THRESHOLD

def detect_head_tilt(landmarks):
    left_ear = landmarks[234].y
    right_ear = landmarks[454].y
    diff = abs(left_ear - right_ear)
    return diff > 0.05  

# Main Loop
video_played_once = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)
    gesture_detected = False

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0].landmark
        nose_positions.append(face_landmarks[1].x)

        tongue_out = detect_tongue(face_landmarks)
        head_shake = detect_head_shake()
        head_tilt = detect_head_tilt(face_landmarks)

        #  Debug info on screen
        cv2.putText(frame, f"Tongue: {tongue_out}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if tongue_out else (0,0,255), 2)
        cv2.putText(frame, f"Shake: {head_shake}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if head_shake else (0,0,255), 2)
        cv2.putText(frame, f"Tilt: {head_tilt}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if head_tilt else (0,0,255), 2)

        # Gesture combinations 
        #if tongue_out and head_shake and cooldown == 0:
            #current_video_path = VIDEO_TONGUE_SHAKE
            #print("Trigger: tongue + shake")
            #play_video_flag.set()
            #cooldown = TRIGGER_COOLDOWN
        if tongue_out and cooldown == 0:
            current_video_path = VIDEO_TONGUE
            print("Trigger: tongue")
            play_video_flag.set()
            cooldown = TRIGGER_COOLDOWN
        elif head_shake and cooldown == 0:
            current_video_path = VIDEO_HEAD_SHAKE
            print("Trigger: shake")
            play_video_flag.set()
            cooldown = TRIGGER_COOLDOWN
        elif head_tilt and cooldown == 0:
            current_video_path = VIDEO_HEAD_TILT
            print("Trigger: head tilt")
            play_video_flag.set()
            cooldown = TRIGGER_COOLDOWN

    # Cooldown counter
    if cooldown > 0:
        cooldown -= 1

    # Show webcam
    cv2.imshow("Facial Gesture Detection", frame)

    # Show video if available
    if not video_queue.empty():
        video_frame = video_queue.get()
        cv2.imshow("Video Playback", video_frame)
        video_played_once = True
    elif video_played_once and not play_video_flag.is_set():
        cv2.destroyWindow("Video Playback")
        video_played_once = False

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
