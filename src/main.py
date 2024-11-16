import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def get_landmark_coordinates(landmark, frame_width, frame_height):
    return int(landmark.x * frame_width), int(landmark.y * frame_height)

def detect_click_gestures(landmarks):
    fingers = {
        "index": landmarks[8].y < landmarks[6].y,
        "middle": landmarks[12].y < landmarks[10].y,
    }
    left_click = not fingers["index"] and fingers["middle"]
    right_click = fingers["index"] and not fingers["middle"]
    return left_click, right_click


cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x, y = get_landmark_coordinates(hand_landmarks.landmark[9], frame_width, frame_height)

            screen_x = np.interp(x, (0, frame_width), (0, screen_width))
            screen_y = np.interp(y, (0, frame_height), (0, screen_height))
            pyautogui.moveTo(screen_x, screen_y)

            left_click, right_click = detect_click_gestures(hand_landmarks.landmark)
            if left_click:
                pyautogui.click(button='left')
            elif right_click:
                pyautogui.click(button='right')

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
