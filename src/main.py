import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def get_landmark_coordinates(landmark, frame_width, frame_height):
    return int(landmark.x * frame_width), int(landmark.y * frame_height)

def is_point_inside_rect(point, left, top, right, bottom):
    px, py = point
    return left <= px <= right and top <= py <= bottom

index_inside = False
middle_inside = False

screen_width, screen_height = pyautogui.size()

click_display_n_frames = 5
right_click_draw_counter = 0
left_click_draw_counter = 0

cap = cv2.VideoCapture(0)
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

            # Get coordinates of landmarks 0, 2, 5, 17 --> to define the rectangle for the click events
            landmark_coords = [
                get_landmark_coordinates(hand_landmarks.landmark[i], frame_width, frame_height)
                for i in [0, 2, 5, 17]
            ]
            xs, ys = zip(*landmark_coords)

            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)

            index_tip = get_landmark_coordinates(hand_landmarks.landmark[8], frame_width, frame_height)
            middle_tip = get_landmark_coordinates(hand_landmarks.landmark[12], frame_width, frame_height)

            if is_point_inside_rect(index_tip, left, top, right, bottom):
                if not index_inside:
                    pyautogui.click(button='left')
                    index_inside = True
                    left_click_draw_counter = 1
            else:
                index_inside = False

            if is_point_inside_rect(middle_tip, left, top, right, bottom):
                if not middle_inside:
                    pyautogui.click(button='right')
                    middle_inside = True
                    right_click_draw_counter = 1
            else:
                middle_inside = False

            palm_base = get_landmark_coordinates(hand_landmarks.landmark[9], frame_width, frame_height)
            screen_x = np.interp(palm_base[0], (0, frame_width), (0, screen_width))
            screen_y = np.interp(palm_base[1], (0, frame_height), (0, screen_height))
            pyautogui.moveTo(screen_x, screen_y)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    if (left_click_draw_counter % click_display_n_frames != 0):
        left_click_draw_counter += 1
        cv2.putText(frame, "Left Click", (00, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA, False)
    if(right_click_draw_counter % click_display_n_frames != 0):
        right_click_draw_counter+=1
        cv2.putText(frame, "Right Click", (00, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA, False)
    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
