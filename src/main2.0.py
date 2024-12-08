import cv2
import mediapipe as mp
import pyautogui
import numpy as np

camera_ID = 0
mouse_dpi = 3200

screen_width, screen_height = pyautogui.size()
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.9,
    max_num_hands=1
)


def get_distance(first, second):
    (x1, y1), (x2, y2) = first, second
    return np.hypot(x2 - x1, y2 - y1)


def is_thumb_closed(landmarks_list):
    thumb_index_dist = get_distance(landmarks_list[4], landmarks_list[5])
    index_finger_joint_size = get_distance(landmarks_list[6], landmarks_list[5])
    return thumb_index_dist < index_finger_joint_size * 0.8


def is_index_finger_closed(landmarks_list):
    index_finger_size = get_distance(landmarks_list[6], landmarks_list[8])
    thumb_size = get_distance(landmarks_list[4], landmarks_list[2])
    return thumb_size * 0.4 > index_finger_size


def is_middle_finger_closed(landmarks_list):
    middle_finger_size = get_distance(landmarks_list[10], landmarks_list[12])
    thumb_size = get_distance(landmarks_list[4], landmarks_list[2])
    return thumb_size * 0.4 > middle_finger_size


def move_mouse(landmark):
    if landmark is not None:
        mouse_pos = pyautogui.position()
        x = int((landmark.x - move_mouse.prev_x) * move_mouse.dpi + mouse_pos.x)
        y = int((landmark.y - move_mouse.prev_y) * move_mouse.dpi + mouse_pos.y)
        if move_mouse.prev_x is None or move_mouse.prev_y is None:
            move_mouse.prev_x = x
            move_mouse.prev_y = y
            return
        try:
            pyautogui.moveTo(x, y)
        finally:
            return


move_mouse.dpi = mouse_dpi
move_mouse.prev_x = 0
move_mouse.prev_y = 0


def find_pointer_landmark(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.WRIST]
    return None


def detect_gestures(landmarks_list, processed):
    if len(landmarks_list) >= 21:
        pointer_landmark = find_pointer_landmark(processed)

        # MOVE mouse
        if is_thumb_closed(landmarks_list):
            move_mouse(pointer_landmark)

        # LEFT click
        if (is_index_finger_closed(landmarks_list) and
                not is_middle_finger_closed(landmarks_list)):
            if not detect_gestures.left_click_prev:
                detect_gestures.left_click_prev = True
                pyautogui.mouseDown()

        # RIGHT click
        elif (not is_index_finger_closed(landmarks_list) and
                is_middle_finger_closed(landmarks_list)):
            if not detect_gestures.right_click_prev:
                detect_gestures.right_click_prev = True
                pyautogui.rightClick()

        # DOUBLE click
        elif (is_index_finger_closed(landmarks_list) and
                is_middle_finger_closed(landmarks_list)):
            if not detect_gestures.middle_click_prev:
                detect_gestures.middle_click_prev = True
                pyautogui.doubleClick()

        # RESET states
        elif detect_gestures.left_click_prev:
            detect_gestures.left_click_prev = False
            pyautogui.mouseUp()
        elif detect_gestures.right_click_prev:
            detect_gestures.right_click_prev = False
        elif detect_gestures.middle_click_prev:
            detect_gestures.middle_click_prev = False
        move_mouse.prev_x = pointer_landmark.x
        move_mouse.prev_y = pointer_landmark.y


detect_gestures.left_click_prev = False
detect_gestures.right_click_prev = False
detect_gestures.middle_click_prev = False


def main():
    cap = cv2.VideoCapture(camera_ID)
    draw = mp.solutions.drawing_utils
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmarks_list = []

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                for landmark in hand_landmarks.landmark:
                    landmarks_list.append((landmark.x, landmark.y))

            detect_gestures(landmarks_list, processed)

            window_name = "Feed"
            cv2.imshow(window_name, frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
