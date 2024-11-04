import cv2
import mediapipe.python.solutions.hands as mph
import mediapipe.python.solutions.drawing_utils as draw
import mediapipe.python.solutions.drawing_styles as drawstyle

IP = "192.168.1.11"

hands = mph.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9)

cap = cv2.VideoCapture('https://' + IP + ':8080/video')

while(cap.isOpened()):
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_hands = hands.process(rgb_frame)
    if detected_hands.multi_hand_landmarks:
        for hand_landmarks in detected_hands.multi_hand_landmarks:
            draw.draw_landmarks(
                frame,
                hand_landmarks,
                mph.HAND_CONNECTIONS,
                drawstyle.get_default_hand_landmarks_style(),
                drawstyle.get_default_hand_connections_style())
    else:
        print("No hands detected, skipping frame...")
    try:
        cv2.imshow('virtual_mosuse', cv2.resize(frame, (640, 360)))
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("quitting...")
            break
    except cv2.error:
        print("There was an error in opencv, aborting...")
        break
cap.release()
cv2.destroyAllWindows()