import sys
import cv2
import mediapipe.python.solutions.hands as mph
import mediapipe.python.solutions.drawing_utils as draw
import mediapipe.python.solutions.drawing_styles as drawstyle
import pyautogui
from screeninfo import get_monitors


class GestureDetector:
    def __init__(self):
        self.hands = mph.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8)
        self.detected_hands = None
        self.previous_hands = None

    def process(self, rgbf):
        self.previous_hands = self.detected_hands
        self.detected_hands = self.hands.process(rgbf)

    def getDetectedHands(self):
        return self.detected_hands.multi_hand_landmarks

    def drawLandmarksOnImage(self, image):
        if self.detected_hands.multi_hand_landmarks:
            for hand_landmarks in self.detected_hands.multi_hand_landmarks:
                draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    None,
                    drawstyle.get_default_hand_landmarks_style(),
                    drawstyle.get_default_hand_connections_style())
            return True
        else:
            return False

IP = ""
if __name__ == "__main__":
    IP = sys.argv[1]

if IP == "":
    print("No ip address provided, aborting...")
    exit(-1)

# cap = cv2.VideoCapture('https://' + IP + ':8080/video')

cap = cv2.VideoCapture(0)
gestureDetector = GestureDetector()

monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

while cap.isOpened():
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("quitting...")
        break

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gestureDetector.process(rgbf=rgb_frame)
    detected_hands = gestureDetector.getDetectedHands()
    if detected_hands:
        index_finger_tip_normalized = detected_hands[0].landmark[8]
        cursor_x = index_finger_tip_normalized.x * screen_width
        cursor_y = index_finger_tip_normalized.y * screen_height
        pyautogui.moveTo(cursor_x, cursor_y)
    if not gestureDetector.drawLandmarksOnImage(frame):
        print("No hands detected, skipping frame...")
    try:
        cv2.imshow('virtual_mosuse', cv2.resize(frame, (640, 360)))
    except cv2.error:
        print("There was an error in opencv, aborting...")
        break
cap.release()
cv2.destroyAllWindows()
