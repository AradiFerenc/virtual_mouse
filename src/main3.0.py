import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import threading
import queue
import pyautogui
pyautogui.FAILSAFE = False

from OneEuroFilter import OneEuroFilter


model_path = './hand_landmarker.task'
camera_ID = 1
mouse_dpi = 1000

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()


def get_distance(first, second):
    (x1, y1), (x2, y2) = first, second
    return np.hypot(x2 - x1, y2 - y1)


def is_thumb_closed(landmarks_list):
    thumb_index_dist = get_distance(landmarks_list[4], landmarks_list[5])
    index_finger_joint_size = get_distance(landmarks_list[6], landmarks_list[5])
    return thumb_index_dist < index_finger_joint_size * 0.7


def is_index_finger_closed(landmarks_list):
    index_finger_size = get_distance(landmarks_list[6], landmarks_list[8])
    thumb_size = get_distance(landmarks_list[4], landmarks_list[2])
    return thumb_size * 0.4 > index_finger_size


def is_middle_finger_closed(landmarks_list):
    middle_finger_size = get_distance(landmarks_list[10], landmarks_list[12])
    thumb_size = get_distance(landmarks_list[4], landmarks_list[2])
    return thumb_size * 0.4 > middle_finger_size


def move_mouse(landmark):
    global mouse_dpi
    if move_mouse.filters is None:
        move_mouse.filters = []
        move_mouse.filters.append(OneEuroFilter(**move_mouse.filter_config))  # X
        move_mouse.filters.append(OneEuroFilter(**move_mouse.filter_config))  # Y
    if landmark is not None:
        mouse_pos = pyautogui.position()
        if move_mouse.prev_x is None or move_mouse.prev_y is None:
            move_mouse.prev_x = landmark.x
            move_mouse.prev_y = landmark.y
            return
        x = int(move_mouse.filters[0]((landmark.x - move_mouse.prev_x) * mouse_dpi + mouse_pos.x))
        y = int(move_mouse.filters[1]((landmark.y - move_mouse.prev_y) * mouse_dpi + mouse_pos.y))
        try:
            pyautogui.moveTo(x, y)
        finally:
            return


move_mouse.filter_config = {
    'freq': 30,  # Hz
    'mincutoff': 0.04,  # Hz
    'beta': 0.05,
    'dcutoff': 1.0
}
move_mouse.filters = None
move_mouse.prev_x = None
move_mouse.prev_y = None


def find_pointer_landmark(processed):
    if processed.hand_landmarks:
        hand_landmarks = processed.hand_landmarks[0]
        return hand_landmarks[0]
    return None


def detect_gestures(landmarks_list, processed):
    if len(landmarks_list) >= 21:
        pointer_landmark = find_pointer_landmark(processed)

        thumb_closed = is_thumb_closed(landmarks_list)
        index_finger_closed = is_index_finger_closed(landmarks_list)
        middle_finger_closed = is_middle_finger_closed(landmarks_list)

        # MOVE mouse
        if thumb_closed:
            move_mouse(pointer_landmark)

        # LEFT click
        if (index_finger_closed and
                not middle_finger_closed):
            if not detect_gestures.left_click_prev:
                detect_gestures.left_click_prev = True
                try:
                    pyautogui.mouseDown()
                finally:
                    None

        # RIGHT click
        elif (not index_finger_closed and
                middle_finger_closed):
            if not detect_gestures.right_click_prev:
                detect_gestures.right_click_prev = True
                try:
                    pyautogui.rightClick()
                finally:
                    None

        # DOUBLE click
        elif (index_finger_closed and
                middle_finger_closed):
            if not detect_gestures.middle_click_prev:
                detect_gestures.middle_click_prev = True
                try:
                    pyautogui.doubleClick()
                finally:
                    None

        # RESET states
        elif detect_gestures.left_click_prev:
            detect_gestures.left_click_prev = False
            try:
                pyautogui.mouseUp()
            finally:
                None
        elif detect_gestures.right_click_prev:
            detect_gestures.right_click_prev = False
        elif detect_gestures.middle_click_prev:
            detect_gestures.middle_click_prev = False
        move_mouse.prev_x = pointer_landmark.x
        move_mouse.prev_y = pointer_landmark.y


detect_gestures.left_click_prev = False
detect_gestures.right_click_prev = False
detect_gestures.middle_click_prev = False


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS)

    return annotated_image


def detection_callback(result: HandLandmarkerResult, output_image, timestamp_ms: int):
    frame = output_image
    if result.hand_landmarks:
        frame = draw_landmarks_on_image(frame, result)
        landmarks_list = []
        for landmark in result.hand_landmarks[0]:
            landmarks_list.append((landmark.x, landmark.y))
        detect_gestures(landmarks_list, result)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def detection_worker(landmarker, input_queue, output_queue):
    while True:
        task = input_queue.get()
        if task is None:  # Exit signal
            break
        timestamp, image, frame = task
        result = landmarker.detect_for_video(image, timestamp)
        output_queue.put((timestamp, detection_callback(result, frame, timestamp)))
        input_queue.task_done()


def main():
    input_queue = queue.Queue()
    output_queue = queue.PriorityQueue()  # Ensure order based on timestamps
    threads = []
    num_threads = 6  # Number of worker threads for detection
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )
    landmarker = HandLandmarker.create_from_options(options)
    # Start worker threads
    for _ in range(num_threads):
        t = threading.Thread(target=detection_worker, args=(landmarker, input_queue, output_queue))
        t.start()
        threads.append(t)

    cap = cv2.VideoCapture(camera_ID)
    timestamp = 0
    next_timestamp_to_draw = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            input_queue.put((timestamp, image, frame))

            # Check the output queue for results to draw
            while not output_queue.empty():
                ts, processed_frame = output_queue.queue[0]  # Peek at the top
                if ts == next_timestamp_to_draw:
                    output_queue.get()  # Remove from queue
                    window_name = "Feed"
                    # cv2.imshow(window_name, cv2.resize(processed_frame, (200, 150)))
                    cv2.imshow(window_name, processed_frame)
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                    next_timestamp_to_draw += 1
                else:
                    break

            timestamp += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Signal threads to exit
        for _ in range(num_threads):
            input_queue.put(None)
        for t in threads:
            t.join()


if __name__ == '__main__':
    main()