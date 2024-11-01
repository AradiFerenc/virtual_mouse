import cv2

IP = "192.168.1.11"

cap = cv2.VideoCapture('https://' + IP + ':8080/video')

while(cap.isOpened()):
    ret, frame = cap.read()
    try:
        cv2.imshow('virtual_mosuse', cv2.resize(frame, (1280, 720)))

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("quitting...")
            break
    except cv2.error:
        print("There was an error in opencv, aborting...")
        break
cap.release()
cv2.destroyAllWindows()