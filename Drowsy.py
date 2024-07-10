import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Load the pre-trained face cascade 
face_cascade = cv2.CascadeClassifier('C:/Users/kotip/Desktop/project_2/xml/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Unable to load face cascade")

# Load the pre-trained eye cascade
eye_cascade = cv2.CascadeClassifier('C:/Users/kotip/Desktop/project_2/xml/haarcascade_eye.xml')
if eye_cascade.empty():
    raise Exception("Unable to load eye cascade")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif 0.21 <= ratio <= 0.25:
        return 1
    else:
        return 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_frame, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

        if len(eyes) == 2:  # Assuming there are two eyes
            left_eye, right_eye = eyes

            left_blink = blinked(left_eye[0], left_eye[1], left_eye[2], left_eye[3], left_eye[0], left_eye[1])
            right_blink = blinked(right_eye[0], right_eye[1], right_eye[2], right_eye[3], right_eye[0], right_eye[1])

            if left_blink == 0 or right_blink == 0:
                sleep += 1
            drowsy = 0
            active = 0

            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
            elif 1 <= left_blink <= 2 or 1 <= right_blink <= 2:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy !"
                    color = (0, 0, 255)
            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status = "Active :)"
                    color = (0, 255, 0)

    cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
