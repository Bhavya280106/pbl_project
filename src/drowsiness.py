import cv2
import winsound
import time

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

closed_start_time = None
ALERT_DURATION = 2   
COOLDOWN = 3         
last_beep_time = 0

print("Starting Driver Drowsiness Detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(25, 25)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        
        if len(eyes) < 2:
            if closed_start_time is None:
                closed_start_time = time.time()

            elapsed = time.time() - closed_start_time

            if elapsed > ALERT_DURATION:
                cv2.putText(frame, "DROWSY ALERT!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

                
                if time.time() - last_beep_time > COOLDOWN:
                    winsound.Beep(1200, 800)
                    last_beep_time = time.time()
        else:
            closed_start_time = None

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
