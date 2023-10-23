import cv2

harcascade = "model/haarcascade_frontalface_default.xml"

cap = cv2.VideoCapture(0)
cap.set(3, 640) # Width
cap.set(4, 480) # Height

while True:
    status, frame = cap.read()

    facecascade = cv2.CascadeClassifier(harcascade)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = facecascade.detectMultiScale(frame_gray, 1.1, 4)

    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


