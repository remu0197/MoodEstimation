import cv2
import time

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

start = time.time()
path = '../data/test/test.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    ret, frame = cap2.read()
    cv2.imshow('frame2', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

    out.write(frame)

cap.release()
cv2.destroyAllWindows()