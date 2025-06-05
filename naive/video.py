import cv2 as cv

from naive.processor import process_frame

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = process_frame(frame)[0]
    cv.imshow('Light Bars Detection', result)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
