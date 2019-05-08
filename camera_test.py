import sys
import cv2
import numpy as np

print("OpenCV Version: {}".format(cv2.__version__))

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame', gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release(0)
cv2.destroyAllWindows()