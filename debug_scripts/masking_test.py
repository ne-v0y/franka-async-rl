import cv2 as cv
import numpy as np


img = cv.imread('t1.png')
lower = [0, 0, 120]
upper = [120, 90, 255]  # bgr
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

mask = cv.inRange(img, lower, upper)

cv.imshow('mask', mask)

cv.waitKey(0) 
  
#closing all open windows 
cv.destroyAllWindows() 