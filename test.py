import cv2
img = cv2.imread("TEST14.jpg")
crp =  img[138:166, 186:306]
cv2.imshow("", crp)
cv2.waitKey(1000)