import cv2
import imutils
import numpy as np
from PIL import Image
from image_treatment import *
from readKNN import readKNN


def read_plate1(img, img_name):
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    _, output_path, img = get_string(img_name, os.getcwd(),img) 

    img1 = noise_removal(img)
    # cv2.imwrite('socorro1.jpg', img1)
    img2 = binarization_gaussian(img1)
    # cv2.imwrite('socorro2.jpg', img2)

    text = result(img2,output_path, "plate_numbers")
    
    if text=='':
        img1 = 255-img1
        img2 = binarization_gaussian(img1)
        # cv2.imwrite('socorr.jpg', img2)
        text = result(img1,output_path, "plate_numbers")
    
    return text   

def read_plate2(img, img_name):
    img = imutils.resize(img, width = 500)
    #pass image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    #find edges
    edged = cv2.Canny(gray, 30, 200)
    #find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #sorts contours based on minimum area 30 and ignores the ones below that
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    screenCnt = None #will store the number plate contour
    img2 = img.copy()
    cv2.drawContours(img2,cnts,-1,(0,255,0),3) 
    
    idx=7
    # loop over contours
    for c in cnts:
    # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4: #chooses contours with 4 corners
            screenCnt = approx
            x,y,w,h = cv2.boundingRect(c) #finds co-ordinates of the plate
            new_img=img[y:y+h,x:x+w]
            idx+=1
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
    
    Cropped_loc=img_name #the filename of cropped image######################
    text=pytesseract.image_to_string(Cropped_loc,lang='eng') #converts image characters to string
    return text

def read_plate3(img, img_name):
    return readKNN(img_name)