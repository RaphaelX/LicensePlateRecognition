import cv2
import os
import pytesseract
import numpy as np


def get_boxes(detections):
    '''get the boxes after detection
    '''
    boxes = []
    for detection in detections:
        boxes.append(detection["box_points"])
    boxes = np.array(boxes)
        
    return boxes

def cropped_plates(img_path, boxes, aug0=5,aug1=5,aug2=5,aug3=5):
    #increase boxes to deal with prediction's imprecision
    aug_boxes = np.array([-int(aug0),-int(aug1),int(aug2),int(aug3)])
    boxes += aug_boxes

    cropped_images = []
    img = cv2.imread(img_path) 
    for box in boxes:
        cropped = img[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
    cropped_images.append(cropped)

    return cropped_images

# ipath=os.getcwd+"\\0.jpg"
def get_string(img_path, output_dir,img):
    # Extract the file name without the file extension
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    # Create a directory for outputs
    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    return file_name, output_path, img

# img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

def noise_removal(img):
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Apply blur to smooth out the edges
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img

def binarization(img):
    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return img
def binarization_gaussian(img):
    img = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11,2)
    return img

def result(img, output_path, file_name):
    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang="eng")

    # Save the filtered image in the output directory
    save_path = os.path.join(output_path, result+".jpg")
    cv2.imwrite(save_path, img)

    return result
