from imageai.Detection.Custom import CustomObjectDetection
import numpy as np
import cv2
from image_treatment import *

#######################################Usage of the model for detection#######################################
#input_image is the path to the image where the detection is done
#output_image_path is the path to the saved image with the detections done 
input_image="voiture2.jpg"
output_image_path="voiture2-detected.jpg"

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("drive/My Drive/Colab Notebooks/plates/models/detection_model-ex-001--loss-0005.291.h5") 
detector.setJsonPath("drive/My Drive/Colab Notebooks/plates/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image, output_image_path)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
##############################################################################################################

#INTEGRATION FROM DETECTION TO READING
#get the boxes from the detection
boxes = get_boxes(detections)
crpd_plates = cropped_plates(img_path=input_image, boxes=boxes)
if not os.path.exists('cropped_images'):
    os.makedirs('cropped_images')


#loop over the cropped images containing the plates
for i in range(len(crpd_plates)):
    img = crpd_plates[i]
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    _, output_path, img = get_string('shot2.jpg', os.getcwd(),img) 

    img1 = noise_removal(img)
    # cv2.imwrite('socorro1.jpg', img1)
    img2 = binarization_gaussian(img1)
    # cv2.imwrite('socorro2.jpg', img2)

    result = save_result(img2,output_path, "plate_numbers")
    if result=='':
        img1 = 255-img1
        img2 = binarization_gaussian(img1)
    # cv2.imwrite('socorr.jpg', img2)

    result = save_result(img1,output_path, "plate_numbers")