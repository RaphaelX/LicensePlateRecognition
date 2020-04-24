from imageai.Detection.Custom import CustomObjectDetection
import numpy as np
import cv2
from image_treatment import *
from read_plate import *
import argparse

input_image = "voiture2.jpg"
output_image = input_image[:-4]+"-detected.jpg"
model_path = "detection_model-ex-040--loss-0001.633.h5"
json_path = "detection_config_2.json"
read_algorithm = '3'

def detect(input_image=input_image, output_image=output_image, model_path=model_path, json_path=json_path):
    #######################################Usage of the model for detection#######################################
    #input_image is the path to the image where the detection is done
    #output_image is the path to the saved image with the detections done 

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path) 
    detector.setJsonPath(json_path)
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image, output_image)
    # for detection in detections:
    #     print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    ##############################################################################################################
    return detections

def plates_detected(input_image=input_image, output_image=output_image, model_path=model_path, json_path=json_path,read='3'):
    detections = detect(input_image, output_image, model_path, json_path)

    boxes = get_boxes(detections)
    crpd_plates = cropped_plates(img_path=input_image, boxes=boxes)
    
    #loop over the cropped images containing the plates
    plates_detected=[]
    for i in range(len(crpd_plates)):
        img = crpd_plates[i]
        if read_algorithm=='1':
            result = read_plate1(img, input_image)
        if read_algorithm=='2':
            result = read_plate2(img, input_image)
        if read_algorithm=='3':
            result = read_plate3(img, input_image)
        plates_detected.append(result)
        # cv2.imwrite('cropped_images/'+result+".jpg", img)    
    return plates_detected

if __name__=="__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("-ii", "--input_image", help="path to input image"  ,default=input_image)                    
    parser.add_argument("-oi", "--output_image", help="path to output image",default=output_image)                    
    parser.add_argument("-m", "--model_path", help="path to model",default=model_path)                    
    parser.add_argument("-j", "--json_path", help="path to json file",default=json_path)                    
    parser.add_argument("-r", "--read_algorithm", help="algorithm to read detected plate",default=read_algorithm)                    

    args = parser.parse_args()
    
    plates = plates_detected(args.input_image, args.output_image, args.model_path, args.json_path, args.read_algorithm)
    print(plates)