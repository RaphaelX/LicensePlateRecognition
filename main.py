from imageai.Detection.Custom import CustomObjectDetection
import numpy as np
import cv2
from image_treatment import *
from read_plate import *
import argparse

input_image = "voiture2.jpg"
output_image = "voiture2-detected.jpg"
model_path = "plates/models/detection_model-ex-001--loss-0005.291.h5"
json_path = "drive/My Drive/Colab Notebooks/plates/json/detection_config.json"
read_algorithm = 2

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
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    ##############################################################################################################
    return detections


if __name__=="__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("-ii", "--input_image", help="path to input image"  ,default=input_image)                    
    parser.add_argument("-oi", "--output_image", help="path to output image",default=output_image)                    
    parser.add_argument("-m", "--model_path", help="path to model",default=model_path)                    
    parser.add_argument("-j", "--json_path", help="path to json file",default=json_path)                    
    parser.add_argument("-r", "--read_algorithm", help="algorithm to read detected plate",default=read_algorithm)                    

    args = parser.parse_args()

    # detections = detect(args.input_image, args.output_image, args.model_path, args.json_path)

    # boxes = get_boxes(detections)
    # crpd_plates = cropped_plates(img_path=input_image, boxes=boxes)
    if not os.path.exists('cropped_images'):
        os.makedirs('cropped_images')

    crpd_plates = [cv2.imread(args.input_image)]

    #loop over the cropped images containing the plates
    for i in range(len(crpd_plates)):
        img = crpd_plates[i]
        if args.read_algorithm=='1':
            result = read_plate1(img, args.input_image)
        if args.read_algorithm=='2':
            result = read_plate2(img, args.input_image)
        if args.read_algorithm=='3':
            result = read_plate3(img, args.input_image)
        print(result)
        cv2.imwrite('cropped_images/'+result+".jpg", img)