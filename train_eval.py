#trains a YOLOv3 detection model
#data_ directory is the directory where our Pascal VOC dataset is saved ("plates" in this case)
#object_names_array is the list of objects we are detecting, i.e, the labels in the dataset
#num_experiments is the number of epochs we do
#batch_size is the batch size used
#train_from_pretrained_model is the path to model from which we start the training

from imageai.Detection.Custom import DetectionModelTrainer

def trainer(data_directory="plates", object_names_array=["LP"], batch_size=4, num_experiments=2, train_from_pretrained_model="plates/models/detection_model-ex-002--loss-0007.222.h5"):
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory)
    trainer.setTrainConfig(object_names_array, batch_size, num_experiments, train_from_pretrained_model)
    trainer.trainModel()

#Evaluates the models we have
#model_path is the path to where the models were saved
#json_path is the path to the detection_config.json file generated in the previous cell. It contains the anchors calculated
def eval(data_directory="plates",model_path="plates/models", json_path="plates/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5):
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory)
    trainer.evaluateModel(model_path, json_path, iou_threshold, object_threshold, nms_threshold)
