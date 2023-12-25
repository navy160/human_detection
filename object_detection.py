import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, weights_path="dnn_model/yolov4.weights", cfg_path="dnn_model/yolov4.cfg"):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="dnn_model/classes.txt"):

        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

# Define a new class that inherits from ObjectDetection
class FaceDetection(ObjectDetection):
    def __init__(self, weights_path="dnn_model/opencv_face_detector_uint8.pb", cfg_path="dnn_model/opencv_face_detector.pbtxt"):
        super().__init__(weights_path, cfg_path)

    # Override the detect function to filter out non-face detections
    def detect(self, frame):
        class_ids, confidences, boxes = super().detect(frame)
        face_boxes = []
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            if self.classes[class_id[0]] == "face":
                face_boxes.append(box)
        return np.array(face_boxes)
