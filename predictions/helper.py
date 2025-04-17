import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        with open(data_yaml, mode='r') as f:
            data_yml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yml['names']
        self.nc = data_yml['nc']
        self.INPUT_WH_YOLO = 640

        # Load YOLO ONNX model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        if image is None:
            raise ValueError("Input image is None.")

        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255, (self.INPUT_WH_YOLO, self.INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        detection = preds[0]
        boxes = []
        predicted_texts = []

        for row in detection:
            objectness = row[4]
            if objectness < 0.4:
                continue

            class_scores = row[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            final_confidence = objectness * class_confidence

            if final_confidence > 0.4:
                cx, cy, w_box, h_box = row[0:4]
                x = int((cx - 0.5 * w_box) * w / self.INPUT_WH_YOLO)
                y = int((cy - 0.5 * h_box) * h / self.INPUT_WH_YOLO)
                w_scaled = int(w_box * w / self.INPUT_WH_YOLO)
                h_scaled = int(h_box * h / self.INPUT_WH_YOLO)

                label = self.labels[class_id].lower()
                if 'car' in label:
                    label = 'non-emergency'

                confidence_percent = int(final_confidence * 100)
                predicted_texts.append(f'{label} : {confidence_percent}%')
                boxes.append([x, y, w_scaled, h_scaled])

        return image, predicted_texts, boxes
