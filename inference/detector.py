import numpy as np
import torch, typing
from configs.attrs import CAT_LABEL, CAT_DETECTED, CAT_NOT_DETECTED


class Detector:
    """ The Detector class perform detection cats on the frame objects (label 15). It is a wrapped YOLO v5 model to
    class for comfortable in use for Raspeberry PI. """
    def __init__(self, model_name: str):
        """ Initialization class object.

        :param model_name: YOLOv5 model type (yolov5n | yolov5s | yolov5m | yolov5l | yolov5x).
        :return:
        """
        # YOLOv5 settings:
        self.model = torch.hub.load(repo_or_dir='ultralytics/yolov5', model=model_name)
        self.cat_label = CAT_LABEL

    def detect(self, image: np.ndarray, thr_conf: float = 0.3) -> typing.Tuple[tuple, bool]:
        """ Detect cat on frame.

        :param image: Input frame pixels.
        :param thr_conf: Confidence detection threshold.
        :return: Labeled cat detection status (True | False) and detected bounded box coordinates.
        """
        preds = self.model(image).xyxy[0].numpy()
        for pred in preds:
            if pred[5] == self.cat_label and pred[4] >= thr_conf:
                return (pred[0], pred[1], pred[2], pred[3]), CAT_DETECTED
        return (), CAT_NOT_DETECTED
