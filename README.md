# Bastet's Eye
The Bastet's Eye is a cat detection system realized on Rasberry PI 3.

## Hardware

## Run PiCamera

## Detection
For cat detection and implementation to Rasberry PI is used pretrained 
[YOLOv5 model](https://github.com/ultralytics/yolov5?ysclid=lex1cjrudj411444774) in the Bastet's eye project. It needs 
to specify detected cat label as 15 because YOLOv5 was pretrained on COCO dataset with specified label `cat` is 15:
```Python
CAT_LABEL = 15  # Check here --> https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml
```

Here's provided simple example how to run YOLOv5 for detection (inference):
```Python
import torch

CAT_LABEL = 15  # Label cat
CONFIDENCE_THRESHOLD = 0.5  # Minimal probability of detection

model = torch.hub.load(repo_or_dir='ultralytics/yolov5', model='yolov5n')   # You can specify another yolo model
preds = model(image).xyxy[0].numpy()
# In preds is stored detected objects:
for pred in preds:
    # Pred struct: bbox_x, bbox_x, bbox_w, bbox_h, confidence, label:
    if pred[5] == CAT_LABEL and pred[4] >= CONFIDENCE_THRESHOLD:
        print('Cat is detected')
```