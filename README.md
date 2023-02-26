# Bastets-eye
Bastet's eye is a cat detection system realized on Rasberry PI 3.

## Train dataset
For this project was chosen the 
[***Kaggle Dog and Cat Detection Dataset***](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection). Here you 
find images and labeled bounded boxes for cat detection. We train YOLO model on this dataset using only cat images 
(total number of images: 3686 files).

## Yolo Dataset Creator
After downloading dataset it needs to prepare for training. **Yolo Dataset Creator** performs sorting image by 
annotation's content (***only cats needed here***) and converting bounded boxes coordinates to *YOLO* format for using 
it in augmentation pipeline.

While `YoloDatasetCreator` was initialized it needs to execute method `create_df()` to prepare train dataset for YOLO.
Here is specified two common parameters: 
- **n_transforms** - number of the augmented images that will be generated
- **test_size** - ratio of the test size from all dataset

*If you want to use own or different dataset you need to rework preprocessing part for exctracting bounded boxes and 
images.*

As result, you need to get following data structure and info dataset **yaml** file:

**Dataset files structure:**
```
bastets_eye_ds
├── train
│   ├── images
│   │   ├── train_image_0.png
│   │   ...
│   │   └── train_image_n.png
│   └── labels
│       ├── train_image_0.txt
│       ...
│       └── train_image_n.txt
│
├── test
│   ├── images
│   │   ├── test_image_0.png
│   │   ...
│   │   └── test_image_n.png
│   └── labels
│       ├── test_image_0.txt
│       ...
│       └── test_image_n.txt
│
└── bastets_eye_ds.yaml
```

**Example for .yaml file:**
```yaml
path: .../bastets_eye_ds  # dataset root dir
train: .../images/train  # train images (relative to 'path')
val: ...images/test  # val images (relative to 'path')
test: ...images/test  # test images (optional)

# Classes (only one class - cat)
names:
  0: cat
```

For more details how to train YOLO using custom dataset you can check 
[here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data?ysclid=lelixivhgj19680677).