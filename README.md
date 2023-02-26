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