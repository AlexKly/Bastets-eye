from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn import model_selection
import os, cv2, yaml, shutil, albumentations


class YoloDatasetCreator:
    def __init__(self, dir_ds, dir_yolo_ds):
        # Images and annotations dataset:
        self.dir_ds = dir_ds
        self.dir_anns = dir_ds/'annotations'
        # Directory for temporary files:
        self.dir_tmp = dir_ds.parent/'tmp'
        if os.path.exists(path=self.dir_tmp):
            shutil.rmtree(path=self.dir_tmp, ignore_errors=True)
        if not os.path.exists(path=self.dir_tmp):
            os.mkdir(path=self.dir_tmp)
        # YOLO dataset:
        self.dir_yolo_ds = dir_yolo_ds
        if os.path.exists(path=self.dir_yolo_ds):
            shutil.rmtree(path=dir_yolo_ds, ignore_errors=True)
        if not os.path.exists(path=self.dir_yolo_ds):
            os.mkdir(path=self.dir_yolo_ds)
        # Data augmentation:
        self.transform = albumentations.Compose(
            [
                albumentations.RandomFog(p=0.8),
                albumentations.RandomBrightnessContrast(contrast_limit=0.5, p=0.5),
                albumentations.BBoxSafeRandomCrop(p=1),
                albumentations.Rotate(p=1),
                albumentations.RGBShift(p=0.7),
                albumentations.RandomSnow(p=0.9),
                albumentations.VerticalFlip(p=1),
            ], bbox_params=albumentations.BboxParams(format='yolo')
        )

    def load_file_paths(self):
        return [f for f in os.listdir(self.dir_anns) if os.path.isfile(os.path.join(self.dir_anns, f))]

    @staticmethod
    def read_xml(path):
        tree = ET.parse(source=path)
        root = tree.getroot()
        if root[4][0].text == 'cat':
            folder = root[0].text
            filename = root[1].text
            # size: (width, height, depth)
            size = (int(root[2][0].text), int(root[2][1].text), int(root[2][2].text))
            # bbox: (xmin, ymin, xmax, ymax)
            bbox = (int(root[4][5][0].text), int(root[4][5][1].text), int(root[4][5][2].text), int(root[4][5][3].text))
            return folder, filename, size, bbox
        else:
            return None

    @staticmethod
    def convert_pascal2yolo(bbox, size):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (bbox[0] + bbox[2]) / 2.0
        y = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        return x * dw, y * dh, w * dw, h * dh

    def save_files(self, fname, img, bbox):
        # Save image:
        cv2.imwrite(f'{self.dir_tmp}/{fname}.png', img)
        # Save annotations:
        bbox_str = f'{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}'
        with (self.dir_tmp/f'{fname}.txt').open('w') as writer:
            writer.write(bbox_str)

    def form_yolo_ds(self, samples, test_size):
        # Init Yolo dataset files and directories:
        dir_imgs, dir_lbls = self.dir_yolo_ds/'images', self.dir_yolo_ds/'labels'
        dir_train_imgs, dir_train_lbls = dir_imgs/'train', dir_lbls/'train'
        dir_test_imgs, dir_test_lbls = dir_imgs/'test', dir_lbls/'test'
        path_ds_info = self.dir_yolo_ds/f'{self.dir_yolo_ds.parts[-1]}.yaml'
        dirs = [dir_imgs, dir_lbls, dir_train_imgs, dir_train_lbls, dir_test_imgs, dir_test_lbls]
        for d in dirs:
            if not os.path.exists(path=d):
                os.mkdir(path=d)
        # Copy files:
        train_samples, test_samples = model_selection.train_test_split(samples, test_size=test_size, shuffle=True)
        for train_sample in tqdm(train_samples, desc='Copying train samples to YOLO Dataset directory -->'):
            shutil.copy2(src=self.dir_tmp/f'{train_sample}.png', dst=dir_train_imgs)    # Image -->
            shutil.copy2(src=self.dir_tmp/f'{train_sample}.txt', dst=dir_train_lbls)    # Label -->
        for test_sample in tqdm(test_samples, desc='Copying test samples to YOLO Dataset directory -->'):
            shutil.copy2(src=self.dir_tmp/f'{test_sample}.png', dst=dir_test_imgs)  # Image -->
            shutil.copy2(src=self.dir_tmp/f'{test_sample}.txt', dst=dir_test_lbls)  # Label -->
        # Create yaml file:
        yaml_content = {
            'path': f'{self.dir_yolo_ds}',
            'train': 'images/train',
            'val': 'images/test',
            'test': 'images/test',
            'names': {0: 'cat'}
        }
        with path_ds_info.open('w') as yaml_file:
            yaml.dump(yaml_content, yaml_file, default_flow_style=False, sort_keys=False)

    def create_df(self, n_transforms, test_size):
        names = list()
        p_anns = self.load_file_paths()
        for p in tqdm(p_anns, desc='Loading images and bounding boxes -->'):
            parsed = self.read_xml(path=self.dir_anns/p)
            if parsed is not None:
                path_img = self.dir_ds/parsed[0]/parsed[1]
                fname = parsed[1].split('.')[0]
                img = cv2.imread(str(path_img))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                bbox_yolo = [list(self.convert_pascal2yolo(bbox=parsed[3], size=parsed[2])) + [None]]
                # Save original image and annotation:
                names += [fname]
                self.save_files(fname=fname, img=img, bbox=bbox_yolo[0])
                for i in range(n_transforms):
                    transformed = self.transform(image=img, bboxes=bbox_yolo)
                    # Save augmented image and annotation:
                    if len(transformed['bboxes']) > 0:
                        names += [f"{fname}_{i}"]
                        self.save_files(fname=f'{fname}_{i}', img=transformed['image'], bbox=transformed['bboxes'][0])
        self.form_yolo_ds(samples=names, test_size=test_size)
        # Remove temporary files and directories:
        shutil.rmtree(path=self.dir_tmp, ignore_errors=True)


if __name__ == '__main__':
    cur_dir = Path().resolve().parent
    dir_ds = cur_dir/'data/dog_and_cat_detection_dataset'
    dir_yolo_ds = cur_dir/'data/bastets_eye_ds'
    ydc = YoloDatasetCreator(dir_ds=dir_ds, dir_yolo_ds=dir_yolo_ds)
    ydc.create_df(n_transforms=5, test_size=0.2)
