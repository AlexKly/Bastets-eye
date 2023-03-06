import numpy as np
import time, picamera
from inference.detector import Detector
from configs.configs import cam_configs, model_configs

det = Detector(model_name=model_configs['modelname'], confidence=model_configs['confidence_thrs'])

WIDTH, HEIGHT = cam_configs['resolution'][0], cam_configs['resolution'][1]

if __name__ == '__main__':
    with picamera.PiCamera() as cam:
        # Camera initialization and configuration:
        cam.resolution = cam_configs['resolution']
        cam.framerate = cam_configs['fps']
        cam.start_preview()
        time.sleep(cam_configs['time_wait'])
        # Capturing image:
        image = np.empty(shape=(HEIGHT * WIDTH * 3, ), dtype=np.uint8)
        cam.capture(image, 'bgr')
        image = image.reshape(shape=(HEIGHT, WIDTH, 3))
        # Perform detection:
        status = det.detect(image=image)
        print(status)
