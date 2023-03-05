import picamera
from inference.detector import Detector
from configs.configs import modelname

det = Detector(model_name=modelname)


if __name__ == '__main__':
    with picamera.PiCamera() as cam:
        cam.start_preview()
