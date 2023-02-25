from camera import CameraWrapper

cam_wrapper = CameraWrapper()


if __name__ == '__main__':
    while True:
        cam_wrapper.cam.capture(cam_wrapper.image_buf)
        print(cam_wrapper.image_buf)
        print('Image captured')
