import busio
from adafruit_ov7670 import OV7670
from adafruit_blinka.board.raspberrypi import raspi_40pin

GPIO2 = raspi_40pin.D2      # SCL
GPIO3 = raspi_40pin.D3      # VS
GPIO4 = raspi_40pin.D4      # PLK
GPIO7 = raspi_40pin.D7      # PWDN
GPIO8 = raspi_40pin.D8      # D0
GPIO9 = raspi_40pin.D9      # RET
GPIO10 = raspi_40pin.D10    # D1
GPIO14 = raspi_40pin.D14    # SDA
GPIO15 = raspi_40pin.D15    # HS
GPIO17 = raspi_40pin.D17    # D7
GPIO18 = raspi_40pin.D18    # XLK
GPIO22 = raspi_40pin.D22    # D3
GPIO23 = raspi_40pin.D23    # D6
GPIO24 = raspi_40pin.D24    # D4
GPIO25 = raspi_40pin.D25    # D2
GPIO27 = raspi_40pin.D27    # D5


class CameraWrapper:
    def __init__(self, cam_size=(80, 60)):
        # Camera initialization and configuration:
        self.cam_bus = busio.I2C(scl=GPIO2, sda=GPIO14)    # Camera configuration interface (I2C)
        self.cam = OV7670(
            i2c_bus=self.cam_bus,
            data_pins=[GPIO8, GPIO10, GPIO25, GPIO22, GPIO24, GPIO27, GPIO23, GPIO17],
            clock=GPIO4,
            vsync=GPIO3,
            href=GPIO15,
            mclk=GPIO18,
            shutdown=GPIO7,
            reset=GPIO9,
        )
        self.cam.size = cam_size
        # Temporary:
        self.image_buf = bytearray(2 * self.cam.width * self.cam.height)
