import busio, adafruit_ov5640
from adafruit_blinka.board.raspberrypi import raspi_40pin

GPIO2 = raspi_40pin.D2      # CMOS_SCL
GPIO3 = raspi_40pin.D3      # CMOS_PCLK
GPIO4 = raspi_40pin.D4      # CMOS_D5
GPIO7 = raspi_40pin.D7      # CMOS_PWDN
GPIO8 = raspi_40pin.D8      # CMOS_D3
GPIO9 = raspi_40pin.D9      # CMOS_RESET
GPIO10 = raspi_40pin.D10    # CMOS_D7
GPIO14 = raspi_40pin.D14    # CMOS_SDA
GPIO15 = raspi_40pin.D15    # CMOS_VSYNC
GPIO17 = raspi_40pin.D17    # CMOS_D9
GPIO18 = raspi_40pin.D18    # CMOS_D4
GPIO22 = raspi_40pin.D22    # CMOS_D2
GPIO23 = raspi_40pin.D23    # CMOS_D8
GPIO24 = raspi_40pin.D24    # CMOS_HREF
GPIO25 = raspi_40pin.D25    # CMOS_D6
GPIO27 = raspi_40pin.D27    # CMOS_XCLK


class CameraWrapper:
    def __init__(self, cam_size=(80, 60)):
        # Camera initialization and configuration:
        self.cam_bus = busio.I2C(scl=GPIO2, sda=GPIO14)    # Camera configuration interface (I2C)
        self.cam = adafruit_ov5640.OV5640(
            i2c_bus=self.cam_bus,
            data_pins=[GPIO22, GPIO8, GPIO18, GPIO4, GPIO25, GPIO10, GPIO23, GPIO17], # D2, D3, D4, D5, D6, D7, D8, D9
            clock=GPIO3,
            vsync=GPIO15,
            href=GPIO24,
            shutdown=GPIO7,
            reset=GPIO9,
            mclk=GPIO27,
            size=adafruit_ov5640.OV5640_SIZE_QQVGA
        )
        self.cam.size = cam_size
        # Temporary:
        self.image_buf = bytearray(self.cam.capture_buffer_size)
