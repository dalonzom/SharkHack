from picamera import PiCamera
from time import sleep
import RPi.GPIO as GPIO
import sys

img_count=0

cam = PiCamera()
cam.rotation = -90

GPIO.setmode(GPIO.BCM)

GPIO.setup(18,GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(15,GPIO.IN, pull_up_down=GPIO.PUD_UP)

cam.start_preview()

while True:
    try:
        take_pic = GPIO.input(18)
        exit_state = GPIO.input(15)
        if take_pic == False:
            cam.capture('/home/pi/Desktop/images/image_%s.jpg' % img_count)
            print("button was pressed")
            sleep(0.2)
            img_count += 1
        elif exit_state == False:
            cam.stop_preview()
            sys.exit()
            
    except KeyboardInterrupt:
        cam.stop_preview()
        sys.exit()
        
