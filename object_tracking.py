"""
Demo of the Bebop vision code (basically flies around and saves out photos as it flies)
"""
from Bebop import Bebop
from DroneVision import DroneVision
import threading
import cv2
import time

isAlive = False

class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        #print("saving picture")
        img = self.vision.get_latest_valid_picture()
        filename = "/images/test_image_%06d.png" % self.index
        cv2.imwrite(filename, img)
        self.index +=1

    def detect(self, args):
        img = self.vision.get_latest_valid_picture()

# make my bebop object
bebop = Bebop()

# connect to the bebop
success = bebop.connect(5)

if (success):
    # start up the video
    bebopVision = DroneVision(bebop, is_bebop=True)

    userVision = UserVision(bebopVision)
    bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
    success = bebopVision.open_video()

    if (success):
        print("Vision successfully started!")
        #removed the user call to this function (it now happens in open_video())
        #bebopVision.start_video_buffering()

        # skipping actually flying for safety purposes indoors - if you want
        # different pictures, move the bebop around by hand
        print("Fly me around by hand!")
        # bebop.smart_sleep(5)
        print("Moving the camera using velocity")
        bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-2, duration=4)
        #
        # bebop.safe_takeoff(10)
        while True:
            bebop.smart_sleep(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        # bebop.smart_sleep(50)
        # bebop.safe_land(10)
        print("Finishing demo and stopping vision")
        bebopVision.close_video()

    # disconnect nicely so we don't need a reboot
    bebop.disconnect()
else:
    print("Error connecting to bebop.  Retry")
