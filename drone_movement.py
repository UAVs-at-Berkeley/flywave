import time
import math
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import signal
import sys
import matplotlib.pyplot as plt
from Bebop import Bebop
from DroneVision import DroneVision
from preprocess import preprocess

def rightArc():
#     #rotate in an arc to the right
    bebop.fly_direct(roll=15, pitch=0, yaw=0, vertical_movement=0, duration=3)
    bebop.fly_direct(roll=0, pitch=15, yaw=0, vertical_movement=0, duration=3)
    bebop.fly_direct(roll=0, pitch=0, yaw=20, vertical_movement=0, duration=4)
    print('right')
    return

def leftArc():
    bebop.fly_direct(roll=-15, pitch=0, yaw=0, vertical_movement=0, duration=3)
    bebop.fly_direct(roll=0, pitch=-15, yaw=0, vertical_movement=0, duration=3)
    bebop.fly_direct(roll=0, pitch=0, yaw=-5, vertical_movement=0, duration=4)
    print('left')
    return

def land():
    active = False
    print('land')
    return

def flip():
    bebop.flip(back)
    print('flip')
    return

def resolveAction(gesture):
    actions = {
        0 : rightArc,
        1 : leftArc,
        2 : land,
        3 : flip # TODO: Replace this with do nothing command?
    }
    actions[gesture]
    return

def kill(signal, frame):
    """ Catches SIGINT and lands the drone """

    print("Ctrl+C registed - exiting gracefully")
    bebop.smart_sleep(5)
    bebop.safe_land(10)
    print("Landed successfully. Now exiting")
    bebop.disconnect()
    sys.exit(0)

signal.signal(signal.SIGINT, kill)

class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        #print("saving picture")
        img = self.vision.get_latest_valid_picture()

        # filename = "test_image_%06d.png" % self.index
        #cv2.imwrite(filename, img)
        self.index +=1

    def detect(self, args):
        img = self.vision.get_latest_valid_picture()

model = load_model('4_gesture_model.h5')

bebop = Bebop()

success = bebop.connect(5)

active = True
COUNT_EVERY = 20

if (success):
    # start up the video
    bebopVision = DroneVision(bebop, is_bebop=True)
    userVision = UserVision(bebopVision)
    bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
    success = bebopVision.open_video()

    if success:
        bebopVision = DroneVision(bebop, is_bebop=True)
        userVision = UserVision(bebopVision)
        bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
        success = bebopVision.open_video()



        print("Vision successfully started!")
        #removed the user call to this function (it now happens in open_video())
        #bebopVision.start_video_buffering()

        # skipping actually flying for safety purposes indoors - if you want
        # different pictures, move the bebop around by hand
        print("Fly me around by hand!")
        bebop.smart_sleep(5)
        print("Moving the camera using velocity")
        # bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-2, duration=4)

        bebop.safe_takeoff(10)
        # bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=1)
        bebop.smart_sleep(5)

        count = 0
        scale = 4

        while active:
            if count % COUNT_EVERY == 0:
                #TODO: How to get individual frames from video stream in bebop?
                img = userVision.vision.get_latest_valid_picture()
        #         cv2.imwrite('curr_frame.png', img)
        #         img = image.load_img('curr_frame.png', grayscale=True)
                img = image.img_to_array(img)
                print(img.shape)
        #         plt.imshow(img)
        #         img = preprocess(img, 4)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(img,(5,5),0)
        #         ret, img = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale))
                img = np.resize(img, (1, img.shape[0], img.shape[1],1))
                prediction = model.predict(img)
                print(prediction)
                if len(prediction) > 0:
                    gesture = np.argmax(prediction[0])
                    #print(gesture, prediction)
                    resolveAction(gesture)
            count += 1
            if count == 2000:
                break

bebop.smart_sleep(5)
bebop.safe_land(10)
print("Done - Disconnecting")

bebop.disconnect()
