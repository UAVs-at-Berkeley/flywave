# from __future__ import print_function
# from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
# from pymavlink import mavutil # Needed for command message definitions
import time
import math
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
from Bebop import Bebop

bebop = Bebop()

success = bebop.connect(5)

active = True

def rightArc():
    #rotate in an arc to the right
    bebop.fly_direct(roll=15, pitch=0, yaw=0, vertical_movement=0, duration=3)
    bebop.fly_direct(roll=0, pitch=15, yaw=0, vertical_movement=0, duration=3)
    bebop.fly_direct(roll=0, pitch=0, yaw=20, vertical_movement=0, duration=4)
    return

def leftArc():
    bebop.fly_direct(roll=-15, pitch=0, yaw=0, vertical_movement=0, duration=3)
    bebop.fly_direct(roll=0, pitch=-15, yaw=0, vertical_movement=0, duration=3)
    bebop.fly_direct(roll=0, pitch=0, yaw=-5, vertical_movement=0, duration=4)
    return

def land():
    active = False

def resolveAction(gesture):
    actions = {
        0 : rightArc,
        1 : leftArc,
        2 : land
    }
    actions[gesture]
    return

if (success):
    # start up the video
    bebopVision = DroneVision(bebop, is_bebop=True)

    # Load model
    model = load_model('arm_model.h5')

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
        bebop.smart_sleep(5)
        print("Moving the camera using velocity")
        # bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-2, duration=4)

        bebop.safe_takeoff(10)
        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=1)
        bebop.smart_sleep(5)

        while active:
            #TODO: How to get individual frames from video stream in bebop?
            img = image.load_img('img.png', target_size=(64, 64), grayscale=True)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            images = np.vstack([img])
            prediction = model.predict_classes(images, batch_size=10)
            if len(prediction) > 0:
                gesture = prediction[0]
                #print(gesture, prediction)
                resolveAction(gesture)
            
        bebop.smart_sleep(5)
        bebop.safe_land(10)
        print("Done - Disconnecting")
    else:
        print("Error - bebopVision.open_video() failed")

    bebop.disconnect()
else:
    print("Error - Bebop cannot connect")


