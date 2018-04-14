import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# from pynput.keyboard import Controller

"""
 Copyright 2015-2016, 3D Robotics.
guided_set_speed_yaw.py: (Copter Only)

This example shows how to move/direct Copter and send commands in GUIDED mode using DroneKit Python.

Example documentation: http://python.dronekit.io/examples/guided-set-speed-yaw-demo.html
"""


# from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
# from pymavlink import mavutil # Needed for command message definitions
import time
import math


#Set up option parsing to get connection string
import argparse
parser = argparse.ArgumentParser(description='Control Copter and send commands in GUIDED mode ')
parser.add_argument('--connect',
                   help="Vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect
sitl = None


#Start SITL if no connection string specified
# if not connection_string:
#     import dronekit_sitl
#     sitl = dronekit_sitl.start_default()
#     connection_string = sitl.connection_string()
#
#
# # Connect to the Vehicle
# print('Connecting to vehicle on: %s' % connection_string)
# vehicle = connect(connection_string, wait_ready=True)
# print("type: ", str(type(vehicle)))
#
# drone_movement.arm_and_takeoff(vehicle, 5)
# vehicle.groundspeed=5

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 1

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
box_x = 0;
box_y = 0;
# keyboard = Controller()
def get_center():
    return box_x, box_y
def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    num = None
    for i in range(len(scores[0])):
        if classes[0,i] == 1:
            num = i
            break

    classes = classes[0,i:i+1]
    classes = np.reshape(classes, (1))
    boxes = boxes[0,i:i+1,:]
    scores = scores[0,i:i+1]
    scores = np.reshape(scores, (1))
    print(boxes[0])
    x = (boxes[0][1]+ boxes[0][3])/2
    y = (boxes[0][0]+ boxes[0][2])/2
    # box_x,box_y = output_q.get()
    scaling_factor = 10
    offset_x = (x - 0.5)*scaling_factor
    offset_y = (y - 0.5)*scaling_factor
    print(offset_x, offset_y, time.time())
    # keyboard.press("s")
    # drone_movement.move(offset_x, offset_y, time.time())
    # moveDrone(boxes[0])
    num_detections = 1

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        classes.astype(np.int32),
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        output_q.put(detect_objects(frame_rgb, sess, detection_graph))
    fps.stop()
    sess.close()


from Bebop import Bebop
from DroneVisionGUI import DroneVisionGUI
import threading
import cv2
import time

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

def detection(drone_vision, args):
            userVision = UserVision(bebopVision)
            bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
            success = bebopVision.open_video()
            # bebop.safe_takeoff(10)
            bebop.smart_sleep(10)
            ("Take off")
            count = 0

            while True and success and count < 200:  # fps._numFrames < 120
                frame = userVision.vision.get_latest_valid_picture()
                input_q.put(frame)

                t = time.time()

                # drone_movement.goto(vehicle, offset_y, offset_x, vehicle.simple_goto)
                #print("OUTPUT", box_x, " ", box_y)
                if count % 10 == 0:
                    output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
                    filename = "/test/test_image_%06d.png" % count
                    cv2.imwrite(filename, output_rgb)
                # cv2.imshow('Video', output_rgb)
                # fps.update()

                print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
                #print(get_center())
                count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # bebop.safe_land(10)
            bebopVision.close_video()
            bebop.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))
    #
    # video_capture = WebcamVideoStream(src=args.video_source,
    #                                   width=args.width,
    #                                   height=args.height).start()
    # fps = FPS().start()

    isAlive = False
    bebop = Bebop()
    success = bebop.connect(5)
    print("Drone Initialized " + str(success))
    if success:
        bebopVision = DroneVisionGUI(bebop, is_bebop=True)
    else:
        print("Error: Bebop not found")
    # fps.stop()
    # print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    # print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
