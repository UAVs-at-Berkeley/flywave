import numpy as np
import cv2
import re
from matplotlib import pyplot as plt

def func(path):
    frame = cv2.imread(path)
    plt.imshow(frame),plt.show()
    if ".DS_Store" not in path:
        frame = cv2.resize(frame,(128,128))

        converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
        #cv2.imshow("original",converted2)

        lowerBoundary = np.array([0,40,30],dtype="uint8")
        upperBoundary = np.array([43,255,254],dtype="uint8")
        skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
        skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
        #cv2.imshow("masked",skinMask)
        plt.imshow(skinMask),plt.show()
        skinMask = cv2.medianBlur(skinMask, 5)

        skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
        #frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
        #skin = cv2.bitwise_and(frame, frame, mask = skinMask)

        #skinGray=cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
        cv2.imshow("masked2",skin)
        img2 = cv2.Canny(skin,60,60)
        plt.imshow(img2),plt.show()
        #cv2.imshow("edge detection",img2)
        letter = re.search(r"/[A-Z]/", path).group(0)
        print(re.search(r"[a-zA-Z0-9]+\.jpg", path).group(0))
        cv2.imwrite("preprocessed/" + letter + "/" + re.search(r"[a-zA-Z0-9]+\.jpg", path).group(0), img2)
        '''
        hog = cv2.HOGDescriptor()
        h = hog.compute(img2)
        print(len(h))

        '''
        surf = cv2.xfeatures2d.SURF_create()
        #surf.extended=True
        img2 = cv2.resize(img2,(256,256))
        kp, des = surf.detectAndCompute(img2,None)
        #print(len(des))
        img2 = cv2.drawKeypoints(img2,kp,None,(0,0,255),4)
        plt.imshow(img2),plt.show()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        print(len(des))
        return des

def func2(path):
    frame = cv2.imread(path)
    frame = cv2.resize(frame,(128,128))
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    #cv2.imshow("original",converted2)

    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    #cv2.imshow("masked",skinMask)

    skinMask = cv2.medianBlur(skinMask, 5)

    skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)

    #cv2.imshow("masked2",skin)
    img2 = cv2.Canny(skin,60,60)
    #cv2.imshow("edge detection",img2)
    img2 = cv2.resize(img2,(256,256))
    orb = cv2.xfeatures2d.ORB_create()
    kp, des = orb.detectAndCompute(img2,None)

    #print(len(des2))
    img2 = cv2.drawKeypoints(img2,kp,None,color=(0,255,0), flags=0)
    #plt.imshow(img2),plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return des2
#func("001.jpg")
