# Project FlyWave

The goal of the FlyWave team of UAVs@Berkeley is to develop a drone that can
maneuver autonomously using computer vision and deep learning to detect hand
gestures.

## Highlights
- Person classification
- Hand gesture detection
- Person movement tracking
- Gesture detection
- Using gestures to move the drone

## Requirements
1. Parrot Bebop 2
2. Packages: Tensorflow, Keras, OpenCV, and PyParrot (dependencies described [here](https://github.com/amymcgovern/pyparrot/wiki/Installing-pyparrot)).

## Setup
1. Connect to the Bebop's Wifi.
2. BEFORE RUNNING ANY SCRIPTS, STAND CLEAR OF DRONE'S TAKEOFF AREA.
3. In terminal, run:
```
python drone_movement.py
```
4. Stand in front of Bebop, and make gestures as desired.
Right arm out: Move bebop in an arc to the right
Left arm out: Move bebop in an arc to the left
Right arm up: Land drone
Left arm up: Flip

## Credits
Credit to:
[PyParrot](https://github.com/amymcgovern/pyparrot)

Copyright for object detection:
See [LICENSE](LICENSE) for details.
Copyright (c) 2017 [Dat Tran](http://www.dat-tran.com/).
