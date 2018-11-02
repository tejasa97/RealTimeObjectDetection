# RealTimeObjectDetection
Python program utilising OpenCV DNN to detect objects video feed and draw bounding boxes. Works on SSD Mobilenet / YOLO.
Used multi threading to increase speed of input capture from webcam to more than 100FPS (only webcam FPS, mind you. Detection rate sits at around 13 FPS), so that comparitively lesser lag is observed.

# Output
*Below are some snapshots of the real time video output*

Car image on webcam

<img src = images/car.png height = 300>

Dog image on webcam

<img src = images/dog.png height = 300>

Cat image on webcam

<img src = images/cat.png height = 300>

Hooman (me!) on webcam

<img src = images/person.png height = 300>

# Usage
Execute with

```sh
python try11.py
```

# Dependencies
```sh
pip install opencv-python
```
