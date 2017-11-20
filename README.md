# Hospital people detector

This ROS package provides a pipeline to detect and classify people in hospital environments.


### Prerequisites

Fast R-CNN (see: [Fast R-CNN installation instructions](https://github.com/rbgirshick/fast-rcnn/blob/master/README.md))


## Running the test

(I have to upload the Bag files)

For Kinect camera, run this package by typing

```
roslaunch hospital_people_detector hospital_Kinect.launch
```

For Xtion:

```
roslaunch hospital_people_detector hospital_Xtion.launch
```



## Running with your own data

In case of lack of "TF" you can simulate it by typing

```
roslaunch hospital_people_detector hospital_Xtion.launch simulate_tf:=true
```
