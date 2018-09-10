# Hospital people detector

This ROS package provides a pipeline to detect and classify people in hospital environments.


### Prerequisites

fast-rcnn-mobility-aids (see [installation instructions](https://github.com/marinaKollmitz/fast-rcnn-mobility-aids/blob/master/README.md))

## Running the test

First download the test bagfile wich contains a sequence collected with a Kinect V2 and Xtion from here:
```
wget https://drive.google.com/open?id=1mNeRLmOISqTkXUaf4OdVDLOUaAtb_5On
```

To run the test for Kinect camera:

```
roslaunch hospital_people_detector hospital_Kinect_test.launch
```

To run the test for Xtion:

```
roslaunch hospital_people_detector hospital_Xtion_test.launch
```

You can choose to use the RGB or DepthJet detector by modifying the parameter in the launch file:

```
<param name="classifier_type" value="RGB" />
```
or

```
<param name="classifier_type" value="DepthJet" />
```

IMPORTANT: Before you run the test you have to change the paths to Caffe and Fast R-CNN directories in the launch file:
```
<param name="caffe_directory" value="/home/my_user/fast-rcnn/caffe-fast-rcnn" />
<param name="fast_rcnn_directory" value="/home/my_user/fast-rcnn" />
```

Play the bagfile by typing:
```
rosbag play Downloads/test_hospital.bag --clock
```

