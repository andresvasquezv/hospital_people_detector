# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import rospy
import os.path as osp
import sys


caffe_directory = rospy.get_param('caffe_directory')
fast_rcnn_directory = rospy.get_param('fast_rcnn_directory')

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(caffe_directory, 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(fast_rcnn_directory, 'lib')
add_path(lib_path)
