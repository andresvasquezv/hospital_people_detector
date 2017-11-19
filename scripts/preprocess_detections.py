###############################################################################
#
# Software License Agreement (BSD License)
# 
# Copyright (c) 2017 Andres Vasquez
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Andres Vasquez
#
###############################################################################


import rospy
from nms import nms
import numpy as np

min_score = rospy.get_param('min_score')
NMS_THRESH = rospy.get_param('NMS_THRESH')


def preprocess_detections(scores, boxes_res, data_candidates): 

    class_boxes = []
    class_scores = []
    class_class = []
    class_depth = []

    n_prop=15 #Number of proposals per segment
    number_of_segments = scores.shape[0]/n_prop

    #Obtain the most representative proposal for each segment.
    #Reject a segment if it is classified as "Background"
    for j in range(number_of_segments): 

        countDetSeg = np.zeros(6) # Contains counters for each class (for a single segment)
        seg_scores = scores[j*n_prop:(j+1)*n_prop]
        seg_boxes_res = boxes_res[j*n_prop:(j+1)*n_prop]
        seg_cls_max_scores = np.argmax(seg_scores, axis=1)

        #Count Nr of detections for each class
        for i in seg_cls_max_scores:
            countDetSeg[i] += 1
        countDetSeg = countDetSeg/np.sum(countDetSeg)
        
        #The class with highest number is "Winner class"
        #Backgrouns is winner only if all detections are background
        win_class = np.argmax(countDetSeg[1:6])+1
        if countDetSeg[0] == 1: 
            win_class = 0

        if win_class == 0: #background
            continue

        #from detections of "winner class" obtain the detection with the highest score,
        #this detection represents the segment
        win_ind = np.argmax(seg_scores[:,win_class])

        cls_box = seg_boxes_res[win_ind, 4*win_class:4*(win_class + 1)]
        cls_score = seg_scores[win_ind, win_class]
        cls_ind = win_class
        cls_depth = data_candidates[(j*n_prop + win_ind)*5 + 4]

        if len(class_boxes) == 0 and cls_score > min_score: 
            class_boxes = np.array([cls_box])
            class_scores = np.array([cls_score])
            class_class = np.array([cls_ind])
            class_depth = np.array([cls_depth])

        if len(class_boxes) > 0 and cls_score > min_score: 
            class_boxes = np.r_[ class_boxes, np.array([cls_box]) ] 
            class_scores = np.concatenate( (class_scores, np.array([cls_score]) ), axis=0 )
            class_class = np.concatenate( (class_class, np.array([cls_ind]) ), axis=0 )
            class_depth = np.concatenate( (class_depth, np.array([cls_depth]) ), axis=0 )

    # We have one detection per segment, 
    # now, we apply Non-maximun Supression on this set of detections
    if class_boxes != []:    
        dets = np.hstack((class_boxes, class_scores[:, np.newaxis])).astype(np.float32)
           
        keep = nms(dets, NMS_THRESH) #indices of class_boxes after NMS
        dets = dets[keep, :] #boxes and scores after NMS
        dets_class = class_class[keep] #classes of the boxes
        dets_depth = class_depth[keep]

        result = dets.copy()
        result = np.c_[ result, dets_depth ].astype(np.float32)
        result = np.c_[ result, dets_class ]

        return result #each row -> [x1 y1 x2 y2 certainty depth class]
 
    return np.empty(0)

