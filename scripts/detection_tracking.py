#!/usr/bin/env python


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

import _init_fastrcnn_paths
import caffe
from fast_rcnn.test import im_detect

from hospital_people_detector.msg import Proposals_msg
from hospital_people_detector.msg import Bbox
from hospital_people_detector.msg import Single_detection
from hospital_people_detector.msg import Detections

from hospital_people_detector.msg import Person
from hospital_people_detector.msg import People


from hospital_people_detector.msg import Single_track
from hospital_people_detector.msg import Tracks

from geometry_msgs.msg import Point



from utils_hpd import *
from preprocess_detections import preprocess_detections

from hungarian import Hungarian
from kalman_filter import KF_tracks
from nms import nms

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tf

import numpy as np

    
classifier_type = rospy.get_param('classifier_type')
validation_gate = rospy.get_param('validation_gate')
min_track_age = rospy.get_param('min_track_age')
class_labels = rospy.get_param('class_labels')


draw_bbox_proposals = rospy.get_param('draw_bbox_proposals')
draw_bbox_detections = rospy.get_param('draw_bbox_detections')
draw_bbox_tracks = rospy.get_param('draw_bbox_tracks')

save_images = rospy.get_param('save_images')
directory_for_images = rospy.get_param('directory_for_images')

hospital_prototxt = rospy.get_param('hospital_prototxt')
DepthJet_caffemodel = rospy.get_param('DepthJet_caffemodel')
RGB_caffemodel = rospy.get_param('RGB_caffemodel')

ID_GPU = rospy.get_param('ID_GPU')

camera_frame = rospy.get_param('camera_frame')
fixed_frame = rospy.get_param('fixed_frame')

tracks_list = {}
listener_tf = []
next_ID = 0

trans_od_base = []
trans_mat = []
rot_mat = []

old_stamp = 0


def callback(data):       

    global trans_od_base
    global trans_mat
    global rot_mat

    try:
        (trans,rot) = listener_tf.lookupTransform(fixed_frame, camera_frame, rospy.Time(0))

        trans_mat = np.asmatrix( tf.transformations.translation_matrix(trans) ) 
        rot_mat = np.asmatrix( tf.transformations.quaternion_matrix(rot) )
        trans_od_base = trans_mat*rot_mat

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.loginfo("- Detection and tracking - no transformation found")
        print(e)
        return


    bridge = CvBridge()
    try:
        if classifier_type == 'DepthJet':
            cv_image = bridge.imgmsg_to_cv2(data.img_jet, "passthrough")

        if classifier_type == 'RGB':
            cv_image = bridge.imgmsg_to_cv2(data.img_rgb, "passthrough")

    except CvBridgeError as e:
        print(e)
        return


    im_h , im_w , im_layers =  cv_image.shape

    intrinsic = [data.cx, data.cy, data.fx, data.fy]
    plane_coeff =[data.coeff_a, data.coeff_b, data.coeff_c, data.coeff_d]

    candidates = np.asarray(get_candidates(data.boxes), dtype=np.float)

    detections = np.empty([0, 0])

    if candidates.shape[0] != 0:

        # Classification using Fast R-CNN 
        scores, boxes_res = im_detect(net, cv_image, candidates)
    
        # Detections:  each row -> [x1 y1 x2 y2 score depth class]
        detections = preprocess_detections(scores, boxes_res, data.boxes)

        
    global tracks_list
    global next_ID
    global old_stamp

    time_stamp = int(str(data.img_jet.header.stamp))*10e-10
    delta_t = time_stamp - old_stamp
    if delta_t > 5: delta_t=0.06
    old_stamp = time_stamp

    #Update dt
    KF_tracks.dt_update(delta_t)
    KF_tracks.V_mat = camera2tracking_rotate_only(KF_tracks.V_mat_cam, rot_mat)
    KF_tracks.P_init = camera2tracking_rotate_only(KF_tracks.P_init_cam, rot_mat)

    # Apply Kalman Prediction Step
    for j in tracks_list:
        tracks_list[j].kalman_prediction()


    # Create cost matrix for data association
    # rows=Tracks (predictions), cols=detections
    m_inf = 1e10
    max_size = np.amax( [len(tracks_list), detections.shape[0]] )
    cost_m = m_inf*np.ones([ max_size,max_size ])

    for i in range(detections.shape[0]):
        d_x, d_y, d_z, d_class, d_score, d_h, d_w = read_detection(detections[i, :], plane_coeff, intrinsic)
        trk_x, trk_y, trk_z = cameraframe2trackingframe([d_x, d_y, d_z], trans_od_base)
        meas = np.array([[ trk_x ], [ trk_y ], [0] ])

        k = -1
        for j in tracks_list:
            k += 1
            Cpred = tracks_list[j].C_mat*np.transpose(tracks_list[j].Xest) 
            d_v = meas-Cpred
            S = tracks_list[j].C_mat * tracks_list[j].Pest * np.transpose(tracks_list[j].C_mat) + tracks_list[j].V_mat 

            # Mahalanobis variable corresponds to Mahalanobis_dist^2
            Mahalanobis = np.transpose(d_v)*inv(S)*d_v
            Mahalanobis = Mahalanobis[0,0]
    
            if ( Mahalanobis <= validation_gate ):
                cost_m[k, i] = Mahalanobis


    #Hungarian Algorithm (find track-detection pairs)
    hungarian = Hungarian()
    hungarian.calculate(cost_m)
    result_hungarian = hungarian.get_results() 

    #remove if cost>=m_inf, means mahalanobis_dist is too big   
    result_hungarian = [i for i in result_hungarian if not cost_m[i]>=m_inf] 
    
    tks_id = [j for j in tracks_list] #indices tracks_list
    #In each tuple replace "row number" by trk_ID  
    result_hungarian = [ (tks_id[ i[0] ], i[1]) for i in result_hungarian] 

    #list of associated tracks
    trks_paired = [i[0] for i in result_hungarian]
    #list of associated detections
    dts_paired = [i[1] for i in result_hungarian]

    #list of NO associated tracks
    trks_no_paired = [i for i in tracks_list if i not in trks_paired] 
    #list of NO associated detections
    dts_no_paired = [i for i in range(detections.shape[0]) if i not in dts_paired] 


    #Update Paired Tracks
    for resul in result_hungarian: 
        trk = resul[0]
        det = resul[1]

        d_x, d_y, d_z, d_class, d_score, d_h, d_w = read_detection(detections[det, :], plane_coeff, intrinsic)
        trk_x, trk_y, trk_z = cameraframe2trackingframe([d_x, d_y, d_z], trans_od_base)

        #APPLY KALMAN UPDATE
        tracks_list[trk].kalman_update([trk_x, trk_y, 0])
        tracks_list[trk].class_detected = d_class
        tracks_list[trk].score_detected = d_score
        tracks_list[trk].height = d_h
        tracks_list[trk].width = d_w
        tracks_list[trk].class_estimation(d_class)


    #Tracks with No detection
    for i in trks_no_paired:  
        x,y,z = tracks_list[i].Xest[0,0], tracks_list[i].Xest[0,1], tracks_list[i].Xest[0,2]
        tracks_list[i].class_detected = -1 #No detection
        tracks_list[i].score_detected = -1 #No detection

        #apply class estimation only for objects in the FOV of the camera        
        if is_in_FOV([x, y, z], trans_od_base, intrinsic, im_h, im_w): 
            tracks_list[i].class_estimation(0)


    #New detections, create tracks
    for i in  dts_no_paired: 
        d_x, d_y, d_z, d_class, d_score, d_h, d_w = read_detection(detections[i, :], plane_coeff, intrinsic)
        trk_x, trk_y, trk_z = cameraframe2trackingframe([d_x, d_y, d_z], trans_od_base)

        #Don't create new track if detection is too close to existing track
        if detection_istooclose([trk_x, trk_y, 0], tracks_list):
            continue

        tracks_list[next_ID] = KF_tracks([trk_x, trk_y, 0]) 
        tracks_list[next_ID].class_detected = d_class
        tracks_list[next_ID].score_detected = d_score
        tracks_list[next_ID].height = d_h
        tracks_list[next_ID].width = d_w
        tracks_list[next_ID].class_estimation(d_class)

        next_ID += 1


    #Remove tracks
    remove_from_list = remove_tracks(tracks_list)
    for i in remove_from_list: 
        del tracks_list[i]

    #**************
    # end Tracking 
    #**************



    #Draw bboxes (proposals)
    if draw_bbox_proposals:
        cv_image = draw_proposals(cv_image, data.boxes)

    #Draw bboxes (dtections)
    if draw_bbox_detections: 
        cv_image = draw_detection(cv_image, detections)

    #Draw bboxes (Tracks)
    if draw_bbox_tracks: 
        cv_image = draw_tracks(cv_image, tracks_list, trans_od_base, plane_coeff, intrinsic)

    #Save Image
    if save_images == True:

        f_temp = str(data.img_jet.header.stamp) 
        file_name = 'seq_' + f_temp[:-9] + '.' + f_temp[10:]
        image_name = directory_for_images + file_name + ".png"

        if not os.path.exists(directory_for_images):
            os.makedirs(directory_for_images)

        cv2.imwrite(image_name,cv_image)

    #Publish Image
    try:
        mobaids_image_pub.publish( bridge.cv2_to_imgmsg(cv_image, encoding="passthrough") )
    except CvBridgeError as e:
        print(e)


    #Publish detections msg
    msg_single_det = Single_detection()
    det_list_msg= []

    for i in range(detections.shape[0]):
        d_x, d_y, d_z, d_class, d_score, d_h, d_w = read_detection(detections[i, :], plane_coeff, intrinsic)

        msg_single_det.bounding_box = Bbox(detections[i, 0], detections[i, 1], detections[i, 2], detections[i, 3])
        msg_single_det.coordinates = Point(d_x, d_y, d_z)
        msg_single_det.class_id = d_class
        msg_single_det.class_label = class_labels[int(d_class)]
        msg_single_det.score = d_score

        det_list_msg.append(msg_single_det)

    msg_detections = Detections()
    msg_detections.header = data.img_jet.header
    msg_detections.header.frame_id = camera_frame
    msg_detections.detections = det_list_msg

    pub_detections.publish(msg_detections)


    #Publish tracks msg
    msg_single_trk = Single_track()
    trk_list_msg= []

    for j in tracks_list:

        if tracks_list[j].age < min_track_age:
            continue

        x,y,z = tracks_list[j].Xest[0,0], tracks_list[j].Xest[0,1], tracks_list[j].Xest[0,2]
        vx, vy, vz = tracks_list[j].Xest[0,3], tracks_list[j].Xest[0,4], tracks_list[j].Xest[0,5]
        trk_class = np.argmax(tracks_list[j].class_bel)

        msg_single_trk.track_ID = j
        msg_single_trk.position = Point(x, y, z)
        msg_single_trk.velocity = Point(vx, vy, vz)
        msg_single_trk.class_id = trk_class
        msg_single_trk.class_label = class_labels[int(trk_class)]
        msg_single_trk.hmm_probability = np.amax(tracks_list[j].class_bel)

        trk_list_msg.append(msg_single_trk)


    msg_tracks = Tracks()
    msg_tracks.header = data.img_jet.header
    msg_tracks.header.frame_id = fixed_frame
    msg_tracks.tracks = trk_list_msg
    pub_tracks.publish(msg_tracks)



    #Publish people msg
    msg_person = Person()
    person_list_msg= []

    for j in tracks_list:

        if tracks_list[j].age < min_track_age:
            continue

        x,y,z = tracks_list[j].Xest[0,0], tracks_list[j].Xest[0,1], tracks_list[j].Xest[0,2]
        vx, vy, vz = tracks_list[j].Xest[0,3], tracks_list[j].Xest[0,4], tracks_list[j].Xest[0,5]
        trk_class = np.argmax(tracks_list[j].class_bel)

        msg_person.name = "Track Nr " + str(j)
        msg_person.position = Point(x, y, z)
        msg_person.velocity = Point(vx, vy, vz)
        msg_person.reliability = np.amax(tracks_list[j].class_bel)
        msg_person.tagnames = ["class_id", "class_label"]
        msg_person.tags = [ str(trk_class), class_labels[int(trk_class)] ]     

        person_list_msg.append(msg_person)


    msg_people = People()
    msg_people.header = data.img_jet.header
    msg_people.header.frame_id = fixed_frame
    msg_people.people = person_list_msg
    pub_people.publish(msg_people)









def listener():
    rospy.init_node('detector_hospital', anonymous=False)
    global listener_tf
    listener_tf = tf.TransformListener()
    rospy.Subscriber("topic_proposals", Proposals_msg, callback, queue_size=1) #None not working , buff_size=2**24)
    rospy.spin()



if __name__ == '__main__':
    rospy.loginfo("Hospital detector node started")
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(ID_GPU)


    if classifier_type == 'DepthJet':
        net = caffe.Net(hospital_prototxt, DepthJet_caffemodel, caffe.TEST)
    if classifier_type == 'RGB':
        net = caffe.Net(hospital_prototxt, RGB_caffemodel, caffe.TEST)

    mobaids_image_pub = rospy.Publisher('topic_hospital_image', Image, queue_size=1)
    pub_detections = rospy.Publisher('Hospital_detections', Detections, queue_size=1)
    pub_tracks = rospy.Publisher('Hospital_tracks', Tracks, queue_size=1)

    pub_people = rospy.Publisher('Hospital_people', People, queue_size=1)

    listener()


