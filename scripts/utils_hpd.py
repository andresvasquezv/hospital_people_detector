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

import numpy as np
from numpy.linalg import inv

import cv2
import PIL
from PIL import ImageFont
from PIL import Image as Image_pil
from PIL import ImageDraw

min_separation = rospy.get_param('min_separation')
uncertainty_tolerance = rospy.get_param('uncertainty_tolerance')
background_tolerance = rospy.get_param('background_tolerance')
min_track_age = rospy.get_param('min_track_age')
class_labels = rospy.get_param('class_labels')


def cameraframe2trackingframe(point, transf_mat):
    point_in = np.asmatrix( [ point[0], point[1], point[2], 1 ] )
    point_out =  transf_mat*np.transpose(point_in)
    return point_out[0,0], point_out[1,0], point_out[2,0]
    #return point_out[0,0], point_out[1,0], 0 #force point to lay on the floor

def trackingframe2cameraframe(point, transf_mat):
    point_in = np.asmatrix( [ point[0], point[1], point[2], 1 ] )
    point_out = inv(transf_mat)*np.transpose(point_in)
    return point_out[0,0], point_out[1,0], point_out[2,0]


def rotate_matrix(matrix_in, rot_mat):
    mm = np.zeros([4,4])
    mm[:3,:3] = matrix_in
    res_rot = rot_mat * mm * np.transpose(rot_mat)
    return res_rot[:3,:3] 


def camera2tracking_rotate_only(matrix_in, rot_mat):
    if matrix_in.shape[0] == 3:
        return rotate_matrix(matrix_in, rot_mat)
    if matrix_in.shape[0] == 6:
        result = np.zeros([6,6])
        result[0:3,0:3] = rotate_matrix(matrix_in[0:3,0:3], rot_mat)
        result[0:3,3:6] = rotate_matrix(matrix_in[0:3,3:6], rot_mat)
        result[3:6,0:3] = rotate_matrix(matrix_in[3:6,0:3], rot_mat)
        result[3:6,3:6] = rotate_matrix(matrix_in[3:6,3:6], rot_mat)
        return result


def detection_istooclose(point, tracks_list):
    for i in tracks_list:
        x = tracks_list[i].Xest[0,0]
        y = tracks_list[i].Xest[0,1]
        z = tracks_list[i].Xest[0,2]
        dist = ( (x-point[0])**2 + (y-point[1])**2 + (z-point[2])**2 )**0.5
        if dist < min_separation:
            return True
    return False


def remove_tracks(tracks_list):
    to_remove=[]
    for j in tracks_list:
        #eigen values
        ev = np.linalg.eigvals(tracks_list[j].Pest)

        if (ev[0]**0.5 > uncertainty_tolerance and 
            ev[1]**0.5 > uncertainty_tolerance or 
            tracks_list[j].class_bel[0] > background_tolerance ):    
            to_remove.append(j)
    return to_remove


def get_candidates(Bbox):
    box_list = []
    for i in range(len(Bbox)/5):
        prop_row = []   
        prop_row.append(Bbox[i*5 + 0]) #x1 
        prop_row.append(Bbox[i*5 + 1]) #y1
        prop_row.append(Bbox[i*5 + 2]) #x2
        prop_row.append(Bbox[i*5 + 3]) #y2
        box_list.append(prop_row)
    return box_list


def read_detection(data, plane_coeff, intrinsic):
    z = data[5] #depth
    x_med = (data[0] + data[2])/2
    x = (x_med - intrinsic[0]) / intrinsic[2] * z
    y_med = (data[1] + data[3])/2
    y = (y_med - intrinsic[1]) / intrinsic[3] * z
    the_class = data[6]
    the_score = data[4]

    y_floor = -(plane_coeff[0]*x + plane_coeff[2]*z + plane_coeff[3])/plane_coeff[1]

    y_top = (data[1] - intrinsic[1]) / intrinsic[3] * z
    height = y_top - y_floor 
    #width = pixel2pcl(data[0],None,z) - pixel2pcl(data[2],None,z)

    width = (data[0] - data[2]) / intrinsic[2] * z

    return x, y, z, the_class, the_score, np.absolute(height), np.absolute(width)


def is_in_FOV(point, transf_mat, intrinsic, im_h, im_w):    
    x_temp, y_tem, d_temp = trackingframe2cameraframe( point, transf_mat )
    x_pixel = x_temp*intrinsic[2]/d_temp + intrinsic[0]

    if x_pixel>0 and x_pixel<im_w and d_temp>0: # and d_temp<10: 
        return True
    else:
        return False


def select_color(id_class):
    if id_class == 0: color = (50,50,50) #cyan
    if id_class == 1: color = (255,255,255) #white
    #if id_class == 1: color = (204,204,0) #cyan
    if id_class == 2: color = (0,255,255) #yellow
    if id_class == 3: color = (255,0,255) #pink
    if id_class == 4: color = (0,0,255) #red
    if id_class == 5: color = (0,145,255) #orange
    return color


def draw_proposals(cv_image, data):
    for i in range(len(data)/5):
        xx1 = data[i*5 + 0]
        yy1 = data[i*5 + 1]
        xx2 = data[i*5 + 2]
        yy2 = data[i*5 + 3]
        cv2.rectangle(cv_image, (xx1, yy1), (xx2,yy2), (0,0,0), 2)
    return cv_image


def draw_detection(cv_image, detections):

    for i in range(detections.shape[0]): 
        x1 = int(detections[i,0])
        y1 = int(detections[i,1])
        x2 = int(detections[i,2])
        y2 = int(detections[i,3])
        score = str(detections[i,4])

        det_class = class_labels[int(detections[i,6])]
        cv2.rectangle(cv_image, (x1, y1), (x2,y2), (0,255,0), 2)

        font_pil = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans-Bold.ttf",15)
        cv_image2 = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        pil_im = Image_pil.fromarray(cv_image2)
        draw = ImageDraw.Draw(pil_im)
        draw.text((x1+5, y1+4),det_class,(255,0,0),font=font_pil)
        draw.text((x1+5, y1+20),score,(255,0,0),font=font_pil)

        cv_image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    return cv_image


def draw_tracks(cv_image, tracks_list, trans_od_base, plane_coeff, intrinsic):

    im_h , im_w , im_layers =  cv_image.shape

    for j in tracks_list:

        x,y,z = tracks_list[j].Xest[0,0], tracks_list[j].Xest[0,1], tracks_list[j].Xest[0,2]
        wid = tracks_list[j].width
        hei = tracks_list[j].height
        det_class = int(tracks_list[j].class_detected)
        det_score = str(tracks_list[j].score_detected)

        if not is_in_FOV([x, y, z], trans_od_base, intrinsic, im_h, im_w) or \
                tracks_list[j].age < min_track_age:
            continue

        xx, yy, de = trackingframe2cameraframe( [x, y, z], trans_od_base )
        yy = -(plane_coeff[0]*xx + plane_coeff[2]*de + plane_coeff[3])/plane_coeff[1]

        #HMM
        hmm = tracks_list[j].class_bel
        hmm_clas = np.argmax(tracks_list[j].class_bel)

        class_str = class_labels[hmm_clas]

        x1 = int( (xx-wid/2)*intrinsic[2]/de + intrinsic[0] )
        y1 = int( (yy-hei)*intrinsic[3]/de + intrinsic[1] )

        x2 = int( (xx+wid/2)*intrinsic[2]/de + intrinsic[0] )
        y2 = int( yy*intrinsic[3]/de + intrinsic[1] )

        if x1<0: x1=0
        if x1>im_w: x1=im_w
        if x2<0: x2=0
        if x2>im_w: x2=im_w
        if y1<0: y1=0
        if y1>im_h: y1=im_h
        if y2<0: y2=0
        if y2>im_h: y2=im_h

        r_color = select_color(hmm_clas)

        #Draw background first: transparent rectangle
        overlay = cv_image.copy()
        Weight = 0.4 #transparency
        cv2.rectangle(overlay, (x1, y1), (x2,y2), (0,0,0), -1)
        cv2.addWeighted(overlay, Weight, cv_image, 1.0 - Weight , 0.0, cv_image)

        #Draw colored Bbox 
        cv2.rectangle(cv_image, (x1, y1), (x2,y2), r_color, 2)

        #Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_pil = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans-Bold.ttf",16)    
        overlay2 = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        pil_im = Image_pil.fromarray(overlay2)
        draw = ImageDraw.Draw(pil_im)
        draw.text((x1+5, y2-20),class_str, r_color[::-1], font=font_pil)
        track_nr = 'T' + str(j) 
        draw.text((x1+5, y1-20),track_nr, r_color[::-1], font=font_pil)

        #Draw text for detection (if exists)
        if det_class != -1:     
            font_pil2 = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans-Bold.ttf",15)
            class_str_det = class_labels[det_class]
            draw.text((x1+5, y1+4),class_str_det,(0,250,0),font=font_pil2)
            draw.text((x1+5, y1+20),det_score,(0,250,0),font=font_pil2)
        cv_image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

        #Draw histogram
        for jj in range(6): 
            r_color = select_color(jj)
            _y = int(y2-30-jj*10)
            pt1 = (int(x1+8), _y)
            pt2 = (int(x1+8+hmm[jj]*50), _y)
            cv2.line(cv_image, pt1, pt2, r_color, 6)

    return cv_image

