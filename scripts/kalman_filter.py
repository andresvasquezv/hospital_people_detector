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

# --------------------------------------------------------
# Detection and tracking of people with mobility aids 
# Written by Andres Vasquez
#
#
# KALMAN FILTER:
#
# State space: [x_coord, depth_coor, x_vel, depth_vel]
#
# Predict state: Xpred = A*Xest + B*u        
# Predict covariance: Ppred = A*Pest*A' + W;
# Kalman gain: KG = Ppred*C'*inv(C*Ppred*C' + V)
# Update state: Xest = Xpred + KG*(Measurement - C*Xpred)
# Update covariance: Pest = (I - KG*C)*Ppred
#
# --------------------------------------------------------

import rospy
import numpy as np
from numpy.linalg import inv


class KF_tracks:

    vel_noise = rospy.get_param('KALFIL/vel_noise') #m/sec
    alpha_x = rospy.get_param('KALFIL/alpha_x') #m (Camera FRAME)
    alpha_y = rospy.get_param('KALFIL/alpha_y') #m (Camera FRAME) 
    alpha_z = rospy.get_param('KALFIL/alpha_z') #m (depth in Camera FRAME)
    alpha_v = rospy.get_param('KALFIL/alpha_v') #m/s


    tr = 2e-4 
    hh = 1 - 4*tr
    transition_model = np.array([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], 
                                [0.0000, hh, tr, tr, tr, tr], 
                                [0.0000, tr, hh, tr, tr, tr], 
                                [0.0000, tr, tr, hh, tr, tr], 
                                [0.0000, tr, tr, tr, hh, tr], 
                                [0.0000, tr, tr, tr, tr, hh]]) 
 
    observ_model = np.array([  [0.97084, 0.02299, 0.00248, 0.00108, 0.00066, 0.00194], 
                                [0.12808, 0.84869, 0.00036, 0.00073, 0.01814, 0.00399],
                                [0.03622, 0.03548, 0.90318, 0.01996, 0.00222, 0.00296],
                                [0.14527, 0.20270, 0.05405, 0.35473, 0.18919, 0.05405],
                                [0.04617, 0.50792, 0.00132, 0.00264, 0.43799, 0.00396],
                                [0.07640, 0.56180, 0.00112, 0.00112, 0.01124, 0.34831] ]) 

    prior_belief = np.array([0.76894, 0.10551, 0.04748, 0.01275, 0.03046, 0.03485])

    dt = 0.06 #seconds (initial value)

    C_mat = np.matrix( [ [1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0], 
             [0, 0, 1, 0, 0, 0] ])  #measurement function C

    W_mat = []
    A_mat = []
    P_init = [] #tracking frame
    V_mat = [] #tracking frame
    P_init_cam = [] #camera frame
    V_mat_cam = [] #camera frame



    @classmethod
    def dt_update(cls, delta_t):

        cls.dt = delta_t

        #Covariance Matrix
        cls.W_mat = np.matrix( [ [(cls.dt**2), 0, 0, (cls.dt), 0, 0], 
                     [0, (cls.dt**2), 0, 0, (cls.dt), 0], 
                     [0, 0, 0, 0, 0, 0], 
                 [(cls.dt), 0, 0, 1, 0, 0],
                     [0, (cls.dt), 0, 0, 1, 0], 
                     [0, 0, 0, 0, 0, 0] ])*(cls.vel_noise**2) 

        #Model Matrix
        cls.A_mat = np.matrix( [ [1, 0, 0, cls.dt, 0, 0], 
             [0, 1, 0, 0, cls.dt, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0], 
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0] ]) 

        cls.P_init_cam = np.matrix(np.diag( [cls.alpha_x**2, cls.alpha_y**2, cls.alpha_z**2,    
                cls.alpha_v**2, cls.alpha_v**2, cls.alpha_v**2] ))

        cls.V_mat_cam = np.matrix(np.diag( [cls.alpha_x**2, cls.alpha_y**2, cls.alpha_z**2] ))



    def __init__(self, point):
        self.age = 0
        self.height = 0
        self.width = 0
        self.class_detected = 0
        self.score_detected = 0
        
        self.Xest = np.matrix([point[0], point[1], point[2], 0, 0, 0])  #Initial state
        self.Pest = self.P_init.copy()  

        #Prior belief
        self.class_bel = self.prior_belief.copy()
    
        

    def class_estimation(self, the_class):
        class_bel_old = np.copy(self.class_bel)

        #Hidden Markov Model
        for t_cla in range(6):
            the_sum = 0
            for t_cla_past in range(6):
                the_sum = the_sum + self.transition_model[t_cla_past,t_cla]*class_bel_old[t_cla_past]
            self.class_bel[t_cla] = self.observ_model[t_cla,int(the_class)]*the_sum 
    
        self.class_bel = self.class_bel/np.sum(self.class_bel)



    def kalman_prediction(self):
        self.Pest = self.A_mat*self.Pest*np.transpose(self.A_mat) + self.W_mat
        _Xest = np.transpose( np.matrix(self.Xest) )
        Xpred = self.A_mat*_Xest
        self.Xest = Xpred.flatten() 


    
    def kalman_update(self, point):
        self.age += 1

        # Kalman Gain
        S = self.C_mat*self.Pest*np.transpose(self.C_mat) + self.V_mat
        KG = self.Pest*np.transpose(self.C_mat)*inv(S)
        
        Xmeas = np.array([ [point[0]], [point[1]], [point[2]] ]) #MEASUREMENT

        # update covariance
        self.Pest = ( np.eye(6) - KG*self.C_mat ) * self.Pest
        
        Xpred = np.transpose( np.matrix( self.Xest ))

        # update state  
        _Xest = Xpred + KG*(Xmeas - self.C_mat*Xpred)
        self.Xest = _Xest.flatten()




