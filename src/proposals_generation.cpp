/*******************************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2017 Andres Vasquez
 *  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Andres Vasquez

 ******************************************************************************/


#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

#include "sensor_msgs/CameraInfo.h"
#include <tf/transform_listener.h>

#include "hospital_people_detector/Proposals_msg.h"

using namespace cv;


tf::TransformListener* pListener = NULL;

ros::NodeHandle* pnh = NULL; 

// Intrinsic parameters of the camera
float fx=0, cx=0, fy=0, cy=0, im_h=0, im_w=0;

// Coefficients of the plane
float coeff_a, coeff_b, coeff_c, coeff_d; 

// Average size of a person
float av_human_h = 1.75; // meters
float av_human_w = 0.6; // meters


ros::Publisher pub_proposals;
ros::Publisher pub_pcl;

sensor_msgs::Image my_RGBimg_msg;

std::string gp_source; //Ground plane can be computed form TF or can be estimated using RANSAC
std::string rgb_image_topic; 
std::string depth_image_topic;
std::string camera_info_topic;
std::string camera_frame;
std::string base_frame;
float camera_height; //Height of the camera
bool Xtion_camera; //True if Xtion camera is used, and false if Kinect is used


void depth2jet(Mat& input_depth, Mat& colored_depth);
std::vector<float > get_proposals(cv::Mat depthImage);


// Callback for camera info. (Intrinsic parameters)
void cb_camera_info(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
  fx = msg->K[0];
  cx = msg->K[2]; 
  fy = msg->K[4];
  cy = msg->K[5];
  im_h = msg->height;
  im_w = msg->width;
}


// Callback for RGB images
void cloud_cb_rgb (const sensor_msgs::ImageConstPtr& img_msg) {
  my_RGBimg_msg = *img_msg;
}


// Calback for Depth images
//   Compute proposals
//   Apply Jet color map
//   publish ros message containing proposals
void cloud_cb_depth (const sensor_msgs::ImageConstPtr& img_msg) {
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::TYPE_32FC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  Mat im_depth_jet;
  depth2jet(cv_ptr->image, im_depth_jet);

  Mat im_CV_16UC1;
  cv_ptr->image.convertTo(im_CV_16UC1, CV_16UC1);

  // Compute proposals and estimates the ground plane
  std::vector<float > proposals = get_proposals(cv_ptr->image);	

  // Ros message contains: 
  // RGB image, DepthJet image, intrinsic parameters and coefficients of the plane
  ::hospital_people_detector::Proposals_msg msg;

  sensor_msgs::Image depthjet_img_msg;

  cv_bridge::CvImage img_bridge;
  img_bridge.header   = img_msg->header;
  img_bridge.encoding = sensor_msgs::image_encodings::TYPE_8UC3; 
  img_bridge.image    = im_depth_jet; 

  img_bridge.toImageMsg(depthjet_img_msg);

  sensor_msgs::ImagePtr im_msg = img_bridge.toImageMsg();

  msg.img_jet = *im_msg;
  msg.img_rgb = my_RGBimg_msg;

  for(int i=0; i<proposals.size(); i++)
  {
    msg.boxes.push_back(proposals.at(i));
  }

  msg.fx = fx;
  msg.fy = fy;
  msg.cx = cx;
  msg.cy = cy;

  msg.coeff_a = coeff_a;
  msg.coeff_b = coeff_b;
  msg.coeff_c = coeff_c;
  msg.coeff_d = coeff_d;

  pub_proposals.publish(msg);
	
}


// Function to apply JET color mapping to Depth images
void depth2jet(Mat& input_depth, Mat& colored_depth) {
  Mat img_mask = Mat::zeros(input_depth.rows, input_depth.cols, CV_32F);
  float im_min = 50000;
  float im_max = 0; 
  for (size_t y = 0; y < input_depth.rows; y++) {
    for (size_t x = 0; x < input_depth.cols; x++) {
      if (input_depth.at<float>(y,x) == 0.0 ) {
        img_mask.at<float>(y,x) = 1.0;
      }
      // Calculate min and max but igore zero entries in image
      if (input_depth.at<float>(y,x) < im_min && input_depth.at<float>(y,x) > 0.0) {
        im_min = input_depth.at<float>(y,x);
      }
      if (input_depth.at<float>(y,x) > im_max) {
        im_max = input_depth.at<float>(y,x);
      }
    }
  }

  if (im_min <= 0.0) {
    ROS_ERROR ("Calculated min value in image is not greater than zero, but should be because we ignored zero entries %f", im_min);
  }
  Mat shifted_range_image = Mat::zeros(input_depth.rows, input_depth.cols, CV_32F);
  // Depth shifting and scaling between 0-255;
  float diff = 1.0f / (im_max - im_min);
  shifted_range_image = ( input_depth - im_min ) * diff;
  // Set values of img_mask to zero again
  for (size_t y = 0; y < img_mask.rows; y++) {
    for (size_t x = 0; x < img_mask.cols; x++) {
      if (img_mask.at<float>(y,x) == 1.0 ) {
        shifted_range_image.at<float>(y,x) = 0.0;
      }
    }
  }
  shifted_range_image = 255.0 * shifted_range_image;
  Mat im_CV_8UC1;
  shifted_range_image.convertTo(im_CV_8UC1, CV_8UC1);
  // Apply the colormap:
  applyColorMap(im_CV_8UC1, colored_depth, COLORMAP_JET);
}




// Function that generates proposals:
// Depth image is sampled and converted to a point cloud
// Ground plane estimation: it can be estimated using the point cloud or from TF
// We remove the ground plane and apply euclidian segmentation
// Apply sliding templates to generate proposals
std::vector<float > get_proposals(cv::Mat depthImage)
{
  //Proposals array
  std::vector<float > output_vec;

  //Return if there is no camera parameteres yet
  if (fx == 0 ){return output_vec;}

  //Convert image to point cloud
  //Sample every 4th pixel in the image to build the point cloud 
  int sampler = 4;  

  int width_im = depthImage.cols;
  int height_im = depthImage.rows;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
  cloud_in->width = int(width_im/sampler); 
  cloud_in->height = int(height_im/sampler); 
  cloud_in->is_dense = true; //all the data in cloud_in is finite (No Inf/NaN values)
  cloud_in->points.resize (cloud_in->width * cloud_in->height);

  // Factor to convert camera measuremets to meters
  float factor;

  if (Xtion_camera){
    factor = 1.0;  //Xtion camera
  }
  else{
    factor = 1000.0; //kinect camera
  }


  int i=0;
  for (int x=0; x<width_im; x+=sampler) {
    for (int y=0; y<height_im; y+=sampler) {

      float depth = (float)depthImage.at<float>(y, x) / factor;

      if (std::isnan(depth)) depth = INFINITY;
      if (depth==INFINITY) continue;

      float pclx = (x - cx ) / fx * depth;
      float pcly = (y - cy ) / fy * depth;

      if ( i <= cloud_in->points.size() )
      {
	     cloud_in->points[i].x = pclx;
	     cloud_in->points[i].y = pcly;
	     cloud_in->points[i].z = depth;
      }

      i++;

    }
  }


  //Voxel filter: Downsample the point cloud using a leaf size of 0.1mts
  double leaf_size = 0.1f;
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setLeafSize (leaf_size, leaf_size, leaf_size); 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud_in);
  vg.filter (*cloud_down);



  //Estimate ground plane using TF
  if (gp_source == "tf"){

    //Create points in base frame on the ground plane (z = o)
    float points_plane[][3] = { {1,1,0}, {1,-1,0}, {0,0,0}, {-1,1,0}, {-1,1,0} }; 
    int nr_points =  sizeof points_plane / sizeof points_plane[0];   

    //Create a small point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane_tf (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_plane_tf->width = nr_points; 
    cloud_plane_tf->height = 1; 
    cloud_plane_tf->is_dense = true; //all the data in cloud_in is finite (No Inf/NaN values)
    cloud_plane_tf->points.resize (cloud_plane_tf->width * cloud_plane_tf->height);

    //before filling the point cloud convert points to camera frame 
    geometry_msgs::PointStamped in_point, out_point;
    in_point.header.frame_id = base_frame;
    in_point.header.stamp = ros::Time();

    for (int i=0; i<nr_points; i++) {
      in_point.point.x = points_plane[i][0];
      in_point.point.y = points_plane[i][1];
      in_point.point.z = points_plane[i][2];
      try{
        pListener->transformPoint(camera_frame, in_point, out_point);
        cloud_plane_tf->points[i].x = out_point.point.x;
        cloud_plane_tf->points[i].y = out_point.point.y;
        cloud_plane_tf->points[i].z = out_point.point.z;
      }
      catch(tf::TransformException& ex){
        ROS_ERROR("Received: %s", ex.what());
        return output_vec;
      }
    }

    pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers2 (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg_plane;
    seg_plane.setOptimizeCoefficients (true);
    seg_plane.setModelType (pcl::SACMODEL_PLANE);
    seg_plane.setMethodType (pcl::SAC_RANSAC);
    seg_plane.setDistanceThreshold (0.05);

    // Get inliers of the predicted plane
    seg_plane.setInputCloud (cloud_plane_tf);

    // Estimate coefficients
    seg_plane.segment (*inliers2, *coefficients_plane);

    //std::cout  << "Model inliers: " << inliers2->indices.size() << std::endl;

    coeff_a = coefficients_plane->values[0];
    coeff_b = coefficients_plane->values[1];
    coeff_c = coefficients_plane->values[2];
    coeff_d = coefficients_plane->values[3];  


  } //End estimate ground plane using TF



  // Estimate ground plane from point cloud
  if(gp_source == "pcl"){

    // Keep points whose "y" coordinates are between "camera_height-above" and "camera_height+under"
    float above = 0.6; //meters above the expected ground plane
    float under = 0.2; //meters under the expected ground plane	
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud_down);
    pass.setFilterFieldName ("y"); 
    pass.setFilterLimits (camera_height-above, camera_height+under);
    pass.filter (*cloud_cropped);

    int min_cropped_cloud_points = 10;
    if (cloud_cropped->points.size() > min_cropped_cloud_points)
    {
      pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
      pcl::SACSegmentation<pcl::PointXYZ> seg;
      seg.setOptimizeCoefficients (true);
      seg.setModelType (pcl::SACMODEL_PLANE);
      seg.setMethodType (pcl::SAC_RANSAC);
      seg.setDistanceThreshold (0.05);

      // Get inliers of the predicted plane
      seg.setInputCloud (cloud_cropped);

      // Estimate coefficients
      seg.segment (*inliers, *coefficients);

      float plane_max_tolerance = 0.2; //meters
      int min_num_inliers = 40;	

      //estimated plane is valid only if it is close to the expected ground plane 
      if(inliers->indices.size() >= min_num_inliers && abs(coefficients->values[3]-camera_height) < plane_max_tolerance)
      {
	      coeff_a = coefficients->values[0];
        coeff_b = coefficients->values[1];
	      coeff_c = coefficients->values[2];
	      coeff_d = coefficients->values[3];  
      }
    }


    //std::cout << "coeff_a: " << coeff_a << std::endl;
    //std::cout << "coeff_b: " << coeff_b << std::endl;
    //std::cout << "coeff_c: " << coeff_c << std::endl;
    //std::cout << "coeff_d: " << coeff_d << std::endl;

  } //end compute plane from point cloud



  // Get index of points close to ground plane
  pcl::PointIndices index_cloud;
  for (size_t i = 0; i < cloud_down->points.size (); ++i)
  {
    double plane_y = -(coeff_a*cloud_down->points[i].x + coeff_c*cloud_down->points[i].z +coeff_d)/coeff_b;	
    // Keep points above the plane and under the ceiling (between 30cm - 2mts)
    if(cloud_down->points[i].y < plane_y-0.3 && cloud_down->points[i].y > plane_y-2){	
      index_cloud.indices.push_back(i);}
  }

  // Remove ground plane: Remove points close to ground plane  
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_without_plane (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ExtractIndices<pcl::PointXYZ> eifilter (true); 
  eifilter.setInputCloud (cloud_down);
  eifilter.setIndices (boost::make_shared<const pcl::PointIndices> (index_cloud));
  eifilter.filter (*cloud_without_plane);

  // Publish point cloud without ground plane
  pcl::PointCloud<pcl::PointXYZ>::Ptr msg_pcl (new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(*cloud_without_plane, *msg_pcl);
  msg_pcl->header.frame_id = camera_frame;
  msg_pcl->header.stamp = ros::Time().toNSec(); 
  pub_pcl.publish(msg_pcl);

  // Creating the KdTree object for segmentation
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_without_plane);

  // Apply segmentation
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.12); //0.12 points separated less than 12cm are part of same cluster  
  ec.setMinClusterSize (15); 
  ec.setMaxClusterSize (1300); 
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_without_plane);
  ec.extract (cluster_indices);


  // Evaluate each cluster
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
  {

    double cent_x = 0;
    double cent_z = 0;
    int count_cent = 0;

    //iterate over points in cluster to find centroid of cluster
    for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
    {  
      cent_x += cloud_without_plane->points[*pit].x; 
      cent_z += cloud_without_plane->points[*pit].z; 
      count_cent++;
    }

    cent_x /= count_cent;
    cent_z /= count_cent;

    double offset_z = 0;//0.3 //Generate bounding boxes 30cm closer to the camera
    double min_z = cent_z - offset_z; 

    double eval_x, y_ground_plane;
    int x1, x2, y1, y2, bb_x1, bb_x2, bb_y1, bb_y2; 

    //Compute y coordinate on the plane at (cent_x, min_z)
    y_ground_plane = -(coeff_a*cent_x + coeff_c*min_z +coeff_d)/coeff_b;

    //Start local sliding templates
    int stride = 0.1; //Slide windows 10cm to each side 
	
    //Generate a bounding box for an average pedestrian at point (eval_x, y_plane, min_z) 
    //top-left (bb_x1, bb_y1)   bottom-right (bb_x2, bb_y2)
    bb_y2 = y_ground_plane * fy/(min_z) + cy; //y
    bb_y1 = (y_ground_plane - av_human_h) * fy/(min_z) + cy;

    for(int kk=-1; kk<=1; kk++)
    {
      eval_x = cent_x + kk*stride;
      bb_x1 = (eval_x-av_human_w/2) * fx/(min_z) + cx;
      bb_x2 = (eval_x+av_human_w/2) * fx/(min_z) + cx;

      for(int ii=0; ii<=2; ii++)
      {
    	  for(int jj=0; jj<=1; jj++)
	      {
	        x1 = bb_x1 - ii*abs(bb_x1 - bb_x2)/3;
          x2 = bb_x2 + ii*abs(bb_x1 - bb_x2)/3;
	        y1 = bb_y1 + jj*abs(bb_y1 - bb_y2)/4;
	        y2 = bb_y2;

          if(x1<0){x1=0;}
      	  if(x1>width_im){x1=width_im;}
      	  if(x2<0){x2=0;}
      	  if(x2>width_im){x2=width_im;}
      	  if(y1<0){y1=0;}
      	  if(y1>height_im){y1=height_im;}
      	  if(y2<0){y2=0;}
      	  if(y2>height_im){y2=height_im;}

      	  //[x1 y1 x2 y2 depth x1 y1 x2 ..... ]
      	  if(ii==2 && jj ==1){
      	  }
      	  else{
      	    output_vec.push_back(x1);		
      	    output_vec.push_back(y1);		
      	    output_vec.push_back(x2);		
      	    output_vec.push_back(y2);
      	    output_vec.push_back(cent_z);
      	  }

        }

      }

    }

  }

  return output_vec;
}



int main (int argc, char** argv)
{
  ros::init (argc, argv, "proposals_generation");
  //ros::NodeHandle nh;

  ROS_INFO("Proposals generation node started");

  pnh = new(ros::NodeHandle);

  pnh->getParam("ground_plane_source", gp_source);
  pnh->getParam("camera_height", camera_height);
  pnh->getParam("rgb_image_topic", rgb_image_topic);
  pnh->getParam("depth_image_topic", depth_image_topic);
  pnh->getParam("camera_info_topic", camera_info_topic);
  pnh->getParam("camera_frame", camera_frame);
  pnh->getParam("base_frame", base_frame);
  pnh->getParam("Xtion_camera", Xtion_camera);

  coeff_a = 0;  
  coeff_b = -1;
  coeff_c = -0.1;  
  coeff_d = camera_height; 

  ros::Subscriber sub_depth = pnh->subscribe (depth_image_topic, 1, cloud_cb_depth);
  ros::Subscriber sub_rgb = pnh->subscribe (rgb_image_topic, 1, cloud_cb_rgb);
  ros::Subscriber sub_camera_info = pnh->subscribe(camera_info_topic, 1, cb_camera_info);

  pub_proposals = pnh->advertise<hospital_people_detector::Proposals_msg>("topic_proposals", 1); //("topic_proposals", 1);

  //typedef pcl::PointCloud<pcl::PointXYZ> PCLCloud;
  //pub_pcl = nh.advertise<PCLCloud>("mobility_pcl", 1);
  pub_pcl = pnh->advertise< pcl::PointCloud<pcl::PointXYZ> >("mobility_pcl", 1);

  pListener = new (tf::TransformListener);


  ros::spin ();
}


