#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include "manager.hpp"
#include "yolov5deepsort/code.h"

cv::Mat imagegot;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
try
  {   
    imagegot=cv_bridge::toCvShare(msg, "bgr8")->image;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv){
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub_rgb = it.subscribe("/camera/color/image_raw", 1, imageCallback);
    int numimg=3001;
    while(numimg<=3500){
        cv::Mat imtmp=imagegot;
        if(imtmp.channels()>1){ 
            cv::imwrite("/home/unitree/training_img/4/"+to_string(numimg)+".jpg",imtmp);
            numimg++;
            ros::Duration(0.5).sleep();
        }
        else{
            cout<<"loading..."<<std::endl;
            ros::Duration(1).sleep();
        }
        ros::spinOnce();
    }
    return 0;
}