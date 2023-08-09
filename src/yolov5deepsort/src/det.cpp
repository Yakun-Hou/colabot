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
cv::Mat depthgot;
float cinfo[4];

char* yolo_engine = "/home/unitree/priv_sqx/yolov5-deepsort-tensorrt-main/engine/yolov5s.engine";
char* sort_engine = "/home/unitree/priv_sqx/yolov5-deepsort-tensorrt-main/engine/deepsort.engine";



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

void depthCallback(const sensor_msgs::ImageConstPtr& msg){
  try
  {   
    depthgot=cv_bridge::toCvCopy(msg, "16UC1")->image;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to '16UC1'.", msg->encoding.c_str());
  }
}

void cinfoCallback(const sensor_msgs::CameraInfoConstPtr& msg){
  //std::cout<<msg;
    
  cinfo[0]=((*msg).K)[0];
  cinfo[1]=((*msg).K)[4];
  cinfo[2]=((*msg).K)[2];
  cinfo[3]=((*msg).K)[5];
    

}
int main(int argc, char **argv)
{
  Trtyolosort yosort(yolo_engine,sort_engine);
  std::vector<DetectBox> boxes;
  ros::Time::init();
  ros::Rate loop_rate(20);
  float t=0.4;
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub_rgb = it.subscribe("/camera/color/image_raw", 1, imageCallback);

  image_transport::ImageTransport dt(nh);
  image_transport::Subscriber sub_dth = dt.subscribe("/camera/aligned_depth_to_color/image_raw", 1, depthCallback);

  ros::NodeHandle cinfoget;
  ros::Subscriber sub_cinfo = cinfoget.subscribe("/camera/aligned_depth_to_color/camera_info", 1, cinfoCallback);

  ros::NodeHandle dopub;
  ros::Publisher point_pub = dopub.advertise<yolov5deepsort::code>("Target_pos",1);

  
  int cont=0;
  while(cont<10000){
    cont++;
    cv::Mat tempimage=imagegot;
    if(tempimage.channels()>1){ 
      bool Isobjfound=0;
      yosort.TrtDetect(tempimage,t,boxes,0);
      yolov5deepsort::code point;
      if(boxes.size()>0){
        float point_x=0,point_y=0;
        //ushort point_z=0;
        for (auto box : boxes){
          if((int)box.classID==67||(int)box.classID==77||(int)box.classID==87||(int)box.classID==76){
            Isobjfound=1;
            point_x=(box.x1+box.x2)/2;
            point_y=(box.y1+box.y2)/2;
            float point_z=depthgot.at<ushort>((int)point_y,(int)point_x);
            //cout<<point_x<<" "<<point_y<<" "<<point_z<<endl;
            
            
            point_z*=0.001;
            point_x=(point_x-cinfo[2])/cinfo[0]*point_z;
            point_y=-(point_y-cinfo[3])/cinfo[1]*point_z;
            //QR_code_detector::code point;
            point.flag=1;
            point.x=point_x;
            point.y=point_y;
            point.z=point_z;
            if(point.z==0) {
              point.flag=0;
              point.z=-1;
            }

          }
        }
      }
      else{
        
        point.flag=0;
        point.x=0;
        point.y=0;
        point.z=-1;
      }
      point_pub.publish(point);
      if(point.flag==1) cout<<"++++Detected. x: "<<point.x<<" y: "<<point.y<<" z: "<<point.z<<"++++"<<endl;
      else cout<<"----Detecting...----"<<endl;

      

    }
    else{
      ros::Duration(0.5).sleep();
      std::cout<<"camera loading..."<<std::endl;
    }

    
    //cout<<tempimage.channels();
    ros::spinOnce();
    loop_rate.sleep();
  }
  
}
