#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"  
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <stdio.h>
#include "manager.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include "detectobs/code.h"
#include "detectobs/bbox.h"
#include <time.h> 
#include "yolo/yolov5_lib.h"
#define targetid 1
using namespace std;
cv::Mat imagegot;
cv::Mat depthgot;
float cinfo[4];

char* yolo_engine = (char*)"/home/unitree/colabot/src/detectobs/engine/best.engine";

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
/*calculate depth through 3 reference points*/
float depth_get(int x1,int y1,int x2,int y2){
  float x_ref1=0.5*x1+0.5*x2; //point 1
  float x_ref2=0.6*x1+0.4*x2; //point 2
  float x_ref3=0.4*x1+0.6*x2; //point 3
  float y_ref123=0.8*y1+0.2*y2;
  float dep_ref1=depthgot.at<ushort>((int)y_ref123,(int)x_ref1);
  float dep_ref2=depthgot.at<ushort>((int)y_ref123,(int)x_ref2);
  float dep_ref3=depthgot.at<ushort>((int)y_ref123,(int)x_ref3);
  float point_z=(dep_ref1+dep_ref2+dep_ref3)/3;
  return point_z;
}

int main(int argc, char **argv){
    Trtyolosort yosort(yolo_engine);
    std::vector<DetectBox> boxes;
    ros::Time::init();
  ros::Rate loop_rate(20);
  float t=0.4;
  //time_t start,finish,last_finish;
  ros::Time finish=ros::Time::now();
  ros::Time last_finish=ros::Time::now();
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub_rgb = it.subscribe("/up_camera/color/image_raw", 1, imageCallback);

  image_transport::ImageTransport dt(nh);
  image_transport::Subscriber sub_dth = dt.subscribe("/up_camera/depth/image_rect_raw", 1, depthCallback);

  ros::NodeHandle cinfoget;
  ros::Subscriber sub_cinfo = cinfoget.subscribe("/up_camera/depth/camera_info", 1, cinfoCallback);

  ros::NodeHandle dopub;
  ros::Publisher point_pub = dopub.advertise<detectobs::code>("Target_pos",1);

  ros::NodeHandle imgpub;
  ros::Publisher img_pub = imgpub.advertise<sensor_msgs::Image>("img_with_bbox",1);

  ros::NodeHandle bboxpub;
  ros::Publisher bbox_pub = imgpub.advertise<detectobs::bbox>("bbox_x1y1x2y2",1);

int cout = 0;
float fps = 0;
int id = 0;
while(1){
    cout++;
    cv::Mat tempimage = imagegot;
    if(tempimage.channels()>1){
        bool Isobjfound=0;
        yosort.TrtDetect(tempimage,t,boxes);
        sensor_msgs::ImagePtr msg_img_with_bbox = cv_bridge::CvImage(std_msgs::Header(), "bgr8", tempimage).toImageMsg();
        img_pub.publish(msg_img_with_bbox);
        detectobs::code point;
        detectobs::bbox bbox;
        id = 0;
        if(boxes.size()>0){
            float point_x=0,point_y=0;
            for (auto box : boxes){
                if(box.classID==0){
                    if (id==targetid){
                    Isobjfound=1;
                    bbox.x1=box.x1;
                    bbox.y1=box.y1;
                    bbox.x2=box.x2;
                    bbox.y2=box.y2;
                    point_x=(box.x1+box.x2)/2;
                    point_y=(box.y1+box.y2)/2;
                    int x1=(int)box.x1,y1=(int)box.y1,x2=(int)box.x2,y2=(int)box.y2;
                    float point_z=depth_get(x1,y1,x2,y2);
                    point_z*=0.001;
                    point_x=(point_x-cinfo[2])/cinfo[0]*point_z;
                    point_y=-(point_y-cinfo[3])/cinfo[1]*point_z;



                    point.flag=1;
                    point.x=point_x;
                    point.y=point_y;
                    point.z=point_z;
                      if(point.z==0) {
                    point.flag=0;
                    point.z=-1;
                    }
                    }
                
                    id++;
                }
            }
        }
        else{
        bbox.x1=-1;
        bbox.x2=-1;
        bbox.y1=-1;
        bbox.y2=-1;
        point.flag=0;
        point.x=0;
        point.y=0;
        point.z=-1;
      }
      point_pub.publish(point);
      bbox_pub.publish(bbox);
      fps=1/((finish-last_finish).toSec());
      
      if(point.flag==1) std::cout<<"++++Detected. x: "<<point.x<<" y: "<<point.y<<" z: "<<point.z<<" fps: "<<fps<<std::endl;
      else std::cout<<"----Detecting...----"<<" fps: "<<fps<<std::endl;
      //start+=(finish-last_finish);
      last_finish=finish;
      finish=ros::Time::now();
    }
    else{
      ros::Duration(0.5).sleep();
      std::cout<<"camera loading..."<<std::endl;
    }
    ros::spinOnce();
    loop_rate.sleep();
}
}
