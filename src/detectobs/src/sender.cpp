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
#include "detectobs/sender.h"
using namespace std;
// typedef struct humanboxes {
//     humanboxes(float x1=0, float y1=0, float x2=0, float y2=0) {
//         this->x1 = x1;
//         this->y1 = y1;
//         this->x2 = x2;
//         this->y2 = y2;
//     }
//     float x1, y1, x2, y2;
// } humanboxes;
cv::Mat imagegot;
cv::Mat depthgot;
float cinfo[4];
int num;
detectobs::code point;
detectobs::bbox bbox;
sensor_msgs::ImagePtr msg_img_with_bbox;
sensor_msgs::ImagePtr depth_img;
// std::vector<humanboxes> humanbox;
std::vector<DetectBox> humanbox;


bool test=false;

char* yolo_engine = (char*)"/home/unitree/test_dog/dog_3/src/detectobs/engine/best.engine";

void imageCallback(const sensor_msgs::ImageConstPtr& msg){
  //printf("imageCallback\n");
  test=true;
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
    //printf("depthCallback\n");

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
    //printf("cinfoCallback\n");

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

bool doReq(detectobs::sender::Request& req, detectobs::sender::Response& resp){
    int num = req.num-1;

    ROS_INFO("server has receved:num = %d",num+1);
    if (humanbox.size()==0){
        ROS_INFO("no target found");
        return false;
    }
    else{
        if (num>humanbox.size()){
            ROS_ERROR("num is out of range");
            return false;
        }
        float point_x=0,point_y=0;
        point_x=(humanbox[num].x1+humanbox[num].x2)/2;
        point_y = (humanbox[num].y1+humanbox[num].y2)/2;
        int x1=(int)humanbox[num].x1,y1=(int)humanbox[num].y1,x2=(int)humanbox[num].x2,y2=(int)humanbox[num].y2;
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
    resp.image_with_bbox = *msg_img_with_bbox;
    resp.depth = *depth_img;
    resp.point = point;
    // resp.bbox = bbox;
    resp.bbox.x1 = humanbox[num].x1;
    resp.bbox.x2 = humanbox[num].x2;
    resp.bbox.y1 = humanbox[num].y1;
    resp.bbox.y2 = humanbox[num].y2;

    return true;
    //缺少返回值
}

int main(int argc, char **argv){
    Trtyolosort yosort(yolo_engine);
    std::vector<DetectBox> boxes;
    ros::Time::init();
    ros::Rate loop_rate(20);
    float t=0.4;
    ros::init(argc, argv, "sender");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub_rgb = it.subscribe("/camera/color/image_raw", 1, imageCallback);

    image_transport::ImageTransport dt(nh);
    image_transport::Subscriber sub_dth = dt.subscribe("/camera/aligned_depth_to_color/image_raw", 1, depthCallback);

    ros::NodeHandle cinfoget;
    ros::Subscriber sub_cinfo = cinfoget.subscribe("/camera/aligned_depth_to_color/camera_info", 1, cinfoCallback);
    printf("是否开始选择跟踪目标，按任意键开始检测\n");
    getchar();
    while(1){
      ros::spinOnce();
      if(test){
        break;
      }
    }
    cv::Mat tempimage = imagegot;
      if(tempimage.channels()>1){
        yosort.TrtDetect(tempimage,t,boxes);
        printf("detecting...\n");
        sensor_msgs::ImagePtr msg_img_with_bbox_local = cv_bridge::CvImage(std_msgs::Header(), "bgr8", tempimage).toImageMsg();
        sensor_msgs::ImagePtr depth_img_local = cv_bridge::CvImage(std_msgs::Header(), "16UC1", depthgot).toImageMsg();
        msg_img_with_bbox = msg_img_with_bbox_local;
        // cout << "size:" << tempimage.size() << endl;
        depth_img = depth_img_local;
        struct DetectBox bbox_human;
        int i = 0;
        if(boxes.size()>0){
            for (auto box : boxes){
                if(box.classID==0){
                    i++;
                    bbox_human.classID=box.classID;
                    bbox_human.x1=int(box.x1*0.6);
                    if (bbox_human.x1<0){
                        bbox_human.x1=0;
                    }
                    if (bbox_human.x1>768){
                        bbox_human.x1=768;
                    }
                    bbox_human.y1=int(box.y1*0.6);
                       if (bbox_human.y1<0){
                        bbox_human.y1=0;
                    }
                     if (bbox_human.y1>432){
                        bbox_human.y1=432;
                    }
                    bbox_human.x2=int(box.x2*0.6);
                     if (bbox_human.x2<0){
                        bbox_human.x2=0;
                    }
                     if (bbox_human.x2>768){
                        bbox_human.x2=768;
                    }
                    bbox_human.y2=int(box.y2*0.6);
                     if (bbox_human.y2<0){
                        bbox_human.y2=0;
                    }
                     if (bbox_human.y2>432){
                        bbox_human.y2=432;
                    }
                    humanbox.push_back(bbox_human);
                    // humanbox_detected.push_back(box);
                    printf("Id: %d x1: %g x2: %g y1: %g y2: %g\n",i,bbox_human.x1,bbox_human.x2,bbox_human.y1,bbox_human.y2);
                }
            }
          if (humanbox.size()==0){
              ROS_INFO("no target found\n");
          }
        }
      }
    else if(tempimage.channels()==1){
        printf("Depth\n");
        cout << "size:" << tempimage.size() << endl;
    }
    else{
        printf("error\n");
    }
    
    ros::ServiceServer server = nh.advertiseService("sender",doReq);
    ROS_INFO("service is ready\n");
    ros::spin();
}


  
