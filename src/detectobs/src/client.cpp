#include "ros/ros.h"
#include<stdio.h>
#include "detectobs/sender.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"  
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
    setlocale(LC_ALL,"");
    ros::init(argc,argv,"Client");
    ros::NodeHandle nh;
    ros::ServiceClient client = nh.serviceClient<detectobs::sender>("sender");
    ros::NodeHandle imgpub;
    ros::Publisher img_pub = imgpub.advertise<sensor_msgs::Image>("img_rgb_test",1);
    ros::NodeHandle dptpub;
    ros::Publisher dpt_pub = dptpub.advertise<sensor_msgs::Image>("img_dpt_test",1);
    ros::NodeHandle bboxpub;
    ros::Publisher bbox_pub = imgpub.advertise<detectobs::bbox>("bbox_test",1);
    ros::service::waitForService("sender");
    detectobs::sender ai;
    int target=0;
    int count=0;
    while(true){
        printf("请输入跟踪目标的ID:\n");
        while(scanf("%d",&target)!=1){
            printf("格式错误，请重新输入:\n");
            count++;
            if(count>=10)
            {
                printf("多次输错目标,程序中断");
                break;
            }
        }
        if(count>=10)
            break;
        while(getchar()!='\n'){
            continue;
        }
        ai.request.num = target;
        bool flag = client.call(ai);
        if (flag)
        {
            printf("x:%g y:%g z:%g",ai.response.point.x,ai.response.point.y,ai.response.point.z);
            // int times=0;
            while(1){
                // times++;
                img_pub.publish(ai.response.image_with_bbox);
                dpt_pub.publish(ai.response.depth);
                bbox_pub.publish(ai.response.bbox);
            }
        }
        else
        {
            ROS_ERROR("请求处理失败....");
        }
    }
    return 0;
}