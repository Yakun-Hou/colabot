#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Point.h>
#include "QR_code_detector/code.h"

int main(int argc, char *argv[])
{

    //1.初始化 ROS 节点
    ros::init(argc,argv,"test_my");

    //2.创建 ROS 句柄
    ros::NodeHandle nh;

    //3.创建发布者对象
    ros::Publisher pub = nh.advertise<QR_code_detector::code>("QR_code", 10);

    //4.组织被发布的消息，编写发布逻辑并发布消息
    QR_code_detector::code c;
    c.flag = true;
    c.x = 1.0;
    c.y = 2.0;
    c.z = 3.0;

    ros::Rate r(1);
    while (ros::ok())
    {
        pub.publish(c);
        c.z += 1;
        ROS_INFO("%.5f, %.5f, %.5f", c.x, c.y, c.z);

        r.sleep();
        ros::spinOnce();
    }



    return 0;
}
