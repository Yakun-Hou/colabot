#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Point.h>
#include "QR_code_detector/code.h"

#define Stop 1
#define Move 2
#define Rotate 3

#define depDis 0.3
#define horDis 0.3

// global parameter
geometry_msgs::Twist twist;
bool seeFlag = false;
QR_code_detector::code qr;

void seeCallback(const QR_code_detector::code::ConstPtr &loc) {
    // ROS_INFO("HERE");
    if (loc->flag) {
        seeFlag = true;
        qr.flag = loc->flag;
        qr.x = loc->x;
        qr.y = loc->y;
        qr.z = loc->z;
        ROS_INFO("QR found at x:%.5f, y:%.5f, z:%.5f", qr.x, qr.y, qr.z);
    } else {
        ROS_INFO("No QR found");
    } 
}

void receiver(ros::NodeHandle nh) {
    int cnt = 0;
    seeFlag = false;
    ros::Rate rec(60);
    
    while (cnt < 5) {
        // ROS_INFO("HERE");
        ros::Subscriber sub = nh.subscribe<QR_code_detector::code>("QR_code", 10, seeCallback);
        cnt++;
        rec.sleep();
        ros::spinOnce();
    }
}

int decoder(ros::NodeHandle nh) {
    receiver(nh);
    // QR is not found
    if (!seeFlag) {
        return Rotate;
    } else {
        // distance is far away
        if (qr.z > depDis + 0.05) {
            twist.linear.x = 0.5;
            // is on the left side
            if (qr.x < -horDis) {
                twist.angular.z = 0.5;
            // is on the right side
            } else if (qr.x > horDis) {
                twist.angular.z = -0.5;
            } else {
                twist.angular.z = 0;
            }
            return Move;
        // distance is too close
        } else if (qr.z < depDis - 0.05) {
            twist.linear.x = -0.5;
            // is on the left side
            if (qr.x < -horDis) {
                twist.angular.z = 0.5;
            //is on the right side
            } else if (qr.x > horDis) {
                twist.angular.z = -0.5;
            } else {
                twist.angular.z = 0;
            }
            return Move;
        } else {
            twist.linear.x = 0;
            // is on the left side
            if (qr.x < -horDis) {
                twist.angular.z = 0.5;
                return Move;
                // is on the right side
            } else if (qr.x > horDis) {
                twist.angular.z = -0.5;
                return Move;
            } else {
                return Stop;
            }
        } 
    }
}

bool stop(ros::NodeHandle nh) {
    ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("cmd_vel",10);

    twist.linear.x = 0;
    twist.linear.y = 0;
    twist.linear.z = 0;
    twist.angular.x = 0;
    twist.angular.y = 0;
    twist.angular.z = 0;
    
    pub.publish(twist);
    ROS_INFO(" [Robot stopped] ");
    return true;
}

bool move(ros::NodeHandle nh) {
    ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("cmd_vel",10);

    twist.linear.y = 0;
    twist.linear.z = 0;
    twist.angular.x = 0;
    twist.angular.y = 0;

    pub.publish(twist);
    ROS_INFO(" [Robot is moving] ");
    return true;
}

bool rotate(ros::NodeHandle nh) {
    ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("cmd_vel",10);

    twist.linear.x = 0;
    twist.linear.y = 0;
    twist.linear.z = 0;
    twist.angular.x = 0;
    twist.angular.y = 0;
    twist.angular.z = 0.5;
    
    pub.publish(twist);
    ROS_INFO(" [Robot is rotating to find a target] ");
    return true;
}

int main(int argc, char *argv[]) {
    using namespace std;

    ros::init(argc,argv,"FSM");
    ros::NodeHandle nh;

    int cnt = 0;
    while (ros::ok()) {
    // while (cnt <= 10000) {
        int command = decoder(nh);
        bool status = false;

        switch (command) {
            case Stop:
                status = stop(nh);
                break;
            case Move:
                status = move(nh);
                break;
            case Rotate:
                status = rotate(nh);
                break;
            default:
                ROS_INFO("What the fuck!");
        }

        // ROS_INFO("Running in the epoch [%d], the status is [%d]", cnt, status ? 1 : 0);
        cnt++;
        // r.sleep();
    }
}