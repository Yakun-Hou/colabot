#include <math.h>
#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include "detectobs/code.h"
#include "std_msgs/String.h"
#include <sensor_msgs/LaserScan.h>

#define EPS 0.3 //缓冲区
#define distance 1.0 //保持距离
#define memory_capacity 7 //记录过往位置的数量
#define time_for_waiting 10 //丢失目标的等待帧数
#define limite_for_depth 7.0 //深度相机精度极限
struct position
{
    bool flag;
    bool reachable;
    float x;
    float y;
    float z;
};
struct last_position
{
    bool flag;
    float x;
    float y;
    float z;
};

struct position pos={false,true,0.0,0.0,0.0};        //复制消息
struct last_position last_pos={false,0.0,0.0,0.0};   //过往检测到的位置（均值）
struct last_position pos_ensemble[memory_capacity];	//过往位置（集合）
detectobs::code target;                      //发送给导航的目标
detectobs::code target_1;                    //发送目标，让导航停止
geometry_msgs::Twist T;                             //发送给控制端口的速度
int count=0;					      //计算丢失目标帧数
bool navigating=false;				      //判断是否处于导航状态
bool goal_reach=false;				      //判断目标丢失后是否到达丢失位置
float laser_distance=0.0;
void doMSG(detectobs::code p)
{
    pos.flag = p.flag;
    //pos.reachable = p.reachable;
    pos.x=p.z;
    pos.y=-p.x;
    pos.z=0.0;
}

void doMSG1(std_msgs::String a){
    printf("已到达目标丢失的位置\n");
    goal_reach=true;
}

void doMSG2(sensor_msgs::LaserScan a){
    laser_distance=1000.0;
    int theta=180;
    if(pos.x>limite_for_depth){
        theta=180-(int)round(atan(pos.y/limite_for_depth)/M_PI*180.0);
    }
    else{
        theta=180-(int)round(atan(pos.y/pos.x)/M_PI*180.0);
    }
    int i=0;
    for(i=theta-15;i<theta+15;i++){
        if(a.ranges[i]<laser_distance && a.ranges[i]>0.3){
            laser_distance=a.ranges[i];
        }
    }
}

void memory(struct position p)
{
    int i=0;
    //更新过往位置
    if(p.flag && p.x<limite_for_depth)
    {
    	for(i=0;i<memory_capacity-1;i++)
    	{
    	    pos_ensemble[i].flag=pos_ensemble[i+1].flag;
    	    pos_ensemble[i].x = pos_ensemble[i+1].x;
    	    pos_ensemble[i].y = pos_ensemble[i+1].y;
    	    pos_ensemble[i].z = pos_ensemble[i+1].z;
    	}
    	pos_ensemble[memory_capacity-1].flag=p.flag;
    	pos_ensemble[memory_capacity-1].x = p.x;
    	pos_ensemble[memory_capacity-1].y = p.y;
    	pos_ensemble[memory_capacity-1].z = p.z;
    }
    
    //计算过往位置均值
    if(!last_pos.flag)
    {
    	for(i=0;i<memory_capacity;i++)
    	{
    	    if(pos_ensemble[i].flag)
    	    {
    	        last_pos.flag=true;
    	        break;
    	    }
    	    else
    	    {
                continue;
    	    }
    	}
    }
    int num=0;

    float sum_x=0.0;
    float sum_y=0.0;
    float sum_z=0.0;

    float max_x=0.0;
    float max_y=0.0;
    float max_z=0.0;

    float min_x=1000.0;
    float min_y=1000.0;
    float min_z=1000.0;
    
    for(i=0;i<memory_capacity;i++)
    {
        if(pos_ensemble[i].flag)
        {
            num+=1;
            sum_x=sum_x+pos_ensemble[i].x;
            sum_y=sum_y+pos_ensemble[i].y;
            sum_z=sum_z+pos_ensemble[i].z;
            if(pos_ensemble[i].x>max_x)
            {
                max_x=pos_ensemble[i].x;
            }
            if(pos_ensemble[i].y>max_y)
            {
                max_y=pos_ensemble[i].y;
            }
            if(pos_ensemble[i].z>max_z)
            {
                max_z=pos_ensemble[i].z;
            }
            if(pos_ensemble[i].x<min_x)
            {
                min_x=pos_ensemble[i].x;
            }
            if(pos_ensemble[i].y<min_y)
            {
                min_y=pos_ensemble[i].y;
            }
            if(pos_ensemble[i].z<min_z)
            {
                min_z=pos_ensemble[i].z;
            }
        }

        if(num>=3)
        {
            last_pos.x=(sum_x-max_x-min_x)/(num-2);
            last_pos.y=(sum_y-max_y-min_y)/(num-2);
            last_pos.z=(sum_z-max_z-min_z)/(num-2);
        }
        else if(num<3&&num>0)
        {
            last_pos.x=sum_x/num;
            last_pos.y=sum_y/num;
            last_pos.z=sum_z/num;
        }
        else
        {
            continue;
        }
    }
}

int Agent(struct position p) //返回0为走直线，返回1为目标不可达导航,返回2为丢失导航
{
    if(p.flag)			//第一个分支,检测到
    {
        count=0;		//检测到立刻清空丢失目标计算时间
        if(p.reachable)      //第二分支，判断可不可达，可达用直线
        {
            if(abs(p.x-distance)>EPS)
            {
                T.linear.x=(p.x-distance)/abs(p.x-distance)*0.45;
            }
            else
            {
                T.linear.x=0.0;
            }
            if(abs(p.y)>EPS)
            {
                T.angular.z=p.y/abs(p.y)*0.15;
            }
            else
            {
                T.angular.z=0.0;
            }
            return 0;
        }
        else			//不可达用导航，计算发给move base的位置，使用过往位置的均值，防止当前深度检测错误
        {
            if(p.x>limite_for_depth)
            {
                target.x=last_pos.x;
                target.y=last_pos.y;
                target.z=0.0;
            }
            else
            {
                target.x=p.x;
                target.y=p.y;
                target.z=0.0;
            }
            return 1;
        }
    }
    else			//第一分支，未能检测到
    {
        if(!last_pos.flag)      //第二分支，判断初始化，初始化静止
        {
            T.linear.x=0.0;
            T.angular.z=0.0;
            return 0;
        }
        else			//上一帧检测到，丢失目标，启动导航
        {
            count++;		//记数，time_for_waiting外导航
            if(count>=time_for_waiting)       //time_for_waiting内没检测到目标，导航至上次检测的位置
            {
                target.x=last_pos.x;
                target.y=last_pos.y;
                target.z=0.0;
                return 2;
            }
            else		//time_for_waiting内，静止
            {
                T.linear.x=0.0;
                T.angular.z=0.0;
                return 0;
            }
        }
    }
}

int main(int argc,char *argv[]) 
{
    ros::init(argc,argv,"behavior_tree");
    ros::NodeHandle nh;

    int index;
    bool lost_target_goal_sent=false;

    T.linear.x=0.0;
    T.linear.y=0.0;
    T.linear.z=0.0;
    T.angular.x=0.0;
    T.angular.y=0.0;
    T.angular.z=0.0;

    target.x=0.0;
    target.y=0.0;
    target.z=0.0;

    target_1.x=0.0;
    target_1.y=0.0;
    target_1.z=0.0;
    int j=0;
    for(j=0;j<memory_capacity;j++)
    {
        pos_ensemble[j].flag=false;
        pos_ensemble[j].x=0.0;
        pos_ensemble[j].y=0.0;
        pos_ensemble[j].z=0.0;
    }

    ros::Subscriber sub = nh.subscribe<detectobs::code>("Target_pos",1,doMSG);		//订阅检测消息
    ros::Subscriber sub_1 = nh.subscribe<std_msgs::String>("goal_reach",1,doMSG1);			//订阅是否到达导航目标消息
    ros::Subscriber sub_2 = nh.subscribe<sensor_msgs::LaserScan>("slamware_ros_sdk_server_node/scan",1,doMSG2);     //订阅雷达消息
    ros::Publisher pub_1 = nh.advertise<detectobs::code>("target_pos",1);				//发送给导航模块
    ros::Publisher pub_2 = nh.advertise<geometry_msgs::Twist>("cmd_vel",1);				//发送速度给控制端
    ros::Rate rate(10);
    while(ros::ok())
    {
        ros::spinOnce();			//回调，更新pos
        if(pos.x>limite_for_depth)
        {
            pos.x=laser_distance;
        }
        memory(pos);				//更新memory
        index=Agent(pos);			//Agent决策
        if(index==0)				//0，代表走直线，发送速度
        {
            if(navigating)			//判断是否处于导航状态，如果是，立刻发送自身位置为目标点，让机器狗暂停导航
            {
                pub_1.publish(target_1);
                ros::Duration(1.0).sleep();	//给一秒的暂停时间
                navigating=false;
            }
            pub_2.publish(T);
            lost_target_goal_sent=false;
            goal_reach=false;
        }
        else if(index==1)      		//1，代表目标不可达，启动导航模块
        {
            navigating=true;
            pub_1.publish(target);
            ros::Duration(3.0).sleep();	//给3秒的运行时间
            lost_target_goal_sent=false;
            goal_reach=false;
        }
        else					//2，代表目标丢失，启动导航模块
        {
            navigating=true;
            if(!lost_target_goal_sent)
            {
                pub_1.publish(target);
                printf("目标丢失，尝试去上一次检测到的位置\n"); 
                lost_target_goal_sent=true;
            }
            if(goal_reach){
                T.linear.x=0.0;
                T.angular.z=0.3;
                pub_2.publish(T);
                ros::Duration(1.0).sleep();
                T.angular.z=0.0;
                pub_2.publish(T);
                ros::Duration(1.0).sleep();
            }
        }
        rate.sleep();
    }
    return 0;
}
