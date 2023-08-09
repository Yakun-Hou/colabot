#include "pcl_costmap/pcl_test_core.h"


PclTestCore::PclTestCore(ros::NodeHandle &nh){
    sub_point_cloud_ = nh.subscribe("/camera/depth/points",10, &PclTestCore::point_cb, this);

    pub_filtered_points_ = nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 10);

    ros::spin();
}

PclTestCore::~PclTestCore(){}

void PclTestCore::Spin(){
    
}

void PclTestCore::point_cb(const sensor_msgs::PointCloud2ConstPtr & in_cloud_ptr){
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZI>);


    pcl::fromROSMsg(*in_cloud_ptr, *current_pc_ptr);

    pcl::VoxelGrid<pcl::PointXYZI> vg;

    vg.setInputCloud(current_pc_ptr);
    vg.setLeafSize(0.2f, 0.2f, 0.2f); //对0.2m*0.2m*0.2m的正方体作一次采样
    vg.filter(*filtered_pc_ptr);

    pcl::SACSegmentation<pcl::PointXYZI> seg;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients(true);
    seg.setModelType (pcl::SACMODEL_PLANE);   
    seg.setMethodType (pcl::SAC_RANSAC);    
	seg.setDistanceThreshold(0.001); //定义内点到模型内允许的最大值
 
    seg.setInputCloud (filtered_pc_ptr);   //设置输入的点云
    seg.segment (*inliers,*coefficients);  

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud (filtered_pc_ptr);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (*cloud_f);

    sensor_msgs::PointCloud2 pub_pc;
    pcl::toROSMsg(*cloud_f, pub_pc);

    pub_pc.header = in_cloud_ptr->header;

    pub_filtered_points_.publish(pub_pc);
}

