#pragma once

#include <ros/ros.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

#include <pcl/filters/voxel_grid.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/sample_consensus/model_types.h>   
#include <pcl/sample_consensus/method_types.h>   
#include <pcl/segmentation/sac_segmentation.h> 

#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

class PclTestCore
{

  private:
    ros::Subscriber sub_point_cloud_;

    ros::Publisher pub_filtered_points_;

    void point_cb(const sensor_msgs::PointCloud2ConstPtr& in_cloud);

  public:
    PclTestCore(ros::NodeHandle &nh);
    ~PclTestCore();
    void Spin();
};
