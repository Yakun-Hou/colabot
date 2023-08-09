1.需要安装ros_pcl
 sudo apt install ros-melodic-pcl-conversions ros-melodic-pcl-ros

2.关于参数调节
pcl_test_core.cpp中，29行vg.setLeafSize后的三个参数代表对0.2m*0.2m*0.2m的正方体作一次采样，参数值越大，点云越稀疏，如果想提高计算速度可适当调大；
38行seg.setDistanceThreshold(0.001)，为容忍误差范围，单位为m，如果出现误判地面的情况，可适当调小

3.话题接收和发布：pcl_test_core.cpp第五行为接收话题，应改为狗上深度相机所发布的点云话题，接受的话题格式为pointcloud2，7行为发布话题，发布的话题为/filtered_points