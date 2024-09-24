# 功能包介绍

本功能包为检测与跟踪模型的部署代码，接收相机话题，对行人和障碍物进行识别并根据识别结果指定跟踪目标。

# 环境依赖

该功能包基于ROS-melodic，CUDA版本10.2
以下为该功能包的软件库依赖
```     
Opencv4
TensorRT 7.x
numpy>=1.22.2
torch>=1.8.0  
torchvision>=0.9.0
pycocotools>=2.0.6
Pycuda
```
# 使用方法

## 模型选取
本功能包存放的模型文件位于engine文件夹下，其中```best.engine```为检测模型，```body.engine```和```leg.engine```为跟踪模型，可根据使用需求进行替换。此外，在进行不同gpu部署时有可能需要重新转换模型，请参考部署相关文件说明

## 使用

### 仅检测
首先确保相机正常运行，修改src/det.cpp中相关话题为部署机器对应话题，之后运行
```shell
source ./devel/setup.bash
rosrun detectobs det
```            
           

输出结果为检测结果图，可使用rviz进行可视化
### 检测跟踪
确保相机正确运行，修改src/sender.cpp中相关话题为部署机器对应话题。之后运行
```shell
source ./devel/setup.bash
rosrun detectobs sender
rosrun detectobs client
```
开始检测后，sender负责检测相机返回的图像，输出检测到的结果，client在sender运行后开始工作，可在用户界面以id形式选择跟踪目标。选择跟踪目标后，将该目标信息输入跟踪部分
在以上步骤完成后，开启跟踪节点
```shell
source ./devel/setup.bash
rosrun detectobs test.py
```           
接收检测信息，返回跟踪目标的坐标并输入行为树，实现跟踪，结果将自动保存在rt_imgs文件夹下