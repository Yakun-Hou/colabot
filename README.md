# COLABOT

## 介绍
本项目为北航人工智能研究院Colab实验室的机器人项目，开发了一套基于四足机器人的第一视角目标识别与主动跟踪与避障系统，解决四足机器人在面向动态变化、障碍物遮挡等场景导致的难以持续跟踪目标的问题，确保主动感知系统的鲁棒性、高精度、高性能。
### 硬件搭建方案
本课题选用纯电机驱动的宇树公司A1机器人作为算法的验证平台，搭载了2个英伟达TX2的计算平台，思岚单线激光雷达和RealSense的深度相机。为了满足机器人的在运行步态算法时的闭环反馈需求，宇树A1机器人足端拥有力传感器实时反馈足端状态，实验平台具体配置如图所示<div align=center>  <img src=".\images\1.png" width=70%> </div>
### 软件搭建方案
本项目整体框架结构由四个主要模块构成，分别是：检测与跟踪模块、行为树、建图模块以及导航模块。传感器和各模块间通信关系如图所示<div align=center>  <img src=".\images\2.png" width=70%> </div>

框架中各个模块的主要功能如下：检测模块的主要功能为对机器人感知范围内的行人和障碍物进行检测，输出目标位置；跟踪模块的主要功能为对于给定的跟踪目标，在机器人的感知范围内给出目标位置；智能决策模块的主要功能为对于目标位置和人类指令给出机器人的行动导航指令；导航和建图模块保证机器人再运动过程中与环境中的障碍物保持安全距离，从而保证安全跟踪；机器人运动控制接口保证机器人能够执行导航指令的速度，保证整个运动控制过程中的鲁棒控制。

对于各个模块的详细内容，请参考训练代码对应的分支。
## 如何使用
该分支主要为本项目的部署代码，基于ROS机器人操作系统搭建，文件结构如下所示
```text
./src
├── detectobs # 检测跟踪部署代码
├── low_level # 相机及控制底层驱动
├── navigation # 基于ROS实现的导航避障包
├── tensorrtx-transformer # yolo检测模型转换工具
└── yolov5deepsort # 基于yolo和deepsort实现的检测跟踪初版代码
```
对于各子功能包的使用方法，请参考对应的readme
## 目标检测跟踪数据集
为了与四足机器人真实的使用场景相契合，本课题针对四足机器人的视角较低，运动时晃动等因素，使用宇树A1机器人上D435深度相机采集四足机器人第一人称数据集，用于目标跟踪任务的训练和测试。数据集中包含2915张图片，其中训练集2541张，验证集374张。数据集分为单人和多人两类，并包含了室内室外两个场景，共8321个人类实例。数据集中的信息包括RGB与深度图像。在数据标注方面，为了满足3.4.4节中描述的双重匹配方法标注方法相对应，数据集标签包含人类实例头部以下全身，示例如图所示<div align=center>  <img src=".\images\3.png" width=70%> </div>
数据集详细情况如下表所示

|视频ID|序列ID|图片数|画面人数|标注部位|数据集划分|
| :-:  | :-: | :-:  | :-:   | :-:    | :-:|
01|0010|186|single|body|Detecting&Tracking val
01|1010|200|single|leg|Tracking val
02|0020|255|single|body|Detecting val&Tracking train
02|1020|264|single|leg|train
03|0030|241|multiple|body|train
04|0040|300|multiple|body|train
05|0050|276|multiple|body|train
06|0060|284|multiple|body|train
07|0070|174|multiple|body|Tracking val
07|1070|100|multiple|leg|Tracking val
08|0080|295|multiple|body|train
08|0081|287|multiple|body|train
08|1080|295|multiple|leg|train
09|0090|281|multiple|body|train
09|0091|278|multiple|body|train
09|1090|281|multiple|leg|train
09|1091|281|multiple|leg|train
10|0100|600|single|body|train
10|1100|600|single|leg|train

## 实验结果

| dataset ID    | RTST↑ |  TA↑ |      MR↓       | FTTR↓  |       备注         |
| :-:           | :-:   | :-:   | :-:    | :-:            | :-:|
| 1    | 97.50% |86.62% | 0.54% | - | 无不含有目标的帧
| 2    | 99.00% |91.28% | 0.00% | - | 无不含有目标的帧
| 3    | 79.06% |78.65% | 25% | 6.12% | -
| 4    |68.24% |92.27% |39.71% | 3.23% | -

## 实现效果
<div align=left>  <img src=".\images\4.gif" width=30%> <img src=".\images\5.gif" width=65%> </div>

## 资源下载
项目中所使用的数据集，权重等可通过以下链接下载

[下载链接](https://bhpan.buaa.edu.cn/link/AAED102EB837C0480CB6E063A08D7A3D10)