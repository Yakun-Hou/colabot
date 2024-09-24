# 目标检测代码使用说明（YOLO）
## 检测模型介绍
如图所示，目标检测和跟踪模块的总体流程如下：
(1)在跟踪任务执行前，需要对于跟踪目标进行确认，此时相机接收到RGB图像，将图像输入到目标识别模块中，对于目标进行识别，输出跟随领航员的位置和当前障碍物的位置。由于图像中可能会出现多个人，因此需要用户手动选择单个人作为跟踪目标。
(2)在跟踪任务执行过程中，跟踪目标由准备阶段确认完毕后输入到跟踪模块中作为模板，跟踪模块接受RGB图像视频流，输出目标的位置，深度相机中的深度信息对与目标位置进行3D位置估计。
对于目标检测算法，考虑到宇树A1机器人硬件性能有限，因此采用YOLO v5架构的算法，YOLO系列算法是工业界应用较为广泛，在边缘计算领域，能够在保证模型推理效果的同时满足性能需求。
<div align=center>  <img src=".\YOLO\images\1.png" width=50%> </div>
对于目标检测算法，考虑到宇树A1机器人硬件性能有限，因此采用YOLO v5架构的算法，YOLO系列算法是工业界应用较为广泛，在边缘计算领域，能够在保证模型推理效果的同时满足性能需求。

引用：[YOLOv5](https://github.com/ultralytics/yolov5)

## 环境依赖
要求在 [**Python>=3.8.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) ，且要求 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) 。
推荐使用conda进行环境管理。
```bash
conda create --name test python=3.8
conda activate test
pip install -r requirements.txt  # install
```
## 数据集结构
数据集文件应按照如下结构放置
```
your dataset 
├── images      
       ├── train          
              ├── xx.jpg     
       ├── val         
              ├── xx.jpg  
├── labels      
       ├── train          
              ├── xx.txt     
       ├── val         
              ├── xx.txt  
```
## 更改配置
在```data/fused_img.yaml```下将数据集路径指定为自己的路径
```bash
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]

path: /datasetdir/ # dataset root dir
train: /datasetdir/images/train # train images 
val: /datasetdir/images/val # val images 
```
## 运行测试
```
python val.py  --data fused_img.yaml --weights /weightdir --augment --batch-size 8
```