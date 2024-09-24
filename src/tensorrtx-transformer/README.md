# 功能包介绍

本功能包为检测模型```best.engine```的转换工具。在进行不同gpu部署时有可能需要重新转换模型，可用此功能包进行转换

## 环境依赖
    TensorRT 7.x
    gcc，g++版本>=4，<=7
## 使用方法
```shell
mkdir build
cd build
cmake ..
make
```
将附带的```best.wts```模型文件放入build文件夹
```shell
sudo ./yolov5 -s best.wts best.engine s
```
## 引用
https://github.com/wang-xinyu/tensorrtx

