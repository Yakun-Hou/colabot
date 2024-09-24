# 跟踪算法使用说明（Linker）
## Linker 介绍
本项目跟踪部分使用的感知算法为本实验室独立研发的Linker跟踪算法（收录于ACM2021）。与其他算法相比，本算法具有以下优势：
长短期特征融合，以提高跟踪的鲁棒性；
提出具有创新性的增强参照模块，优化跟踪模板特征；
提出本方法时，已在GOT-10k、Lasot等数据计算获得最佳效果。
Linker跟踪算法的总体框架图如下图所示。该算法的输入信息分为三部分：
跟踪模板、参照区域以及搜索区域。
<div align=center>  <img src=".\sot\image\1.png" width=90%> </div>
首先，本方法将以上三部分图像信息，使用Vision Transformer的思路，切割成若干子块（patch）。这若干小块经过线性映射后被投影到向量空间中，即完成嵌入（embedding）过程。嵌入后的向量拼接为一个整体，进入Transformer Block模块，该模块由若干Attention块堆叠而成，每层之间由多层感知器（MLP）连接，每个MLP和Attention块均使用层归一化（LayerNorm）及残差连接。
<div align=center>  <img src=".\sot\image\2.png" width=70%> </div>

随后的RE模块分为两部分，增强模块（Enhancing Block）和耦合模块（Linking Block）。在此模块中，非搜索区域（跟踪模板和参照区域）进入增强模块，该模块主要由交叉注意力机制（Cross-Attention）构成。在此模块中，模板作为注意力机制查询（Query）的来源，参照区域作为键（Key）和值（Value）的来源。交叉注意力机制的输出与原跟踪目标相加，并经过层归一化和MLP（图中省略）后作为增强模块的输出，并作为耦合模块的输入，耦合模块也主要由交叉注意力机制（Cross-Attention）构成，增强模块的输出提供键和值，Transformer Block模块输出的的搜索区域提供查询，交叉注意力机制的输出与搜索区域相加后，并经过层归一化和MLP（图中省略）后作为耦合模块的输出。增强模块和耦合模块分别输出增强后的跟踪模板和耦合后的搜索区域，前者融合了原跟踪模板和参照区域的信息，后者融合了原搜索区域和参照区域的信息。紧接着，再将这两者代替原跟踪模板和原搜索区域，和参照区域一同再次进入Transformer Block模块以及RE模块，迭代若干次。最后一层输出的耦合后的搜索区域作为预测头和分类头的输入，得到最终的预测和分类结果。
<div align=center>  <img src=".\sot\image\3.png" width=90%> </div>
本方法将以上三部分图像信息，使用Vision Transformer的思路，切割成若干子块（patch）。这若干小块经过线性映射后被投影到向量空间中，即完成嵌入（embedding）过程。嵌入后的向量拼接为一个整体，进入Transformer Block模块，该模块由若干Attention块堆叠而成，每层之间由多层感知器（MLP）连接，每个MLP和Attention块均使用层归一化（LayerNorm）及残差连接。

引用链接：[Linker](https://ieeexplore.ieee.org/abstract/document/10416989)
## 环境配置
- 方法1：直接运行 `install.sh` 配置运行所需的conda环境	
	```
	bash install.sh
	```
- 方法2：使用`environment.yaml`配置文件进行安装
	```
	# 需自行补充environment.yaml中的prefix
	conda env create -f environment.yaml
	```

## 数据集准备和路径配置
- 将数据集文件解压到项目目录data文件夹下
- 配置以下文件中的各路径，以符合本地的运行环境
	- lib/test/evaluation/local.py
	- lib/train/admin/environment.py
	- lib/train/admin/local.py
	
## 训练与测试
### 训练
```
# 多GPU训练
python3 tracking/train.py --script linker  --config 256_32x8_ep30_dev  --save_dir test1  --mode multiple --nproc_per_node 4
# 单GPU训练
python3 tracking/train.py --script linker  --config 256_32x8_ep30_dev  --save_dir test1  --mode single
```

### 测试

```
python3 tracking/test.py  linker  256_32x8_ep30_dev  --dataset_name DEV
```