# Deep-learning-object-detection
Related papers, code summary

## 1 Description
   * Deep learning object detection: a survey on various types of image object detection. The paper, code and dataset are maintained.
   
   * More details on image object detection can be found [here](https://www.zhihu.com/column/c_1335912341511663616).

## 2 Image Quality Metrics
* mAP (mean Average Precision) For more on mAP, click [here](https://zhuanlan.zhihu.com/p/358164270).

## 3 Object detection Research
### 3.1 Datasets
------------
现在前沿的目标检测算法（如Faster R-CNN, Yolo, SSD, Mask R-CNN等）基本都是在这些常规数据集上实验的，但是在航空遥感图像上的检测效果并不好，对于航空遥感图像目标检测任务，常规数据集往往难以训练出理想的目标检测器，需要专门的航空遥感图像数据库。主要原因是航空遥感图像有其特殊性：

尺度多样性：航空遥感图像从几百米到近万米的拍摄高度都有，且地面目标即使是同类目标也大小不一，如港口的轮船大的有300多米，小的也只有数十米；

视角特殊性：航空遥感图像的视角基本都是高空俯视，但常规数据集大部分还是地面水平视角，所以同一目标的模式是不同的，在常规数据集上训练的很好的检测器，使用在航空遥感图像上可能效果很差；

小目标问题：航空遥感图像的目标很多都是小目标（几十个甚至几个像素），这就导致目标信息量不大，基于CNN的目标检测方法在常规目标检测数据集上一骑绝尘，但对于小目标，CNN的Pooling层会让信息量进一步减少，一个24*24的目标经过4层pooling后只有约1个像素，使得维度过低难以区分出来；

多方向问题：航空遥感图像采用俯视拍摄，目标的方向都是不确定的（而常规数据集上往往有一定的确定性，如行人、车辆基本都是立着的），目标检测器需要对方向具有鲁棒性；

背景复杂度高：航空遥感图像视野比较大（通常有数平方公里的覆盖范围），视野中可能包含各种各样的背景，会对目标检测产生较强的干扰。

1.DOTA：A Large-scale Dataset for Object Detection in Aerial Images。这是武大遥感国重实验室夏桂松和华科电信学院白翔联合做的一个数据集，2806张遥感图像（大小约4000*4000），188,282个instances，分为15个类别。样本类别及数目如下（与另一个开放数据集NWPU VHR-10对比）。

链接：http://captain.whu.edu.cn/DOTAweb/

2.UCAS-AOD：Dataset of Object Detection in Aerial Images，中科大模式识别实验室标注的，只包含两类目标：汽车，飞机，以及背景负样本。

链接：http://www.ucassdl.cn/resource.asp

DownLoad：Dataset 及其基本情况概述Instruction Instruction-cn

References:[1]H. Zhu, X. Chen, W. Dai, K. Fu, Q. Ye, J. Jiao, "Orientation Robust Object Detection in Aerial Images Using Deep Convolutional Neural Network," IEEE Int'l Conf. Image Processing, 2015.

3.NWPU VHR-10：西北工业大学标注的航天遥感目标检测数据集，共有800张图像，其中包含目标的650张，背景图像150张，目标包括：飞机、舰船、油罐、棒球场、网球场、篮球场、田径场、港口、桥梁、车辆10个类别。开放下载，大概73M。

链接：http://jiong.tea.ac.cn/people/JunweiHan/NWPUVHR10dataset.html

4.RSOD-Dataset：武汉大学团队标注，包含飞机、操场、立交桥、 油桶四类目标，数目分别为：

飞机：4993 aircrafts in 446 images.

操场： 191 playgrounds in 189 images.

立交桥： 180 overpass in 176 overpass.

油桶：1586 oiltanks in 165 images.

链接：https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-

5.INRIA aerial image dataset：Inria是法国国家信息与自动化研究所简称，该机构拥有大量数据库，其中此数据库是一个城市建筑物检测的数据库，标记只有building, not building两种，且是像素级别，用于语义分割。训练集和数据集采集自不同的城市遥感图像。

链接：https://project.inria.fr/aerialimagelabeling/

6.TGRS-HRRSD-Dataset：HRRSD是中国科学院西安光学精密机械研究所光学影像分析与学习中心制作用于研究高分辨率遥感图像目标检测的数据集。

链接：https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset
### 3.2 Papers
--------------
### 2021
* Feng et al, DARDet: A Dense Anchor-free Rotated Object Detector in Aerial Images.注：无锚框自然图像 [[paper](https://arxiv.org/pdf/2110.01025.pdf)][[code](https://github.com/zf020114/DARDet)]
* Steven Lang et al, DAFNe: A One-Stage Anchor-Free Deep Model for Oriented Object Detection.遥感目标检测 [[paper](https://arxiv.org/pdf/2109.06148.pdf)][[code](https://github.com/steven-lang/DAFNe)]
* Jiaming Han et al, ReDet: A Rotation-equivariant Detector for Aerial Object Detection.遥感旋转目标检测 [[paper](https://arxiv.org/pdf/2103.07733.pdf)][[code]( https:
//github.com/csuhan/ReDet)]
* Gong Cheng, et al, Anchor-free Oriented Proposal Generator for Object Detection.遥感旋转目标检测 [[paper](https://arxiv.org/pdf/2110.01931.pdf)][[code](https://github.com/jbwang1997/AOPG)]

### 2020
* D2Det: Towards High Quality Object Detection and Instance Segmentation.自然图像目标检测 [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.pdf)][[code](https://github.com/JialeCao001/D2Det)]
* s2anet  Align Deep Features for Oriented Object Detection.遥感旋转目标检测 [[paper](https://arxiv.org/pdf/2008.09397.pdf)][[code](https://github.com/csuhan/
s2anet)]
* Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection.遥感目标检测 [[paper](https://arxiv.org/pdf/1912.02424.pdf)][[code](https://github.com/sfzhang15/ATSS)]


R-CNN https://arxiv.org/abs/1311.2524
Fast R-CNN https://arxiv.org/abs/1504.08083
Faster R-CNN https://arxiv.org/abs/1506.01497
Cascade R-CNN: Delving into High Quality Object Detection https://arxiv.org/abs/1712.00726
Mask R-CNN https://arxiv.org/abs/1703.06870
SSD https://arxiv.org/abs/1512.02325
FPN(Feature Pyramid Networks for Object Detection) https://arxiv.org/abs/1612.03144
RetinaNet(Focal Loss for Dense Object Detection) https://arxiv.org/abs/1708.02002
Bag of Freebies for Training Object Detection Neural Networks https://arxiv.org/abs/1902.04103
YOLOv1 https://arxiv.org/abs/1506.02640
YOLOv2 https://arxiv.org/abs/1612.08242
YOLOv3 https://arxiv.org/abs/1804.02767
YOLOv4 https://arxiv.org/abs/2004.10934
YOLOX(Exceeding YOLO Series in 2021) https://arxiv.org/abs/2107.08430
PP-YOLO https://arxiv.org/abs/2007.12099
PP-YOLOv2 https://arxiv.org/abs/2104.10419
CornerNet https://arxiv.org/abs/1808.01244
FCOS(Old) https://arxiv.org/abs/1904.01355
FCOS(New) https://arxiv.org/abs/2006.09214
CenterNet https://arxiv.org/abs/1904.07850

### 2019
* Zhi et al, FCOS: Fully Convolutional One-Stage Object Detection.注：无锚框自然图像 [[paper](https://arxiv.org/pdf/1904.01355.pdf)][[code](https://github.com/tianzhi0549/FCOS/)]


### 2018


### Before


## 4 Note
* The above content is constantly updated, welcome continuous attention!

## 5 Contact
* If you have any question, please feel free to contact Longgang Dai (Email: longgang6688@163.com).
