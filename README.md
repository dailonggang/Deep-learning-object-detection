# Remote-sensing-object-detection
Related papers, code summary

## 1 Description
   * DehazeZoo: A survey on haze removal from video and single image. Papers, codes and datasets are maintained.

   * Thanks for the sharing of [DerainZoo](https://github.com/nnUyi/DerainZoo) by [Youzhao Yang](https://github.com/nnuyi).
   
   * More details about image dehazing and deraining are available [here](https://www.zhihu.com/column/c_1335912341511663616).

## 2 Image Quality Metrics
* PSNR (Peak Signal-to-Noise Ratio) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4550695) [[matlab code]](https://www.mathworks.com/help/images/ref/psnr.html) [[python code]](https://github.com/aizvorski/video-quality)
* SSIM (Structural Similarity) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1284395) [[matlab code]](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m) [[python code]](https://github.com/aizvorski/video-quality/blob/master/ssim.py)
* VIF (Visual Quality) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1576816) [[matlab code]](http://sse.tongji.edu.cn/linzhang/IQA/Evalution_VIF/eva-VIF.htm)
* FSIM (Feature Similarity) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5705575) [[matlab code]](http://sse.tongji.edu.cn/linzhang/IQA/FSIM/FSIM.htm)
* NIQE (Naturalness Image Quality Evaluator) [[paper]](http://live.ece.utexas.edu/research/Quality/niqe_spl.pdf)[[matlab code]](http://live.ece.utexas.edu/research/Quality/index_algorithms.htm)[[python code]](https://github.com/aizvorski/video-quality/blob/master/niqe.py)

## 3 Dehazing Research
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
* Feng et al, DARDet: A Dense Anchor-free Rotated Object Detector in Aerial Images. [[paper](https://arxiv.org/pdf/2110.01025.pdf)][[code](https://github.com/zf020114/DARDet)]
* Zhao et al, Hybrid Local-Global Transformer for Image Dehazing. [[paper](https://arxiv.org/abs/2109.07100)][code]
* Liu et al, From Synthetic to Real: Image Dehazing Collaborating with Unlabeled Real Data. (ACMMM) [[paper](https://arxiv.org/pdf/2108.02934.pdf)][[code](https://github.com/liuye123321/DMT-Net)]
* Chen et al, PSD: Principled Synthetic-to-Real Dehazing Guided by Physical Priors. (CVPR) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_PSD_Principled_Synthetic-to-Real_Dehazing_Guided_by_Physical_Priors_CVPR_2021_paper.pdf)][[code](https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors)]
* Zheng et al, Ultra-High-Defifinition Image Dehazing via Multi-Guided Bilateral Learning. (CVPR) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Ultra-High-Definition_Image_Dehazing_via_Multi-Guided_Bilateral_Learning_CVPR_2021_paper.pdf)][[code](https://github.com/zzr-idam/4KDehazing)]
* Wu et al, Contrastive Learning for Compact Single Image Dehazing. (CVPR) [[paper](https://arxiv.org/pdf/2104.09367.pdf)][[code](https://github.com/GlassyWu/AECR-Net)]
* Shyam et al, Towards Domain Invariant Single Image Dehazing. (AAAI) [[paper](https://arxiv.org/abs/2101.10449)][[code](https://github.com/PS06/DIDH)]
* Liu et al, Indirect Domain Shift for Single Image Dehazing. [[paper](https://arxiv.org/abs/2102.03268v1)][code]
* Yi et al, Two-Step Image Dehazing with Intra-domain and Inter-domain Adaption. [[paper](https://arxiv.org/pdf/2102.03501.pdf)][code]

### 2020
* Dong et al, Physics-based Feature Dehazing Networks. (ECCV) [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750188.pdf)][code]
* Deng et al, HardGAN: A Haze-Aware Representation Distillation GAN for Single Image Dehazing. (ECCV) [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510715.pdf)][[code](https://github.com/huangzilingcv/HardGAN)]
* Anvari et al, Dehaze-GLCGAN: Unpaired Single Image De-hazing via Adversarial Training. [[paper](http://xxx.itp.ac.cn/abs/2008.06632)][code]
* Zhang et al, Nighttime Dehazing with a Synthetic Benchmark. [[paper](https://arxiv.org/abs/2008.03864)][[code](https://github.com/chaimi2013/3R)]
* Kar et al, Transmission Map and Atmospheric Light Guided Iterative Updater Network for Single Image Dehazing. (CVPR) [[paper](http://xxx.itp.ac.cn/abs/2008.01701)][[code](https://github.com/aupendu/iterative-dehaze)]
* Shen et al, Implicit Euler ODE Networks for Single-Image Dehazing. [[paper](https://arxiv.org/abs/2007.06443)][code]
* Liu et al, Efficient Unpaired Image Dehazing with Cyclic Perceptual-Depth Supervision. [[paper](https://arxiv.org/abs/2007.05220)][code]
* Li et al, You Only Look Yourself: Unsupervised and Untrained Single Image Dehazing Neural Network. [[paper](https://arxiv.org/abs/2006.16829)][code]
* Pang et al, BidNet: Binocular Image Dehazing without Explicit Disparity Estimation. (CVPR) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_BidNet_Binocular_Image_Dehazing_Without_Explicit_Disparity_Estimation_CVPR_2020_paper.pdf)][code]
* Sourya et al, Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing. [[paper](https://arxiv.org/abs/2005.05999)][code]
* Dong et al, Multi-Scale Boosted Dehazing Network with Dense Feature Fusion. (CVPR) [[paper](https://arxiv.org/abs/2004.13388)][[code](https://github.com/BookerDeWitt/MSBDN-DFF)]
* Li et al, Learning to Dehaze From Realistic Scene with A Fast Physics Based Dehazing Network. [[paper](https://arxiv.org/abs/2004.08554)][[code](https://github.com/liruoteng/3DRealisticSceneDehaze)]
* Shao et al, Domain Adaptation for Image Dehazing. (CVPR) [[paper](https://arxiv.org/abs/2005.04668)][[code](https://github.com/HUSTSYJ/DA_dahazing)][[web](https://sites.google.com/site/renwenqi888)]
* Wu et al, Accurate Transmission Estimation for Removing Haze and Noise from a Single Image. (TIP) [[paper](https://ieeexplore.ieee.org/document/8891906)][code]
* Ren et al, Single Image Dehazing via Multi-Scale Convolutional Neural Networks with Holistic Edges. (IJCV) [[paper](https://link.springer.com/article/10.1007%2Fs11263-019-01235-8)][code]
* Dong et al, FD-GAN: Generative Adversarial Networks with Fusion-discriminator for Single Image Dehazing. [[paper](https://arxiv.org/abs/2001.06968)][[code](https://github.com/WeilanAnnn/FD-GAN)]
* Qin et al, FFA-Net: Feature Fusion Attention Network for Single Image Dehazing. (AAAI) [[paper](https://arxiv.org/abs/1911.07559)][[code](https://github.com/zhilin007/FFA-Net)]

### 2019
* Wu et al, Learning Interleaved Cascade of Shrinkage Fields for Joint Image Dehazing and Denoising. (TIP) [[paper](https://ieeexplore.ieee.org/document/8852852)][code]
* Li et al, Semi-Supervised Image Dehazing. (TIP) [[paper](https://ieeexplore.ieee.org/abstract/document/8902220/)][code]
* Li et al, Benchmarking Single Image Dehazing and Beyond. (TIP) [[paper](https://arxiv.org/abs/1712.04143)][code][[web](https://sites.google.com/site/boyilics/website-builder/reside)]
* Pei et al, Classification-driven Single Image Dehazing. [[paper](https://arxiv.org/abs/1911.09389)][code]
* Liu et al, GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing. (ICCV) [[paper](https://arxiv.org/abs/1908.03245)][[code](https://github.com/proteus1991/GridDehazeNet)]
* Li et al, Joint haze image synthesis and dehazing with mmd-vae losses. [[paper](https://arxiv.org/abs/1905.05947)][code]
* Peter et al, Feature Forwarding for Efficient Single Image Dehazing. [[paper](https://arxiv.org/abs/1904.09059)][code]
* Shu et al, Variational Regularized Transmission Refinement for Image Dehazing. [[paper](https://arxiv.org/abs/1902.07069)][code]
* Liu et al, End-to-End Single Image Fog Removal using Enhanced Cycle Consistent Adversarial Networks. [[paper](https://arxiv.org/abs/1902.01374)][code]
* Chen et al, Gated Context Aggregation Network for Image Dehazing and Deraining. (WACV) [[paper](https://arxiv.org/abs/1811.08747)][[code](https://github.com/cddlyf/GCANet)]
* Ren et al, Deep Video Dehazing with Semantic Segmentation. (TIP) [[paper](https://ieeexplore.ieee.org/document/8492451)][code]

### 2018
* Ren et al, Gated Fusion Network for Single Image Dehazing. (CVPR) [[paper](https://arxiv.org/abs/1804.00213)][[code](https://github.com/rwenqi/GFN-dehazing)][[web](https://sites.google.com/site/renwenqi888/research/dehazing/gfn)]
* Zhang et al, FEED-Net: Fully End-To-End Dehazing. (ICME) [paper][code]
* Zhang et al, Densely Connected Pyramid Dehazing Network. (CVPR) [[paper](https://arxiv.org/abs/1803.08396)][[code](https://github.com/hezhangsprinter/DCPDN)]
* Yang et al, Towards Perceptual Image Dehazing by Physics-based Disentanglement and Adversarial Training. (AAAI) [paper][code]
* Deniz et al, Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing. (CVPRW) [[paper](https://arxiv.org/abs/1805.05308v1)][code]

### Before
* Li et al, An All-in-One Network for Dehazing and Beyond. (ICCV) [[paper](https://arxiv.org/pdf/1707.06543.pdf)][[code](https://github.com/MayankSingal/PyTorch-Image-Dehazing)][[web](https://sites.google.com/site/boyilics/website-builder/project-page)]
* Zhu et al, Single Image Dehazing via Multi-Scale Convolutional Neural Networks. (ECCV) [[paper](https://drive.google.com/open?id=0B7PPbXPJRQp3TUJ0VjFaU1pIa28)][[code](https://sites.google.com/site/renwenqi888/research/dehazing/mscnndehazing/MSCNN_dehazing.zip?attredirects=0&d=1)][[web](https://sites.google.com/site/renwenqi888/research/dehazing/mscnndehazing)]
* Cai et al, DehazeNet: An end-to-end system for single image haze removal. (TIP) [[paper](http://caibolun.github.io/papers/DehazeNet.pdf)][[code](https://github.com/caibolun/DehazeNet)][[web](http://caibolun.github.io/DehazeNet/)]
* Zhu et al, A fast single image haze removal algorithm using color attenuation prior. (TIP) [[paper](https://ieeexplore.ieee.org/document/7128396)][code]
* He et al, Single Image Haze Removal Using Dark Channel Prior. (CVPR) [[paper](http://www.jiansun.org/papers/Dehaze_CVPR2009.pdf)][code]

## 4 Note
* The above content is constantly updated, welcome continuous attention!

## 5 Contact
* If you have any question, please feel free to contact Xiang Chen (Email: cv.xchen@gmail.com).
