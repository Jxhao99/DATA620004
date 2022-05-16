## Faster-Rcnn：Two-Stage目标检测模型在Pytorch当中的实现
---

## 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [评估步骤 How2eval](#评估步骤)
7. [参考资料 Reference](#Reference)



## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12 | [voc_weights_resnet.pth](https://github.com/bubbliiiing/faster-rcnn-pytorch/releases/download/v1.0/voc_weights_resnet.pth) | VOC-Test07 | - | - | 80.36
| VOC07+12 | [voc_weights_vgg.pth](https://github.com/bubbliiiing/faster-rcnn-pytorch/releases/download/v1.0/voc_weights_vgg.pth) | VOC-Test07 | - | - | 77.46

## 文件下载
训练所需的voc_weights_resnet.pth或者voc_weights_vgg.pth以及主干的网络权重可以在百度云下载。  
voc_weights_resnet.pth是resnet为主干特征提取网络用到的；  
voc_weights_vgg.pth是vgg为主干特征提取网络用到的；   
链接: https://pan.baidu.com/s/1S6wG8sEXBeoSec95NZxmlQ      
提取码: 8mgp    

VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/1YuBbBKxm2FGgTU5OfaeC5A    
提取码: uack   

## 训练步骤
### a、训练VOC07+12数据集
1. 数据集的准备   
**本文使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录**  

2. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。   
  ```bash
  python main.py  --train_gpu [0,] #训练的gpu选择
  		--model_path '' #预训练，则为空
  		--pretrained True #是否预训练
  		--backbone resnet50 #主干网络 resnet50/vgg16
  		--Freeze_Epoch 50 #解冻训练的轮数，ps.解冻训练是指对所有参数训练
  		--Freeze_batch_size 4
  		--UnFreeze_Epoch #冻结训练的轮数，ps.冻结训练时冻结主干网络的参数
  		--Unfreeze_batch_size 2
  		--Init_lr 1e-4 #模型最大学习率
  		--Min_lr Init_lr*0.01 #模型最小学习率
  		--Freeze_Train True #是否解冻训练
  		--optimizer_type "adam" #优化器 adam/sgd
  		--momentum 0.9
  		--weight_decay 0
  		--lr_decay_type "cos" #学习率衰减方式 cos/step
  		--num_workers 4
  ```

3. 训练结果预测   
训练结果预测需要用到两个文件，分别是frcnn.py和predict.py。我们首先需要去frcnn.py里面修改model_path以及classes_path，这两个参数必须要修改。   
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**   
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载frcnn_weights.pth，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在frcnn.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/voc_weights_resnet.pth',
    "classes_path"  : 'model_data/voc_classes.txt',
    #---------------------------------------------------------------------#
    #   网络的主干特征提取网络，resnet50或者vgg
    #---------------------------------------------------------------------#
    "backbone"      : "resnet50",
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"    : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"       : 0.3,
    #---------------------------------------------------------------------#
    #   用于指定先验框的大小
    #---------------------------------------------------------------------#
    'anchors_size'  : [8, 16, 32],
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"          : True,
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

## 评估步骤 
### 评估VOC07+12的测试集
1. 本文使用VOC格式进行评估。VOC07+12已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2. 在frcnn.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
3. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

## Reference
https://github.com/chenyuntc/simple-faster-rcnn-pytorch  
https://github.com/eriklindernoren/PyTorch-YOLOv3  
https://github.com/BobLiu20/YOLOv3_PyTorch  
