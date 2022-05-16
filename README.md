## 模型选择

Wide_ResNet    http://arxiv.org/abs/1605.07146 


## 文件下载

测试所需的baseline.t7，cutmix.t7，cutout.t7，mixup.t7放到 model_data/cifar100 文件下

下载链接：

## 训练步骤

```bash
python main.py  --method baseline #选择数据增强方式 baseline/cutout/cutmix/mixup
		--lr 0.1 #初始的学习率
		--depth 28  #Wide_ResNet的depth 一般28/40
		--widen_factor 10 #Wide_ResNet的宽度，推荐10/14/20
		--dropout 0.3
		--dataset cifra100 #默认cifar100，可改为cifar10
		--num_epochs 200 #训练的epoch数
		--seed 980038 #随机种子
		--weight_decay 5e-4
		--momentum 0.9
```



## 测试方式

```bash
python main.py --t
               --method baseline #测试时候的数据增强方式，baseline/cutout/cutmix/mixup
```



## 实施细节

|   epoch   | learning rate |  weight decay | Optimizer | Momentum | Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:--------:|
|   0 ~ 60  |      0.1      |     0.0005    | Momentum  |    0.9   |   true   |
|  61 ~ 120 |      0.02     |     0.0005    | Momentum  |    0.9   |   true   |
| 121 ~ 160 |     0.004     |     0.0005    | Momentum  |    0.9   |   true   |
| 161 ~ 200 |     0.0008    |     0.0005    | Momentum  |    0.9   |   true   |



## CIFAR-100 训练结果

| network           | method   | epochs | accuracy% |
| :---------------- | -------- | -----  | --------- |
| wide-resnet 28x10 | baseline | 200    | 81.34     |
| wide-resnet 28x10 | cutout   | 200    | 81.94     |
| wide-resnet 28x10 | cutmix   | 200    | 83.71     |
| wide-resnet 28x10 | mixup    | 200    | 82.73     |



## 参考

https://github.com/facebookresearch/mixup-cifar10.

https://github.com/szagoruyko/wide-residual-networks.
