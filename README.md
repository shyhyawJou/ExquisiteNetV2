# Requirements
[Pytorch 1.7.0](https://pytorch.org/)  
[torchsummaryX](https://github.com/nmhkahn/torchsummaryX)  
[Ranger optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)  

# Optional
[EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)  
[SENet](https://github.com/moskomule/senet.pytorch)  
[MobileNetV3](https://github.com/d-li14/mobilenetv3.pytorch)  
[ghostnet](https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch)

if you only want to use ExquisiteNet, you don't need to download these models.

# Training
python train.py
In train.py,  
the variable `data_dir` is the directory of images  
the variable `optim` is the optimizer  
the variable `backbone` is the classifiaction model you want to use

# Inference
Defined in the `util.py`  
`infer_time, loss, acc = util.inference(model, dset["val"], dset_num["val"], batchs_num["val"], loss_func, device)`
