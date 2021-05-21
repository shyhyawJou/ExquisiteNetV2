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

if you want to reproduce the results of non-ExquisiteNet mentioned in my [article](https://arxiv.org/abs/2105.09008), remember to add `nn.Dropout(0.2)` into the scripts where models is defined in and comment the weights initialization method to make each layer initialize by default. 

For example, please add `nn.Dropout(0.2)` at `line 177` in torchvision/models/densenet.py

# Training
Run the command `python train.py`

In `train.py`,  
the variable `data_dir` is the directory of dataset  
the variable `optim` is the optimizer  
the variable `backbone` is the classifiaction model you want to use

I have no pre-trained weight of ExquisiteNet

# Inference
Defined in the `util.py`  
`infer_time, loss, acc = util.inference(model, dset["val"], dset_num["val"], batchs_num["val"], loss_func, device)`

# Combined with yolov5
Please download [yolov5](https://github.com/ultralytics/yolov5) first.  
`my.yaml` is the architecture of ExquisiteNetV2-Yolov5. 

# Citation
please cite my [article](https://arxiv.org/abs/2105.09008)
