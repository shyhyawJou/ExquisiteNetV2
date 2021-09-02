# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 23:39:22 2020

@author: dddd
"""
from efficientnet_pytorch import EfficientNet #https://github.com/lukemelas/EfficientNet-PyTorch
from senet import se_resnet #https://github.com/moskomule/senet.pytorch
from mobilenetv3 import mobilenetv3#https://github.com/d-li14/mobilenetv3.pytorch
from ghostnet import ghostnet #https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch

import torch as th
import torch.nn as nn
import torchvision as thv
from torch.nn import functional as F
from pathlib import Path as p
import numpy as np
import collections
from PIL import Image
import time

class My_Dataset():
    def __init__(self, img_dir, transform=None, is_int=False):
        self.img_paths = [str(i) for i in p(img_dir).glob('*/*')]
        self.classes = sorted([i.name for i in p(img_dir).glob('*') if i.is_dir()])
        self.transform = transform
        self.dict = dict(zip(self.classes, np.arange(len(self.classes)).astype(np.int64)))

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        label = self.dict[p(self.img_paths[index]).parent.name]
        img_path = self.img_paths[index]
        if self.transform != None:
            img = self.transform(img)

        return img, label, img_path
        
    def __len__(self):
        return len(self.img_paths)
    
    def equalize_(self):
        num = dict(zip(self.classes, np.zeros(len(self.classes))))
        for i in self.classes:
            num[i] = len(os.listdir(pj(self.folder, i)))
        min_num = min(num.values())
        self.img_paths = []
        for i in self.classes:
            pp = np.asarray(list(p(pj(self.folder, i)).glob('*')))
            num_in_class = len(pp)
            pp = pp[permutation(num_in_class)].tolist()
            self.img_paths += pp[:min_num]
        self.img_paths = sorted(str(i) for i in self.img_paths)

    def concat_(self, ds):
        self.img_paths = sorted(self.img_paths + ds.img_paths)
        assert self.classes == ds.classes

class My_Sampler(th.utils.data.Sampler):
    def __init__(self, my_dset, init_seed):
        self.my_dset = my_dset
        self.seed = init_seed
        
    def __iter__(self):
        th.manual_seed(self.seed)
        order = th.randperm(len(self.my_dset))
        self.seed += 1
        return iter(order)
    
def get_dataset(data_dir, data_transforms):    
    if data_dir == 'birds2011':
        tr_set = My_Dataset(data_dir+'/train', 
                                 transform=data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                 transform=data_transforms['val'])

    if data_dir == 'oxford_flowers102':
        tr_set = My_Dataset(data_dir+'/train', 
                                 data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                data_transforms['val'])

    if data_dir == 'food101':
        tr_set = My_Dataset(data_dir+'/train', 
                                 transform=data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                 transform=data_transforms['val'])

    if data_dir == 'dtd':
        tr_set = My_Dataset(data_dir+'/train', 
                                 transform=data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                 transform=data_transforms['val'])

    if data_dir == 'RAF':
        tr_set = My_Dataset(data_dir+'/train', 
                                 transform=data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                 transform=data_transforms['val'])

    if data_dir == 'chest_xray':
        tr_set = My_Dataset(data_dir+'/train', 
                                 transform=data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                 transform=data_transforms['val'])

    if data_dir == 'stl10':
        tr_set = My_Dataset(data_dir+'/train', 
                                 transform=data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                 transform=data_transforms['val'])

    if data_dir == 'mnist':
        tr_set = My_Dataset(data_dir+'/trains', 
                                 transform=data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/tests', 
                                 transform=data_transforms['val'])

    if data_dir == 'Leukemia':
        data_dir = 'C-NMC_Leukemia'
        tr_set = My_Dataset(data_dir+'/training_data', 
                                 transform=data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/validation_data', 
                                 transform=data_transforms['val'])

    if data_dir == 'Fruit360':
        tr_set = My_Dataset(data_dir+'/Training', 
                                 transform=data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/Test', 
                                 transform=data_transforms['val'])

    if data_dir == 'online_prodicts':
        tr_set = My_Dataset(data_dir+'/train', 
                                 data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                 data_transforms['val'])
    
    if data_dir == 'caltech256':
        tr_set = My_Dataset(data_dir+'/train', 
                                 data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                 data_transforms['val'])

    if data_dir == 'oct':
        tr_set = My_Dataset(data_dir+'/trains', 
                                 data_transforms['train'])
           
        val_set = My_Dataset(data_dir+'/test', 
                                 data_transforms['val'])

    if data_dir == 'SUN397':
        tr_set = My_Dataset(data_dir+'/train', 
                                 data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                 data_transforms['val'])

    if data_dir == 'plant_seedingv2':
        tr_set = My_Dataset(data_dir+'/train', 
                                 data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/test', 
                                 data_transforms['val'])

    if data_dir == 'synthetic_digits':
        tr_set = My_Dataset(data_dir+'/imgs_train', 
                                 data_transforms['train'])
            
        val_set = My_Dataset(data_dir+'/imgs_valid', 
                                 data_transforms['val'])

    class_names = tr_set.classes
    class_num = len(class_names)
    
    if not np.array(class_names == val_set.classes).all():
        raise ValueError('tr class name should equal to val class name')
    if tr_set[0][0].size()[1:] != (224,224):
        raise ValueError('train img size should be (224x224), but get {}'.format(tr_set[0][0].size()[1:]))
    if val_set[0][0].size()[1:] != (224,224):
        raise ValueError('val img size should be (224x224), but get {}'.format(val_set[0][0].size()[1:]))
    
    return  tr_set, val_set, class_names, class_num

def model(pretrained, backbone, class_num):  
    if backbone.split('-')[0] == 'efficientnet' and pretrained:
        model = EfficientNet.from_pretrained(backbone) 
    if backbone.split('-')[0] == 'efficientnet' and not pretrained:
        model = EfficientNet.from_name(backbone, num_classes=class_num) 

    if backbone == 'squeezenet1.0':
        model = thv.models.squeezenet1_0(pretrained, num_classes=class_num)
    if backbone == 'squeezenet1.1':
        model = thv.models.squeezenet1_1(pretrained, num_classes=class_num)

    if backbone == 'resnet18':
        model = thv.models.resnet18(pretrained, num_classes=class_num)
    if backbone == 'resnet50':
        model = thv.models.resnet50(pretrained, num_classes=class_num)
    if backbone == 'resnet101':
        model = thv.models.resnet101(pretrained, num_classes=class_num)

    if backbone.split('-')[0] == 'mobilenetv3' and backbone.split('-')[1] == 'large':
        model = aaa.mobilenetv3_large(num_classes=class_num)
        if pretrained:
            model.load_state_dict(th.load('mobilenetv3/pretrained/mobilenetv3-large-1cd25616.pth'))
    if backbone.split('-')[0] == 'mobilenetv3' and backbone.split('-')[1] == 'small':
        model = mbv3.mobilenetv3_small(num_classes=class_num)
        if pretrained:
            model.load_state_dict(th.load('mobilenetv3/pretrained/mobilenetv3-small-55df8e1f.pth'))

    if backbone == 'shufflenetv2_2.0':
        model = thv.models.shufflenet_v2_x2_0(pretrained, num_classes=class_num)
    
    if backbone == 'densenet121':
        model = thv.models.densenet121(pretrained, num_classes=class_num)

    if backbone == 'ghostnet':
        model = ghostnet(num_classes=class_num)
        if pretrained:
            model.load_state_dict(th.load('models/state_dict_93.98.pth'))

    if backbone == 'senet18':
        model = se_resnet.se_resnet18(num_classes=class_num)
    if backbone == 'seln_net':
        model = se_resnet.seln_resnet18(num_classes=class_num)
    
    if backbone == 'ExquisiteNetV1':
        model = ExquisiteNetV1(class_num)
    if backbone == 'LN-ExquisiteNetV1':
        model = ExquisiteNetV1_LN(class_num)
    if backbone == 'SE-ExquisiteNetV2':
        model = ExquisiteNetV2_SE(class_num)
    if backbone == 'ExquisiteNetV2': 
        model = ExquisiteNetV2(class_num)

    print(model)
    return model

def train(model, ckp, dset, dset_num, batchs_num, loss_func, optimizer, scheduler, epochs, device):
    
    best_acc = 0.0
    max_tr_acc = 0.0
    min_tr_loss = 100000.0
    c = 0
        
    time0 = time.time()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
    
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0
            dset_No = 0
            
            for inputs, labels, img_paths in dset[phase]:
                dset_No += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
    
                with th.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = th.max(outputs, 1) 
                    loss = loss_func(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                running_loss += loss.detach()*inputs.size(0)
                running_corrects += th.sum(preds == labels)

                print(('%s_set: %d/%d'  %(phase, dset_No, batchs_num[phase])).ljust(25),end='\r')
    
            epoch_loss = running_loss / dset_num[phase]
            epoch_acc = running_corrects.double() / dset_num[phase]
    
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                            
            if phase == 'train':
                if epoch_acc > max_tr_acc:
                    max_tr_acc = epoch_acc
                if epoch_loss < min_tr_loss:
                    min_tr_loss = epoch_loss

            if phase == 'val':
                if epoch_acc > best_acc:
                    c = 0
                    best_acc = epoch_acc
                    th.save({'model': model.state_dict(),
                             'optimizer': optimizer.state_dict()}, ckp)
                    th.save(model, p(ckp).with_name('md.pth'))
                else :
                    c += 1
                
                #print('lr: %f' %optimizer.param_groups[-1]['lr'])
                if scheduler != None:
                    scheduler.step(min_tr_loss)

        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best val acc has not changed for %d times'%c)
        print('seed: %d'%th.initial_seed())
        print()        
            
        if min_tr_loss <= 0.01 or epoch+1 == 150:
            print('early stop!')
            break
                    
    timez = time.time()
    
    return timez - time0, max_tr_acc, min_tr_loss, best_acc, epoch+1

def inference(model, dset, dset_num, batchs_num, loss_func, device):
    running_loss = 0.0
    running_corrects = 0
    dset_No = 0
    
    time0 = time.time()         
    for inputs, labels, img_paths in dset:
        with th.set_grad_enabled(False):
            dset_No += 1
            inputs = inputs.to(device)
            labels = labels.to(device) 
            outputs = model(inputs)
            _, preds = th.max(outputs, 1)
            loss = loss_func(outputs, labels)
        
            running_loss += loss.detach()*inputs.size(0)
            running_corrects += th.sum(preds == labels)
            print(('dset: %d/%d'  %(dset_No, batchs_num)).ljust(25),end='\r')
    
    epoch_loss = running_loss / dset_num
    epoch_acc = running_corrects.double() / dset_num
    
    print('dset Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch_loss, epoch_acc))
    timez = time.time()
    
    return timez - time0, epoch_loss, epoch_acc

def pad_num(k_s):
    pad_per_side = int((k_s-1)*0.5)
    return pad_per_side

class SE(nn.Module):
    def __init__(self, cin, ratio):
        super(SE, self).__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(cin, int(cin/ratio), bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1.out_features, cin, bias=False)
        self.act2 = nn.Sigmoid()
        
    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1,x.size()[1])
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = x.view(-1,x.size()[1],1,1)
        return x*y

class SE_LN(nn.Module):
    def __init__(self, cin):
        super(SE_LN, self).__init__()  
        self.gavg = nn.AdaptiveAvgPool2d(1)
        self.ln = nn.LayerNorm(cin)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1,x.size()[1])
        x = self.ln(x)
        x = self.act(x)   
        x = x.view(-1,x.size()[1],1,1)
        return x*y

class DFSEBV1(nn.Module):
    def __init__(self, cin, dw_s, ratio, is_LN):
        super(DFSEBV1, self).__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.ReLU()
        self.dw1 = nn.Conv2d(cin,cin,dw_s,1,pad_num(dw_s),groups=cin)
        self.act2 = nn.Hardswish()
        if is_LN:
            self.se1 = SE_LN(cin)
        else:
            self.se1 = SE(cin,3)

        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act3 = nn.ReLU()
        self.dw2 = nn.Conv2d(cin,cin,dw_s,1,pad_num(dw_s),groups=cin)
        self.act4 = nn.Hardswish()
        if is_LN:
            self.se2 = SE_LN(cin)
        else:
            self.se2 = SE(cin,3)

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.act2(x)
        x = self.se1(x)
        x = x + y

        x = self.pw2(x)
        x = self.bn2(x)
        x = self.act3(x)
        x = self.dw2(x)
        x = self.act4(x)
        x = self.se2(x)
        x = x + y
        #del y
        return x

class DFSEBV2(nn.Module):
    def __init__(self, cin, dw_s, is_LN):
        super(DFSEBV2, self).__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.GELU()
        self.dw1 = nn.Conv2d(cin,cin,dw_s,1,pad_num(dw_s),groups=cin)
        if is_LN:
            self.seln = SE_LN(cin)
        else:
            self.seln = SE(cin,3)
            
        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act2 = nn.Hardswish()
        self.dw2 = nn.Conv2d(cin,cin,dw_s,1,pad_num(dw_s),groups=cin)

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.seln(x)
        x = x + y
        
        x = self.pw2(x)       
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dw2(x)
        x = x + y
        #del y
        return x
        
# Feature concentrator
class FCT(nn.Module):
    def __init__(self, cin, cout):
        super(FCT, self).__init__()
        self.dw = nn.Conv2d(cin,cin,4,2,1,groups=cin,bias=False)
        self.minpool = MinPool2d()
        self.maxpool = nn.MaxPool2d(2,ceil_mode=True)
        self.pw = nn.Conv2d(3*cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        z = self.dw(x)
        y = self.minpool(x)
        x = self.maxpool(x)
        x = th.cat((x,y,z), 1)
        x = self.pw(x)
        x = self.bn(x)
        return x

class EVE(nn.Module):
    def __init__(self, cin, cout):
        super(EVE, self).__init__()
        self.minpool = MinPool2d()
        self.maxpool = nn.MaxPool2d(2,ceil_mode=True)
        self.pw = nn.Conv2d(2*cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        y = self.minpool(x)
        x = self.maxpool(x)
        x = th.cat((x,y), 1)
        x = self.pw(x)
        x = self.bn(x)
        return x

class ME(nn.Module):
    def __init__(self, cin, cout):
        super(ME, self).__init__()
        self.maxpool = nn.MaxPool2d(2,ceil_mode=True)
        self.pw = nn.Conv2d(cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.pw(x)
        x = self.bn(x)
        return x

class MinPool2d(nn.Module):
    def forward(self, x):
        x = -F.max_pool2d(-x,2,ceil_mode=True)
        return x

class ExquisiteNetV1(nn.Module):
    def __init__(self, class_num):
        super(ExquisiteNetV1, self).__init__()
        self.features = nn.Sequential(
            collections.OrderedDict([
                ('ME1', ME(3,12)),  
                ('DFSEB1', DFSEBV1(12,3,3,False)),
                
                ('ME2', ME(12,50)),  
                ('DFSEB2', DFSEBV1(50,3,3,False)),
                
                ('ME3', ME(50,100)),  
                ('DFSEB3', DFSEBV1(100,3,3,False)),
                
                ('ME4', ME(100,200)),  
                ('DFSEB4', DFSEBV1(200,3,3,False)),
                
                ('ME5', ME(200,350)),  
                ('DFSEB5', DFSEBV1(350,3,3,False)),

                ('conv', nn.Conv2d(350,640,1,1)),  
                ('act', nn.Hardswish())
                                    ])
                                     )       
        self.gavg = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(640, class_num)

    def forward(self, x):
        x = self.features(x)
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1,x.size()[1])
        x = self.fc(x)
        return x

class ExquisiteNetV1_LN(nn.Module):
    def __init__(self, class_num):
        super(ExquisiteNetV1_LN, self).__init__()
        self.features = nn.Sequential(
            collections.OrderedDict([
                ('ME1', ME(3,12)),  
                ('DFSEB1', DFSEBV1(12,3,3,True)),
                
                ('ME2', ME(12,50)),  
                ('DFSEB2', DFSEBV1(50,3,3,True)),
                
                ('ME3', ME(50,100)),  
                ('DFSEB3', DFSEBV1(100,3,3,True)),
                
                ('ME4', ME(100,200)),  
                ('DFSEB4', DFSEBV1(200,3,3,True)),
                
                ('ME5', ME(200,350)),  
                ('DFSEB5', DFSEBV1(350,3,3,True)),

                ('conv', nn.Conv2d(350,640,1,1)),  
                ('act', nn.Hardswish())
                                    ])
                                     )       
        self.gavg = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(640, class_num)

    def forward(self, x):
        x = self.features(x)
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1,x.size()[1])
        x = self.fc(x)
        return x

class ExquisiteNetV2_SE(nn.Module):
    def __init__(self, class_num):
        super(ExquisiteNetV2_SE, self).__init__()
        self.features = nn.Sequential(
            collections.OrderedDict([
                ('FCT', FCT(3,12)),  
                ('DFSEB1', DFSEBV2(12,3,False)),

                ('EVE', EVE(12,48)),  
                ('DFSEB2', DFSEBV2(48,3,False)),

                ('ME1', ME(48,96)),  
                ('DFSEB3', DFSEBV2(96,3,False)),

                ('ME2', ME(96,192)),  
                ('DFSEB4', DFSEBV2(192,3,False)),

                ('ME3', ME(192,384)),  
                ('DFSEB5', DFSEBV2(384,3,False)),

                ('dw', nn.Conv2d(384,384,3,1,pad_num(3),groups=384)),  
                ('act', nn.Hardswish())
                                    ])
                                     )       
        self.gavg = nn.AvgPool2d((7,7))
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(384,class_num)

    def forward(self, x):
        x = self.features(x)
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1,x.size()[1])
        x = self.fc(x)
        return x

class ExquisiteNetV2(nn.Module):
    def __init__(self, class_num):
        super(ExquisiteNetV2, self).__init__()
        self.features = nn.Sequential(
            collections.OrderedDict([
                ('FCT', FCT(3,12)),  
                ('DFSEB1', DFSEBV2(12,3,True)),

                ('EVE', EVE(12,48)),  
                ('DFSEB2', DFSEBV2(48,3,True)),

                ('ME1', ME(48,96)),  
                ('DFSEB3', DFSEBV2(96,3,True)),

                ('ME2', ME(96,192)),  
                ('DFSEB4', DFSEBV2(192,3,True)),

                ('ME3', ME(192,384)),  
                ('DFSEB5', DFSEBV2(384,3,True)),

                ('dw', nn.Conv2d(384,384,3,1,pad_num(3),groups=384)),  
                ('act', nn.Hardswish())
                                    ])
                                     )       
        self.gavg = nn.AvgPool2d((7,7))
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(384,class_num)

    def forward(self, x):
        x = self.features(x)
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1,x.size()[1])
        x = self.fc(x)
        return x
