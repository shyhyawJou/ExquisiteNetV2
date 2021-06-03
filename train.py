# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:49:15 2020

@author: dddd
"""
import torch as th
import torch.nn as nn
from torch.utils.data import BatchSampler as Bsp
import torchvision as thv
import time
import shutil

import util 

from torchsummaryX import summary
from ranger import RangerQH  #https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
from ranger import RangerVA  
from ranger import Ranger
from pathlib import Path as p

def main():
    data_dir = 'beans'
    data_dir = 'birds2011'
    data_dir = 'oxford_flowers102'
    data_dir = 'dtd'
    #data_dir = 'RAF'
    #data_dir = 'chest_xray'
    #data_dir = 'Leukemia'
    #data_dir = 'Fruit360'
    #data_dir = 'food101'
    #data_dir = 'online_prodicts'
    #data_dir = 'caltech256'
    #data_dir = 'oct'
    #data_dir = 'SUN397'
    #data_dir = 'stl10'
    #data_dir = 'plant_seedingv2'
    #data_dir = 'mnist'
    #data_dir = 'synthetic_digits'

    batch_size = 50
    core_num = 4
    epochs = 150
    seed = 21
    img_shape = [224,224]
    lr = 0.05
    dampen = 0
    decay = 0
    factor = 0.1
    ckp = 'ckp50/wt.pth'
    p(ckp).parent.mkdir(parents=True, exist_ok=True)
    lr_schedule = True
    pretrained = False
    my_sampler = True
    optim = 'rangerva'
    #optim = 'rangerqh'
    optim = 'sgd'
    #optim = 'asgd'
    #optim = 'adam'
    
    backbone = 'mobilenetv3-large'
    #backbone = 'densenet121'
    #backbone = 'efficientnet-b0'
    #backbone = 'resnet18'
    #backbone = 'resnet50'
    #backbone = 'squeezenet1.1'
    #backbone = 'shufflenetv2_2.0'
    #backbone = 'senet18'
    #backbone = 'seln_net'
    #backbone = 'ghostnet'
    backbone = 'ExquisiteNetV1' 
    #backbone = 'LN-ExquisiteNetV1'
    #backbone = 'SE-ExquisiteNetV2'
    backbone = 'ExquisiteNetV2' 

    device = th.device("cuda")
    if not th.cuda.is_available():
        raise ValueError('there is no detected gpu')  
    ##################################################    
    
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.manual_seed(seed)

    data_transforms = { 'train': thv.transforms.Compose([
                                 #thv.transforms.RandomHorizontalFlip(),
                                 #thv.transforms.ColorJitter((0.8,1.2), (0.8,1.2), (0.8,1.2)),
                                 thv.transforms.ToTensor()
                                 ]),
                        'val': thv.transforms.ToTensor()}
                                
    tr_set, val_set, class_names, class_num = util.get_dataset(data_dir,data_transforms)
                    
    if my_sampler:
        batch_sampler = Bsp(util.My_Sampler(tr_set, seed), batch_size, False)
        tr_batch_size = 1
    else:
        batch_sampler = None
        tr_batch_size = batch_size
            
    dset = {'train':th.utils.data.DataLoader(
                tr_set,
                batch_size=tr_batch_size,
                shuffle=not my_sampler,
                batch_sampler=batch_sampler,
                pin_memory=True,
                num_workers=core_num), 
        
            'val':th.utils.data.DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=core_num)
    }
    
    dset_num = {'train':len(tr_set), 'val':len(val_set)}
    batchs_num = {}
    for i in dset_num:
        batchs_num[i] = dset_num[i]//batch_size if dset_num[i]%batch_size == 0 else dset_num[i]//batch_size + 1
    
    model = util.model(pretrained, 
                      backbone,
                      class_num
                      )     

    summary(model, th.zeros((1,) + tr_set[0][0].size())) #tr_set[0][0] is img
    model = model.to(device)

    print('img shape: {}'.format(tr_set[0][0].size()))
    print('class name: %s'%class_names)   
    print('class_num: %d'%class_num)
    print('tr_num: %d'%dset_num['train'])     
    print('val_num: %d'%dset_num['val']) 
    print()
    print('='*20+'  %s  '%data_dir+'='*20)  
    print('='*20+'  using %s  '%backbone+'='*20)
    print('='*20+'  pretrained: %s  '%pretrained+'='*20)
    print('='*20+'  my sampler: %s  '%my_sampler+'='*20)
    print('='*20+'  lr schedule: %s  '%lr_schedule+'='*20)
    print('='*20+'  optimizer: %s  '%optim+'='*20)
    print('='*20+'  device: %s  '%device+'='*20)
    print('='*20+'  initial seed: %s  '%seed+'='*20)
    print()

    loss_func = nn.CrossEntropyLoss()

    if optim == 'sgd':
        optimizer = th.optim.SGD(model.parameters(), lr, 0.9, dampen, decay)
    if optim == 'rangerqh':
        optimizer = RangerQH(model.parameters(), lr, weight_decay=decay)
    if optim == 'rangerva':
        optimizer = RangerVA(model.parameters(), lr, weight_decay=decay)
    
    if lr_schedule:
        lr_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=factor,
            mode='min',
            patience=5-1,
            verbose=True)
    else:
        lr_scheduler = None

    c = 0    
    tr_time, max_tr_acc, min_tr_loss, val_acc, epoch = util.train(
                    model, 
                    ckp,
                    dset,
                    dset_num,
                    batchs_num,
                    loss_func, 
                    optimizer, 
                    lr_scheduler,
                    epochs, 
                    device
                    )

    print('-'*50)

    # load weight
    checkpoint = th.load(ckp)
    model.load_state_dict(checkpoint['model'])
    t, val_loss, val_acc = util.inference(model, dset['val'], dset_num['val'], batchs_num['val'], loss_func, device)
                      
    #load weight & model
    model = th.load(p(ckp).with_name('md.pth'))
    t, val_loss, val_acc = util.inference(model, dset['val'], dset_num['val'], batchs_num['val'], loss_func, device)
    print('-'*50)
    print()
    
    print()
    print('-'*50)  
    print('='*20+'  %s  '%data_dir+'='*20)  
    print('='*20+' using %s  '%backbone+'='*20)
    print('='*20+' initial seed %s  '%seed+'='*20)
    print('total epoch has been run: %d'%epoch)
    print('min tr loss: %.4f'%(min_tr_loss))
    print('best tr acc: %.4f'%(max_tr_acc))
    print('best val acc: %.4f'%(val_acc))
    print('tr num: %d'%dset_num['train'])
    print('val num: %d'%dset_num['val'])
    print('train time: %.6f'%(tr_time))

if __name__ == '__main__':
    main()
