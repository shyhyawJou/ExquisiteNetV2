import argparse
from os.path import join as pj

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
from torch.utils.data import BatchSampler, DataLoader

from dataset import My_Dataset, My_Sampler
from network import get_optimizer, ExquisiteNetV2
from tool import train_acc, eval_acc, del_ipynb_ckps, create_save_dir
from img import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help="path of dataset")
    parser.add_argument('--weight', type=str, default=None, help="path of pretrained weight")
    parser.add_argument('--amp', type=bool, default=True, help="auto mixed precision training")
    # won't really run 1000 epochs, when lr less than end_lr, training will be stopped
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default="weight", help="path where the weight will be saved")
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--opt', type=str, default="sgd", help="optimizer")
    parser.add_argument('--init_lr', type=float, default=0.1, help="initial learning rate")
    parser.add_argument('--lr_df', type=float, default=0.7, help="learning rate decent factor")
    parser.add_argument('--end_lr', type=float, default=0.01, help="stop training when the lr less than end_lr")
    parser.add_argument('--wd', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--imgsz', type=int, default=224, help="image size")
    parser.add_argument('--val_r', type=float, default=0.2, help="ratio of val dataset accounting for training set")
    parser.add_argument('--worker', default=4)
    parser.add_argument('--seed', default=None)
    return parser.parse_args()


def main():
    args = get_args()

    args.save_dir = create_save_dir(args.save_dir)

    # for jupyter notebook
    del_ipynb_ckps(args.data)
 
    tr_T = [
        To_mode("RGB"),            
        T.Resize((args.imgsz, args.imgsz), InterpolationMode.BILINEAR),   
        T.RandomHorizontalFlip(p=0.5),   
        RandomColorJitter(
            p=0.5,
            brightness = 0.05,
            contrast = 0.05,
            saturation = 0.05
        ),  
        RandomTranslate(0.5, (0.05,0.05)),
        T.ToTensor()
    ]
    
    val_T = [
        To_mode("RGB"), 
        T.Resize((args.imgsz, args.imgsz), InterpolationMode.BILINEAR),  
        T.ToTensor()
    ]

    preprocess = {
        "train": T.Compose(tr_T),
        "val": T.Compose(val_T)
    } 
        
    if args.seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)

    tr_set = My_Dataset(pj(args.data, "train"), preprocess["train"])         
    if args.val_r == 0:
        val_set = My_Dataset(pj(args.data, "val"), preprocess["val"])
    else:
        val_set = tr_set.split_(args.val_r, preprocess["val"])
                                            
    class_names = tr_set.classes
    class_num = len(class_names)

    batch_sampler = BatchSampler(My_Sampler(tr_set, args.seed), args.bs, False)
                
    ds = {
        'train':DataLoader(
            tr_set,
            batch_size=1,
            shuffle=False,
            batch_sampler=batch_sampler,
            pin_memory=True,
            num_workers=args.worker
        ), 
        'val':DataLoader(
            val_set,
            batch_size=args.bs,
            shuffle=False,
            pin_memory=True,
            num_workers=args.worker
        )
    }
                
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.weight is None:
        model = ExquisiteNetV2(class_num, tr_set[0][0].shape[0]).to(device)
    else:
        model = torch.load(args.weight, device)
        model.fc = nn.Linear(384, class_num).to(device)

    opt = get_optimizer(model, args.init_lr, args.wd)
    # loss function  
    lf = nn.CrossEntropyLoss()
    # lr scheduler     
    lr_schdr = ReduceLROnPlateau( 
        opt,
        factor=args.lr_df,
        mode='min',
        patience=5-1, 
        verbose=True
    )
    
    print()
    print("="*50)
    print("You are training the data:", args.data)
    print("Model weight will be saved to:", args.save_dir)
    print("Device:", device)
    print(f"num of params: {sum(i.numel() for i in model.parameters()) / 1e6 :.4f} M")
    print("train num:", len(tr_set))
    print("val num:", len(val_set))
    print("single data shape:", tr_set[0][0].shape)
    print("="*50)
    if args.amp:
        print("train with auto mixed precision training!!!")
    print("\nFrom epoch 2, you can watch tensorboard, by "
          f"typing 'tensorboard --logdir={args.save_dir}/tensorboard'")
    print("\nstart training...\n")

    tr_time, end_epoch = train_acc(
        model,
        args.save_dir, 
        ds, 
        lf, 
        opt, 
        lr_schdr, 
        args.epoch, 
        device, 
        args.end_lr, 
        args.amp,
    )
    
    print("\nHas loaded the best weight")

    print("\n", '-'*50)
    print(' '*20, "Evaluate:")

    #load best weight & model
    model = torch.load(pj(args.save_dir, "md.pt"), device)
    model.eval()

    tr_loss, tr_acc = eval_acc(
        model, 
        ds['train'], 
        lf, 
        device,
        "train" 
    )
    print(f"Train loss: {tr_loss:.4f}")
    print(f"Train Acc: {tr_acc:.4f}")

    val_loss, val_acc = eval_acc(
        model, 
        ds["val"], 
        lf, 
        device,
        "val" 
    )          
    print('-'*20)
    print(f"Val loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")

    test_set = My_Dataset(
        pj(args.data, "test"),
        preprocess["val"]
    )

    ds["test"] = DataLoader(
        test_set,
        batch_size = args.bs,
        shuffle = False,
        pin_memory = True,
        num_workers = args.worker
    )

    test_loss, test_acc = eval_acc(
        model, 
        ds["test"], 
        lf, 
        device,
        "test" 
    )          
    print('-'*20)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    
    print('-'*20)
    print(f"Training time: {tr_time:.2f} seconds")    
    print(f"Total epochs:", end_epoch)    
    print(f"Best weight and tensorboard is in", args.save_dir) 
    print()
    
if __name__ == '__main__':
    main()


