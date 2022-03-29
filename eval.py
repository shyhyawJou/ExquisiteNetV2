import argparse
from os.path import join as pj

import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import My_Dataset
from tool import eval_acc, del_ipynb_ckps
from img import *



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help="path of dataset")
    parser.add_argument('--weight', type=str, default=None, help="path of pretrained weight")
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--imgsz', type=int, default=224, help="image size")
    parser.add_argument('--worker', default=4)
    return parser.parse_args()


def main():
    args = get_args()
    # for jupyter notebook
    del_ipynb_ckps(args.data)
 
    preprocess = T.Compose([
        To_mode("RGB"), 
        T.Resize((args.imgsz, args.imgsz), InterpolationMode.BILINEAR),  
        T.ToTensor()
    ])
        
    val_set = My_Dataset(args.data, preprocess)                                                     
    class_names = val_set.classes
    class_num = len(class_names)
                
    ds = DataLoader(
        val_set,
        batch_size=args.bs,
        shuffle=False,
        pin_memory=True,
        num_workers=args.worker
    )
      
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(args.weight, device)
    # loss function  
    lf = nn.CrossEntropyLoss()
    
    print()
    print("="*50)
    print("You are evaluating the data:", args.data)
    print("Has loaded the weight from:", args.weight)
    print("Device:", device)
    print(f"num of params: {sum(i.numel() for i in model.parameters()) / 1e6 :.4f} M")
    print("Data num:", len(val_set))
    print("single data shape:", val_set[0][0].shape)
    print("="*50)
    print("\nstart evaluating...\n")

    val_loss, val_acc = eval_acc(
        model, 
        ds, 
        lf, 
        device,
        "val" 
    )          

    print('-'*20)
    print(f"Val loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}\n")


if __name__ == '__main__':
    main()


