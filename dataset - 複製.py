from pathlib import Path as p
import numpy as np
from PIL import Image
import random
from copy import deepcopy

import torch


class My_Dataset:
    def __init__(self, folder, transform=None, int_class=False):
        self.img_paths = np.asarray(sorted([str(i) for i in sorted(p(folder).glob('**/*')) if i.suffix in [
                            ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", 
                            ".PNG", ".tiff", ".TIFF", ".bmp", ".BMP", 
                            ".gif", ".GIF"]]))
        self.classes = sorted(
            [i.name for i in p(self.img_paths[0]).parents[1].iterdir() if i.is_dir()]
        ) 
        self.transform = transform
        self.dict = dict(zip(self.classes, torch.arange(len(self.classes))))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)        
        label = self.dict[p(img_path).parts[-2]]
        if self.transform != None:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.img_paths)

    def split_(self, val_ratio, transform=None):
        random.seed(10)
        data_paths, data_num = self.paths_in_every_class()
        self.img_paths = []
        new_set = deepcopy(self)
        new_set.img_paths = []
        new_set.transform = transform
        for class_name, paths in data_paths.items():
            random.shuffle(paths)
            sub_num = int(data_num[class_name] * val_ratio)
            new_set.img_paths.extend(paths[:sub_num])
            self.img_paths.extend(paths[sub_num:])
        self.img_paths = np.asarray(sorted(self.img_paths))
        new_set.img_paths = np.asarray(sorted(new_set.img_paths))
        return new_set

    def paths_in_every_class(self):
        data_path = {i: [] for i in self.classes}
        data_num = {i: 0 for i in self.classes} # every num of data in a class
        for path in self.img_paths:
            class_name = p(path).parts[-2]
            data_path[class_name].append(path)
            data_num[class_name] += 1
        return data_path, data_num

class My_Sampler(torch.utils.data.Sampler):
    def __init__(self, my_dset, init_seed):
        self.ds_num = len(my_dset)
        self.seed = init_seed
        
    def __iter__(self):
        order = torch.randperm(self.ds_num)
        return iter(order)

    def __len__(self):
        return self.ds_num

