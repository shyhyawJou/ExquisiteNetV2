import torch
from torchvision import transforms as T


class To_mode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, img):
        if img.mode != self.mode:
            img = img.convert(self.mode)
        return img

class RandomColorJitter(T.ColorJitter):
    def __init__(self, p, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(
            brightness = brightness, 
            contrast = contrast, 
            saturation = saturation, 
            hue = hue
        )
        self.p = p
        self.my_color_jitter = (brightness, contrast, saturation, hue)
        
    def __call__(self, img):
        if self.p <= torch.rand(1):
            return img
        return super().forward(img)           

class RandomTranslate(T.RandomAffine):
    def __init__(self, p, translate):
        super().__init__(
            degrees = 0, 
            translate = translate
        )
        self.p = p

    def __call__(self, img):
        if self.p <= torch.rand(1):
            return img        
        return super().forward(img)

