# if set "use_mask" to False, model will be ExquisiteNetV2_Faster_RCNN
# if there are no mask files in the data directory, please set use_mask to False
import util

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN, MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import torch

def ExquisiteNetV2_Mask_RCNN(class_num, use_mask):
    backbone = util.ExquisiteNetV2(class_num).features
    backbone.out_channels = 384
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                    output_size=7,
                                    sampling_ratio=2)
    
    mask_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                         output_size=14,
                                         sampling_ratio=2)
    if use_mask:
        model = MaskRCNN(backbone,
                         num_classes=class_num,
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler,
                         mask_roi_pool=mask_roi_pooler)
    else:
        model = FasterRCNN(backbone,
                           num_classes=class_num,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    
    params_num = sum([p.numel() for p in model.parameters()])
    tr_params_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(model)
    print('The model has %d parameters' %params_num)
    print('The model has %d trainable parameters' %tr_params_num)
    print()
    return model