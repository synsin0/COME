
import numpy as np, torch
from torch.utils import data
import torch.nn.functional as F
from copy import deepcopy
from mmengine import MMLogger
logger = MMLogger.get_instance('genocc')
try:
    from . import OPENOCC_DATAWRAPPER
except:
    from mmengine.registry import Registry
    OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')
import torch

@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes(data.Dataset):
    CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
               'vegetation', 'free')
    MOVING_CLASSES = ('bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                      'pedestrian', 'trailer', 'truck')
    PALETTE = ([255, 158, 0], [255, 99, 71], [255, 140, 0], [255, 69, 0], [233, 150, 70],
               [220, 20, 60], [255, 61, 99], [0, 0, 230], [47, 79, 79], [112, 128, 144],
               [0, 207, 191], [175, 0, 75], [75, 0, 75], [112, 180, 60], [222, 184, 135],
               [0, 175, 0], [0, 255, 0], [0, 0, 0])
    SCENE_RANGE = (-40, -40, -1, 40, 40, 5.4)
    VOXEL_SIZE = (0.4, 0.4, 0.4)
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):
        input, target, metas = self.point_cloud_dataset[index]
        #### adapt to vae input
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        return input, target, metas
        

def custom_collate_fn_temporal(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, torch.Tensor):
            data_tuple.append(torch.stack([d[i] for d in data]))
        elif isinstance(item, (dict, str)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple
