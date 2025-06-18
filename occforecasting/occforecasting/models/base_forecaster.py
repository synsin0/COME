import os
import os.path as osp
import numpy as np
import pickle
import math
from mmengine.model import BaseModule

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import MODELS


@MODELS.register_module()
class BaseForecaster(BaseModule):

    def __init__(self,
                 source_seq_size=(6, 16, 200, 200),
                 target_seq_size=(6, 16, 200, 200),
                 num_classes=18,
                 align_source_coors=False,
                 align_target_coors=False,
                 recover_target_coors=False,
                 sem_encode_type='embedding',
                 sem_embedding_dim=32,
                 with_t_embedding=False,
                 with_z_embedding=False,
                 size_divisor=16,
                 act_type='softmax',
                 class_weights=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.source_seq_size = source_seq_size
        self.target_seq_size = target_seq_size
        self.num_classes = num_classes
        self.align_source_coors = align_source_coors
        self.align_target_coors = align_target_coors
        self.recover_target_coors = recover_target_coors
        self.class_weights = class_weights

        T, Z, H, W = self.source_seq_size
        assert sem_encode_type in ['embedding', 'onehot', 'conv', 'none']
        if sem_encode_type == 'embedding':
            self.sem_embedding = nn.Embedding(self.num_classes, sem_embedding_dim)
            if with_t_embedding:
                self.t_embedding = nn.Embedding(T, sem_embedding_dim)
            if with_z_embedding:
                self.z_embedding = nn.Embedding(Z, sem_embedding_dim)
        elif sem_encode_type != 'none':
            in_channels = self.num_classes if sem_encode_type == 'onehot' else 1
            self.sem_encode_conv = nn.Conv2d(in_channels, sem_embedding_dim, 1, 1)
        self.sem_encode_type = sem_encode_type
        self.sem_embedding_dim = 1 if self.sem_encode_type == 'none' else sem_embedding_dim
        self.with_t_embedding = with_t_embedding
        self.with_z_embedding = with_z_embedding
        self.size_divisor = size_divisor
    
    def source_pre_process(self, inputs_dict):
        sources = inputs_dict['source_occs'].cuda()
        source_metas = inputs_dict['source_metas']
        metas = inputs_dict['metas']
        # judge the source shape
        assert sources.shape[1:] == tuple(self.source_seq_size), \
            (f'Require source occs shape as: {self.source_seq_size}, '
             f'but got {sources.shape[1:]}')
        B, T, Z, H, W = sources.shape
        invalid_mask = sources.new_zeros(B, T, Z, H, W, dtype=torch.bool)

        if self.align_source_coors:
            src_matrix, dst_matrix = [], []
            for meta in source_metas:
                src_matrix.extend(meta['ego2global'])
                dst_matrix.extend([meta['ego2global'][-1] for _ in range(T)])

            sources = sources.view(B * T, Z, H, W)
            sources, mask = self.transform_coor_system(
                sources, metas[0]['scene_range'], src_matrix, dst_matrix)
            sources, mask = sources.reshape(B, T, Z, H, W), mask.reshape(B, T, Z, H, W)
            invalid_mask[mask] = True

        _H = math.ceil(H / self.size_divisor) * self.size_divisor
        _W = math.ceil(W / self.size_divisor) * self.size_divisor
        sources = F.pad(sources, (0, _W - W, 0, _H - H), 'constant', 0)
        invalid_mask = F.pad(invalid_mask, (0, _W - W, 0, _H - H), 'constant', True)

        if self.sem_encode_type == 'none':
            sources = sources.float().unsqueeze(2)
        elif self.sem_encode_type == 'embedding':
            sources = self.sem_embedding(sources.long())
            sources = sources.permute(0, 1, 5, 2, 3, 4) # B, T, C, Z, H, W
            if self.with_t_embedding:
                t_embed = self.t_embedding(torch.arange(T).cuda().int())
                sources = sources + t_embed.reshape(1, T, self.sem_embedding_dim, 1, 1, 1)
            if self.with_z_embedding:
                z_embed = self.z_embedding(torch.arange(Z).cuda().int())
                sources = sources + z_embed.T.reshape(1, 1, self.sem_embedding_dim, Z, 1, 1)
        else:
            if self.sem_encode_type == 'onehot':
                sources = F.one_hot(sources, self.num_classes).permute(0, 1, 5, 2, 3, 4)
            sources = sources.reshape(B * T, -1, Z * _H, _W) # B*T, C(1 or num cls), Z*H, W
            sources = self.sem_encode_conv(sources.float())
            sources = sources.reshape(B, T, -1, Z, _H, _W)
        invalid_mask = invalid_mask.unsqueeze(2).expand(sources.shape) 
        
        sources[invalid_mask] = 0
        return sources
    
    def target_pre_process(self, inputs_dict):
        if 'target_occs' not in inputs_dict:
            return None

        targets = inputs_dict['target_occs'].cuda()
        target_metas = inputs_dict['target_metas']
        source_metas = inputs_dict['source_metas']
        metas = inputs_dict['metas']

        # judge the target shape
        assert targets.shape[1:] == tuple(self.target_seq_size), \
            (f'Require target occs shape as: {self.target_seq_size}, '
             f'but got {targets.shape[1:]}')
        B, T, Z, H, W = targets.shape
        invalid_mask = targets.new_zeros(B, T, Z, H, W, dtype=torch.bool)

        if self.align_target_coors:
            src_matrix, dst_matrix = [], []
            for source_meta, target_meta in zip(source_metas, target_metas):
                src_matrix.extend(target_meta['ego2global'])
                dst_matrix.extend([source_meta['ego2global'][-1] for _ in range(T)])
            
            targets = targets.view(B * T, Z, H, W)
            targets, mask = self.transform_coor_system(
                targets, metas[0]['scene_range'], src_matrix, dst_matrix)
            targets, mask = targets.reshape(B, T, Z, H, W), mask.reshape(B, T, Z, H, W)
            invalid_mask[mask] = True
        
        targets[invalid_mask] = -1
        return targets
    
    def pre_process(self, inputs_dict):
        sources = self.source_pre_process(inputs_dict)
        targets = self.target_pre_process(inputs_dict)
        return sources, targets

    def transform_coor_system(self, occs, scene_range, src_matrix, dst_matrix):
        scene_range = occs.new_tensor(scene_range)
        origin = (scene_range[:3] + scene_range[3:]) / 2
        scene_size = scene_range[3:] - scene_range[:3]

        Z, H, W = occs.shape[1:]
        x = torch.arange(0, W, dtype=occs.dtype, device=occs.device)
        x = (x + 0.5) / W * scene_size[0] + scene_range[0]
        y = torch.arange(0, H, dtype=occs.dtype, device=occs.device)
        y = (y + 0.5) / H * scene_size[1] + scene_range[1]
        z = torch.arange(0, Z, dtype=occs.dtype, device=occs.device)
        z = (z + 0.5) / Z * scene_size[2] + scene_range[2]
        xx = x[None, None, :].expand(Z, H, W)
        yy = y[None, :, None].expand(Z, H, W)
        zz = z[:, None, None].expand(Z, H, W)
        coors = torch.stack([xx, yy, zz], dim=-1)
        
        offsets = []
        for src_mat, dst_mat in zip(src_matrix, dst_matrix):
            src_mat = coors.new_tensor(src_mat)
            dst_mat = coors.new_tensor(dst_mat)

            coors_ = F.pad(coors.reshape(-1, 3), (0, 1), 'constant', 1)
            coors_ = coors_ @ dst_mat.T @ torch.inverse(src_mat).T
            offset = (coors_[:, :3] - origin) / scene_size * 2
            offsets.append(offset.reshape(Z, H, W, 3))
        offsets = torch.stack(offsets)

        occs = occs.unsqueeze(1).float()
        occs = F.grid_sample(occs, offsets, mode='nearest', align_corners=False)
        mask = (offsets.abs() > 1).any(-1)
        return occs.long().squeeze(1), mask
    
    def post_process(self, sem_preds, inputs_dict, **kwargs):
        if self.class_weights is not None:
            sem_preds = sem_preds.sigmoid() if self.act_type == 'sigmoid' \
                else sem_preds.softmax(dim=2)
            class_weights = sem_preds.new_tensor(self.class_weights).view(1, 1, -1, 1, 1, 1)
            sem_preds = sem_preds * class_weights
        
        sem_preds = sem_preds.argmax(dim=2)

        assert sem_preds.shape[1:] == tuple(self.target_seq_size), \
            (f'Require model predicted occs shape as: {self.target_seq_size}, '
             f'but got {sem_preds.shape[1:]}')
        B, T, Z, H, W = sem_preds.shape

        if self.recover_target_coors:
            src_matrix, dst_matrix = [], []
            source_metas, metas = inputs_dict['source_metas'], inputs_dict['metas']
            for i in range(B):
                src_matrix.extend([source_metas[i]['ego2global'][-1] for _ in range(T)])
                if 'ego2global' in kwargs:
                    dst_matrix.extend(kwargs['ego2global'])
                else:
                    assert 'target_metas' in inputs_dict
                    dst_matrix.extend(inputs_dict['target_metas'][i]['ego2global'])
            
            sem_preds = sem_preds.reshape(B * T, Z, H, W)
            sem_preds, invisible_mask = self.transform_coor_system(
                sem_preds, metas[0]['scene_range'], src_matrix, dst_matrix)
            
            sem_preds[invisible_mask] = self.num_classes - 1
            sem_preds = sem_preds.reshape(B, T, Z, H, W)
            invisible_mask = invisible_mask.reshape(B, T, Z, H, W)
        else:
            invisible_mask = None

        return dict(sem_preds=sem_preds, metas=inputs_dict['metas'], invisible_mask=invisible_mask)
