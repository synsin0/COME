import numpy as np
import torch
from torch.nn.functional import pad
from tabulate import tabulate

from mmengine.dist import get_rank, all_gather
from .base_evaluator import BaseEvaluator
from occforecasting.registry import EVALUATORS


@EVALUATORS.register_module()
class MIoU(BaseEvaluator):

    def __init__(self, ignore_label=-1, gt_sem_key='target_occs', with_iou=True, dataset=None):
        super().__init__(dataset)
        self._name = 'MIoU'
        self.ignore_label = ignore_label
        self.with_iou = with_iou
        self.free_label = self._classes.index('free') if 'free' in self._classes else None
        self.clean()
        self.gt_sem_key = gt_sem_key

    def clean(self):
        self.num_A = torch.zeros((0, len(self._classes))).int().cuda()
        self.num_B = torch.zeros((0, len(self._classes))).int().cuda()
        self.num_AB = torch.zeros((0, len(self._classes))).int().cuda()

        if self.with_iou and self.free_label is not None:
            self.iou_num_A = torch.zeros((0,)).int().cuda()
            self.iou_num_B = torch.zeros((0,)).int().cuda()
            self.iou_num_AB = torch.zeros((0,)).int().cuda()

        self.states = {}
    
    def update(self, inputs_dict, outputs_dict):
        outputs, targets = outputs_dict['sem_preds'], inputs_dict[self.gt_sem_key]
        if 'target_mask_camera' in inputs_dict:
            target_mask_camera = inputs_dict['target_mask_camera']
            outputs[target_mask_camera] = self.ignore_label
            targets[target_mask_camera] = self.ignore_label


        assert outputs.shape == targets.shape, \
            'The size of outputs and targets should be the same, ' \
            f'but get outputs:{outputs.shape} and targets:{targets.shape}'
        
        if isinstance(outputs, np.ndarray):
            outputs = self.num_A.new_tensor(outputs)
        if isinstance(targets, np.ndarray):
            targets = self.num_A.new_tensor(targets)
        assert isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor)
        outputs, targets = outputs.int().cuda(), targets.int().cuda()

        for output, target in zip(outputs, targets):
            output = output[target!=self.ignore_label]
            target = target[target!=self.ignore_label]

            num_A = torch.bincount(output, minlength=len(self._classes))
            num_B = torch.bincount(target, minlength=len(self._classes))
            num_AB = torch.bincount(output[output == target], minlength=len(self._classes))

            self.num_A = torch.cat([self.num_A, num_A.unsqueeze(0)], dim=0)
            self.num_B = torch.cat([self.num_B, num_B.unsqueeze(0)], dim=0)
            self.num_AB = torch.cat([self.num_AB, num_AB.unsqueeze(0)], dim=0)
        
        if not self.with_iou or self.free_label is None:
            return None

        occ_outputs = outputs_dict['occ_preds'] if 'occ_preds' in outputs_dict \
            else (outputs != self.free_label).int()
        occ_targets = (targets != self.free_label).int()
        assert occ_outputs.shape == occ_targets.shape, \
            'The size of occ outputs and occ targets should be the same, ' \
            f'but get outputs:{occ_outputs.shape} and targets:{occ_targets.shape}'

        if isinstance(occ_outputs, np.ndarray):
            occ_outputs = self.num_A.new_tensor(occ_outputs)
        assert isinstance(occ_outputs, torch.Tensor)
        occ_outputs = occ_outputs.cuda()

        for output, target in zip(occ_outputs, occ_targets):
            output = output[target!=self.ignore_label]
            target = target[target!=self.ignore_label]

            num_A = (output == 1).sum()
            num_B = (target == 1).sum()
            num_AB = ((output == 1) & (target == 1)).sum()

            self.iou_num_A = pad(self.iou_num_A, (0, 1), value=num_A)
            self.iou_num_B = pad(self.iou_num_B, (0, 1), value=num_B)
            self.iou_num_AB = pad(self.iou_num_AB, (0, 1), value=num_AB)
    
    def eval(self):
        num_A = torch.stack(all_gather(self.num_A), dim=1).flatten(end_dim=1)
        num_B = torch.stack(all_gather(self.num_B), dim=1).flatten(end_dim=1)
        num_AB = torch.stack(all_gather(self.num_AB), dim=1).flatten(end_dim=1)

        self.num_A = num_A[:self._len]
        self.num_B = num_B[:self._len]
        self.num_AB = num_AB[:self._len]

        total_num_A = self.num_A.sum(dim=0).float()
        total_num_B = self.num_B.sum(dim=0).float()
        totoal_num_AB = self.num_AB.sum(dim=0).float()
        IoUs = totoal_num_AB / (total_num_A + total_num_B - totoal_num_AB)

        state_list = []
        for c, IoU in zip(self._classes, IoUs):
            if c == 'free':
                continue
            state_list.append((f'IoU_{c}', IoU.item()))
        MIoU = sum([item[1] for item in state_list]) / len(state_list)

        self.states.update({k:v for k, v in state_list})
        self.states['MIoU'] = MIoU

        if not self.with_iou or self.free_label is None:
            return None
        
        iou_num_A = torch.stack(all_gather(self.iou_num_A), dim=1).flatten(end_dim=1)
        iou_num_B = torch.stack(all_gather(self.iou_num_B), dim=1).flatten(end_dim=1)
        iou_num_AB = torch.stack(all_gather(self.iou_num_AB), dim=1).flatten(end_dim=1)

        self.iou_num_A = iou_num_A[:self._len]
        self.iou_num_B = iou_num_B[:self._len]
        self.iou_num_AB = iou_num_AB[:self._len]
        Occ_IoU = self.iou_num_AB.float().sum() / (self.iou_num_A.float().sum() 
                                                   + self.iou_num_B.float().sum()
                                                   - self.iou_num_AB.float().sum())
        self.states['Occ_IoU'] = Occ_IoU.item()

        return self.states
    
    def format_string(self):
        contents = []
        for c in self._classes:
            if c != 'free':
                value = self.states[f'IoU_{c}']
                contents.append([c, f'{value*100:.2f}'])
        
        if self.with_iou and self.free_label is not None:
            contents.append(['Occ_IoU', f'{self.states["Occ_IoU"]*100:.2f}'])
        contents.append(['MIoU', f'{self.states["MIoU"]*100:.2f}'])
        return tabulate(contents, tablefmt='orgtbl')


@EVALUATORS.register_module()
class SeqMIoU(BaseEvaluator):

    def __init__(self, ignore_label=-1, gt_sem_key='target_occs', with_iou=True, timestamps=('1s', '2s', '3s'), dataset=None):
        super().__init__(dataset)
        self._name = 'SeqMIoU'
        self.ignore_label = ignore_label
        self.timestamps = timestamps
        self.with_iou = with_iou
        self.free_label = self._classes.index('free') if 'free' in self._classes else None
        self.gt_sem_key = gt_sem_key
    
    def clean(self):
        self.num_A = torch.zeros((0, len(self.timestamps), len(self._classes))).int().cuda()
        self.num_B = torch.zeros((0, len(self.timestamps), len(self._classes))).int().cuda()
        self.num_AB = torch.zeros((0, len(self.timestamps), len(self._classes))).int().cuda()
        
        if self.with_iou and self.free_label is not None:
            self.iou_num_A = torch.zeros((0, len(self.timestamps))).int().cuda()
            self.iou_num_B = torch.zeros((0, len(self.timestamps))).int().cuda()
            self.iou_num_AB = torch.zeros((0, len(self.timestamps))).int().cuda()

        self.states = {}

    def update(self, inputs_dict, outputs_dict):
        seq_outputs, seq_targets = outputs_dict['sem_preds'], inputs_dict[self.gt_sem_key]
        if 'target_mask_camera' in inputs_dict:
            target_mask_camera = inputs_dict['target_mask_camera']
            seq_outputs[target_mask_camera] = self.ignore_label
            seq_targets[target_mask_camera] = self.ignore_label


        assert seq_outputs.shape == seq_targets.shape, \
            'The size of outputs and targets should be the same, ' \
            f'but get outputs:{seq_outputs.shape} and targets:{seq_targets.shape}'
        B, T = seq_outputs.shape[:2]
        assert len(self.timestamps) == T

        if isinstance(seq_outputs, np.ndarray):
            seq_outputs = self.num_A.new_tensor(seq_outputs)
        if isinstance(seq_targets, np.ndarray):
            seq_targets = self.num_A.new_tensor(seq_targets)
        assert isinstance(seq_outputs, torch.Tensor) and isinstance(seq_targets, torch.Tensor)
        seq_outputs, seq_targets = seq_outputs.int().cuda(), seq_targets.int().cuda()

        num_As, num_Bs, num_ABs = [], [], []
        outputs, targets = seq_outputs.flatten(end_dim=1), seq_targets.flatten(end_dim=1)
        for output, target in zip(outputs, targets):
            output = output[target!=self.ignore_label]
            target = target[target!=self.ignore_label]

            num_As.append(torch.bincount(output, minlength=len(self._classes)))
            num_Bs.append(torch.bincount(target, minlength=len(self._classes)))
            num_ABs.append(torch.bincount(output[output == target], minlength=len(self._classes)))
        
        num_A = torch.stack(num_As, dim=0).reshape(B, T, len(self._classes))
        num_B = torch.stack(num_Bs, dim=0).reshape(B, T, len(self._classes))
        num_AB = torch.stack(num_ABs, dim=0).reshape(B, T, len(self._classes))

        self.num_A = torch.cat([self.num_A, num_A], dim=0)
        self.num_B = torch.cat([self.num_B, num_B], dim=0)
        self.num_AB = torch.cat([self.num_AB, num_AB], dim=0)

        if not self.with_iou or self.free_label is None:
            return None
        
        occ_seq_outputs = outputs_dict['occ_preds'] if 'occ_preds' in outputs_dict \
            else (seq_outputs != self.free_label).int()
        occ_seq_targets = (seq_targets != self.free_label).int()
        assert occ_seq_outputs.shape == occ_seq_targets.shape, \
            'The size of occ outputs and occ targets should be the same, ' \
            f'but get outputs:{occ_seq_outputs.shape} and targets:{occ_seq_targets.shape}'
        
        if isinstance(occ_seq_outputs, np.ndarray):
            occ_seq_outputs = self.num_A.new_tensor(occ_seq_outputs)
        occ_seq_outputs = occ_seq_outputs.int().cuda()
        
        outputs = occ_seq_outputs.flatten(end_dim=1)
        targets = occ_seq_targets.flatten(end_dim=1)
        num_As, num_Bs, num_ABs = [], [], []
        for output, target in zip(outputs, targets):
            output = output[target!=self.ignore_label]
            target = target[target!=self.ignore_label]

            num_As.append((output == 1).sum())
            num_Bs.append((target == 1).sum())
            num_ABs.append(((output == 1) & (target == 1)).sum())
        
        num_As = self.iou_num_A.new_tensor(num_As).reshape(B, T)
        num_Bs = self.iou_num_B.new_tensor(num_Bs).reshape(B, T)
        num_ABs = self.iou_num_AB.new_tensor(num_ABs).reshape(B, T)

        self.iou_num_A = torch.cat([self.iou_num_A, num_As], dim=0)
        self.iou_num_B = torch.cat([self.iou_num_B, num_Bs], dim=0)
        self.iou_num_AB = torch.cat([self.iou_num_AB, num_ABs], dim=0)
    
    def eval(self):
        num_A = torch.stack(all_gather(self.num_A), dim=1).flatten(end_dim=1)
        num_B = torch.stack(all_gather(self.num_B), dim=1).flatten(end_dim=1)
        num_AB = torch.stack(all_gather(self.num_AB), dim=1).flatten(end_dim=1)

        self.num_A = num_A[:self._len]
        self.num_B = num_B[:self._len]
        self.num_AB = num_AB[:self._len]

        total_num_A = self.num_A.sum(dim=0).float()
        total_num_B = self.num_B.sum(dim=0).float()
        totoal_num_AB = self.num_AB.sum(dim=0).float()
        IoUs = totoal_num_AB / (total_num_A + total_num_B - totoal_num_AB)

        for t_i, t in enumerate(self.timestamps):
            t_IoUs = IoUs[t_i]
            state_list = []
            for c, IoU in zip(self._classes, t_IoUs):
                if c == 'free':
                    continue
                state_list.append((f'IoU_{c}_{t}', IoU.item()))
            MIoU = sum([item[1] for item in state_list]) / len(state_list)

            self.states.update({k:v for k, v in state_list})
            self.states[f'MIoU_{t}'] = MIoU
        
        if not self.with_iou or self.free_label is None:
            return None
        
        iou_num_A = torch.stack(all_gather(self.iou_num_A), dim=1).flatten(end_dim=1)
        iou_num_B = torch.stack(all_gather(self.iou_num_B), dim=1).flatten(end_dim=1)
        iou_num_AB = torch.stack(all_gather(self.iou_num_AB), dim=1).flatten(end_dim=1)

        self.iou_num_A = iou_num_A[:self._len]
        self.iou_num_B = iou_num_B[:self._len]
        self.iou_num_AB = iou_num_AB[:self._len]

        iou_total_A = self.iou_num_A.sum(dim=0).float()
        iou_total_B = self.iou_num_B.sum(dim=0).float()
        iou_total_AB = self.iou_num_AB.sum(dim=0).float()
        Occ_IoUs = iou_total_AB / (iou_total_A + iou_total_B - iou_total_AB)

        for t_i, t in enumerate(self.timestamps):
            self.states[f'Occ_IoU_{t}'] = Occ_IoUs[t_i].item()
        
        return self.states

    def format_string(self):
        headers = ['timestamp'] + self.timestamps
        contents = []
        for c in self._classes:
            if c == 'free':
                continue
            _contents = [c]
            for t in self.timestamps:
                value = self.states[f'IoU_{c}_{t}']
                _contents.append(f'{value*100:.2f}')
            contents.append(_contents)
        
        if self.with_iou and self.free_label is not None:
            _contents = ['Occ_IoU']
            for t in self.timestamps:
                value = self.states[f'Occ_IoU_{t}']
                _contents.append(f'{value*100:.2f}')
            contents.append(_contents)
        
        _contents = ['MIoU']
        for t in self.timestamps:
            value = self.states[f'MIoU_{t}']
            _contents.append(f'{value*100:.2f}')
        contents.append(_contents)
        return tabulate(contents, headers=headers, tablefmt='orgtbl')
        