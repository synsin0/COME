import numpy as np
import torch
from torch.nn.functional import pad
from tabulate import tabulate

from mmengine.dist import get_rank, all_gather
from .base_evaluator import BaseEvaluator
from occforecasting.registry import EVALUATORS


@EVALUATORS.register_module()
class ADE(BaseEvaluator):
    def __init__(self, timestamps=('1s', '2s', '3s'), dataset=None):
        super().__init__(dataset)
        self._name = 'ADE'
        self.timestamps = timestamps

    def clean(self):
        self.ade = None
        self.matrix_l2 = None
        self.states = {}

    def update(self, inputs_dict, outputs_dict):
        pred, target = outputs_dict['pred'], outputs_dict['target']
        assert pred.shape == target.shape, f'{pred.shape} != {target.shape}'
        assert pred.shape[-1] == 16

        batch_size, num_timestamps, num_features = pred.size()
        assert num_timestamps == len(self.timestamps)

        pred = pred.view(batch_size, num_timestamps, 3, 4)
        target = target.view(batch_size, num_timestamps, 3, 4)
        # trajectory location
        pred_xyz = pred[:, :, :3, 3]
        target_xyz = target[:, :, :3, 3]
        # rotation matrix
        pred_matrix = pred[:, :, :3, :3]
        target_matrix = target[:, :, :3, :3]

        # calculation
        ade = torch.mean(
            torch.norm(pred_xyz - target_xyz, dim=-1), 
            dim=0)
        matrix_l2 = torch.mean(
            torch.norm(pred_matrix - target_matrix, dim=(-2, -1)), 
            dim=0)
        if self.ade is None:
            self.ade = ade
            self.matrix_l2 = matrix_l2
        else:
            self.ade = torch.cat((self.ade, ade), dim=0)
            self.matrix_l2 = torch.cat((self.matrix_l2, matrix_l2), dim=0)

    def eval(self):
        ade = torch.stack(all_gather(self.ade), dim=1)
        matrix_l2 = torch.stack(all_gather(self.matrix_l2), dim=1)
        for t_i, t in enumerate(self.timestamps):
            self.states[f'ADE_{t}'] = ade[t_i].mean().item()
            self.states[f'MatrixL2_{t}'] = matrix_l2[t_i].mean().item()
        return self.states

    def format_string(self):
        headers = ['timestamp'] + self.timestamps
        contents = []

        _contents = ['ADE']
        _contents.extend([self.states[f'ADE_{t}'] for t in self.timestamps])
        contents.append(_contents)

        _contents = ['MatrixL2']
        _contents.extend([self.states[f'MatrixL2_{t}'] for t in self.timestamps])
        contents.append(_contents)
        
        return tabulate(contents, headers=headers, tablefmt='orgtbl')