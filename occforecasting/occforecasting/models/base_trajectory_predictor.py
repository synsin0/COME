import os
import os.path as osp
import numpy as np
import pickle
import math
from mmengine.model import BaseModule, bias_init_with_prob, normal_init

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import MODELS

# class SubgraphNet_Layer(nn.Module):
#     def __init__(self, in_channels=128, hidden_channels=64):
#         super(SubgraphNet_Layer, self).__init__()
#         self.fc_layer = nn.Linear(in_channels, hidden_channels)
#         self.act = nn.ReLU()

#     def forward(self, x):
#         hidden = self.fc(x)
#         encode_data = F.relu(F.layer_norm(hidden, hidden.size()[1:]))
#         kernel_size = encode_data.size()[1]
#         maxpool = nn.MaxPool1d(kernel_size)
#         polyline_feature = maxpool(encode_data)
#         polyline_feature = polyline_feature.repeat(kernel_size)
#         output = torch.cat([encode_data, polyline_feature], 1)
#         return output

class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleEncoder, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(F.layer_norm(x, x.size()[1:]))
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super(SimpleDecoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


@MODELS.register_module()
class BaseTrajPredictor(nn.Module):
    def __init__(self, 
                 in_ts, 
                 in_channels, 
                 out_ts, 
                 out_channels,
                 intermediate_channels=128,
                 losses=None,
                 loss_weight_loc=1,
                 loss_weight_trans=1,
                **kwargs):
        """
        Args:
            in_ts (int): input time steps
            in_channels (int): input channels
            out_ts (int): output time steps
            out_channels (int): output channels
            intermediate_channels (int): intermediate channels
        """
        super().__init__(**kwargs)
        self.in_ts, self.out_ts = in_ts, out_ts
        self.out_channels = out_channels

        # encoding layers
        self.sublayer1 = SimpleEncoder(in_channels, intermediate_channels)
        self.sublayer2 = SimpleEncoder(intermediate_channels, intermediate_channels)
        self.sublayer3 = SimpleEncoder(intermediate_channels, intermediate_channels)
        self.maxpool = nn.MaxPool1d(in_ts)

        # decoder
        self.decoder = SimpleDecoder(intermediate_channels, out_ts * out_channels)

        # loss
        self.loss = nn.MSELoss(reduction='none')
        self.loss_weights = torch.ones(3, 4)
        self.loss_weights[:, 3] = loss_weight_loc
        self.loss_weights[:3, :3] = loss_weight_trans
        self.loss_weights = self.loss_weights.view(-1).cuda()

    def init_weights(self):
        pass

    def forward(self, batch_dict):
        x = batch_dict['source_traj'].cuda()
        y = batch_dict['target_traj'].cuda()

        batch_size, num_timestamps, num_features = x.size()
        assert num_timestamps == self.in_ts

        # encode feats
        x = x.view(-1, num_features)
        x = self.sublayer1(x)
        x = self.sublayer2(x)
        x = self.sublayer3(x)
        x = x.view(batch_size, num_timestamps, -1)
        x = self.maxpool(x.permute(0, 2, 1)).squeeze(2)

        # decode pred
        pred = self.decoder(x)
        pred = pred.view(batch_size, self.out_ts, self.out_channels)

        if self.training:
            return self.losses(pred, y)
        else:
            return dict(pred=pred, target=y)

    def losses(self, pred, target):        
        losses = dict()
        losses['traj_loss'] = torch.sum(
            self.loss(pred, target) * self.loss_weights)
        return losses
