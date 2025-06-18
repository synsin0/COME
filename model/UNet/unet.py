# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from mmengine.model import BaseModule, bias_init_with_prob, normal_init

from .modules import UpConvBlock, BasicConvBlock, BottleNeckBlock
from .base_forecaster import BaseForecaster

from mmengine.registry import MODELS



@MODELS.register_module()
class UNet(BaseForecaster):
    """UNet.

    This backbone is the implementation of `U-Net: Convolutional Networks
    for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_.

    Args:
        in_channels (int): Number of input image channels. Default" 3.
        base_channels (int): Number of base channels of each stage.
            The output channels of the first stage. Default: 64.
        num_stages (int): Number of stages in encoder, normally 5. Default: 5.
        strides (Sequence[int 1 | 2]): Strides of each stage in encoder.
            len(strides) is equal to num_stages. Normally the stride of the
            first stage in encoder is 1. If strides[i]=2, it uses stride
            convolution to downsample in the correspondence encoder stage.
            Default: (1, 1, 1, 1, 1).
        enc_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence encoder stage.
            Default: (2, 2, 2, 2, 2).
        dec_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence decoder stage.
            Default: (2, 2, 2, 2).
        downsamples (Sequence[int]): Whether use MaxPool to downsample the
            feature map after the first stage of encoder
            (stages: [1, num_stages)). If the correspondence encoder stage use
            stride convolution (strides[i]=2), it will never use MaxPool to
            downsample, even downsamples[i-1]=True.
            Default: (True, True, True, True).
        enc_dilations (Sequence[int]): Dilation rate of each stage in encoder.
            Default: (1, 1, 1, 1, 1).
        dec_dilations (Sequence[int]): Dilation rate of each stage in decoder.
            Default: (1, 1, 1, 1).
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Notice:
        The input image size should be divisible by the whole downsample rate
        of the encoder. More detail of the whole downsample rate can be found
        in UNet._check_input_divisible.
    """

    def __init__(self,
                 base_channels=256,
                 num_stages=5,
                 block_type='BasicConv',
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 enc_channels=None,
                 dec_channels=None,
                 temporal_num_convs=2,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 losses=None,
                 **kwargs):
        super().__init__(**kwargs)
        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, '\
            f'while the strides is {strides}, the length of '\
            f'strides is {len(strides)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_num_convs) == num_stages, \
            'The length of enc_num_convs should be equal to num_stages, '\
            f'while the enc_num_convs is {enc_num_convs}, the length of '\
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_num_convs) == (num_stages-1), \
            'The length of dec_num_convs should be equal to (num_stages-1), '\
            f'while the dec_num_convs is {dec_num_convs}, the length of '\
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(downsamples) == (num_stages-1), \
            'The length of downsamples should be equal to (num_stages-1), '\
            f'while the downsamples is {downsamples}, the length of '\
            f'downsamples is {len(downsamples)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_dilations) == num_stages, \
            'The length of enc_dilations should be equal to num_stages, '\
            f'while the enc_dilations is {enc_dilations}, the length of '\
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_dilations) == (num_stages-1), \
            'The length of dec_dilations should be equal to (num_stages-1), '\
            f'while the dec_dilations is {dec_dilations}, the length of '\
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        src_T, src_Z = self.source_seq_size[:2]
        tgt_T, tgt_Z = self.target_seq_size[:2]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        if enc_channels is None:
            enc_channels = [base_channels * 2**i for i in range(num_stages)]
        if dec_channels is None:
            dec_channels = enc_channels
        assert len(enc_channels) == len(dec_channels) == num_stages
        assert block_type in ['BasicConv', 'BottleNeck']
        Block = BasicConvBlock if block_type == 'BasicConv' else BottleNeckBlock

        in_channels = self.sem_embedding_dim * src_T * src_Z
        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=Block,
                        in_channels=dec_channels[i],
                        skip_channels=enc_channels[i - 1],
                        mid_channels=dec_channels[i - 1],
                        out_channels=dec_channels[i - 1],
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if upsample else None))

            enc_conv_block.append(
                Block(
                    in_channels=in_channels,
                    out_channels=enc_channels[i],
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.encoder.append(nn.Sequential(*enc_conv_block))
            in_channels = enc_channels[i]
        
        self.temporal_stem = nn.Conv2d(
            dec_channels[0], base_channels * tgt_T, kernel_size=3, stride=1, padding=1)
        self.temporal_shared_branch = BasicConvBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            num_convs=temporal_num_convs,
            stride=1,
            dilation=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.occ_sem_conv = nn.Conv2d(
            base_channels, self.num_classes * tgt_Z, kernel_size=3, stride=1, padding=1)
        
        self.occ_losses = None
        if losses is not None:
            losses = {name: MODELS.build(loss) for name, loss in losses.items()}
            self.occ_losses = nn.ModuleDict(losses)
    
    def init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.occ_sem_conv, std=0.01, bias=bias_cls)

    def forward(self, inputs_dict):
        x, targets = self.pre_process(inputs_dict)
        B, T, C, Z, H, W = x.shape
        x = x.reshape(B, -1, H, W)
        self._check_input_divisible(x)

        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)
        
        tgt_T, tgt_Z, tgt_H, tgt_W = self.target_seq_size

        x = self.temporal_stem(x)
        x = x.reshape(B * tgt_T, -1, H, W)
        x = self.temporal_shared_branch(x)
        sem_preds = self.occ_sem_conv(x)
        sem_preds = sem_preds.reshape(B, tgt_T, -1, tgt_Z, H, W)
        sem_preds = sem_preds[..., :tgt_H, :tgt_W]

        if self.training:
            return self.losses(sem_preds, targets, inputs_dict)
        else:
            return self.post_process(sem_preds, inputs_dict)
    
    def losses(self, sem_preds, targets, inputs_dict):
        sem_preds = sem_preds.permute(
            0, 1, 3, 4, 5, 2).reshape(-1, self.num_classes)
        targets = targets.reshape(-1)

        # remove targets == -1
        sem_preds = sem_preds[targets != -1]
        targets = targets[targets != -1]

        avg_factor = (targets != (self.num_classes - 1)).sum()

        losses = dict()
        for loss_name, loss in self.occ_losses.items():
            losses[loss_name] = loss(sem_preds, targets, avg_factor=avg_factor)
        return losses

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'