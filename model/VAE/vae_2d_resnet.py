""" adopted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py """
# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import numpy as np
from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Tuple
from torch.nn import  Conv2d,Conv3d
from einops import rearrange

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
    
    def forward(self, x, shape):
        do_rearrange=False
        if x.dim()==5:
            # bchw or bcfhw
            b,_,f,_,_=x.size()
            do_rearrange=True
            x = rearrange(x, "b c f h w -> (b f) c h w")
        assert x.dim()==4
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        diffY = shape[0] - x.size()[2]
        diffX = shape[1] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])

        if self.with_conv:
            x = self.conv(x)
        if do_rearrange:
            x = rearrange(x, "(b f) c h w -> b c f h w",b=b,f=f)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
    
    def forward(self, x):
        do_rearrange=False
        if x.dim()==5:
            # bchw or bcfhw
            b,_,f,_,_=x.size()
            do_rearrange=True
            x = rearrange(x, "b c f h w -> (b f) c h w")
        assert x.dim()==4
        if self.with_conv:
            #pad = (0, 1, 0, 1, 0, 1)
            #x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        if do_rearrange:
            x = rearrange(x, "(b f) c h w -> b c f h w",b=b,f=f)
        return x
    

class TimeDownsample2x(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: int = 3
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.AvgPool3d((kernel_size,1,1), stride=(2,1,1),padding=(1,0,0))
        
    def forward(self, x):
        # first_frame_pad = x[:, :, :1, :, :].repeat(
        #     (1, 1, self.kernel_size - 1, 1, 1)
        # )# the first element will still be the same
        # x = torch.concatenate((first_frame_pad, x), dim=2)
        return self.conv(x)

class TimeUpsample2x(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out
    ):
        super().__init__()
    def forward(self, x):
        # if x.size(2) > 1:
        #     x,x_= x[:,:,:1],x[:,:,1:]
        #     x_= F.interpolate(x_, scale_factor=(2,1,1), mode='trilinear')
        #     x = torch.concat([x, x_], dim=2)
        x= F.interpolate(x, scale_factor=(2,1,1), mode='trilinear')
        return x
    

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h



class ResnetBlock3D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels,  kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels,  kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b, c, h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class AttnBlock3D(nn.Module):
    """Compatible with old versions, there are issues, use with caution."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = q.reshape(b * t, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b * t, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b * t, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, t, h, w)

        h_ = self.proj_out(h_)

        return x + h_


@MODELS.register_module()
class VAERes2D(BaseModule):
    def __init__(
            self, 
            encoder_cfg, 
            decoder_cfg,
            num_classes=18,
            expansion=8, 
            vqvae_cfg=None,
            scaling_factor=0.18215,
            init_cfg=None):
        super().__init__(init_cfg)

        self.expansion = expansion #8
        self.num_cls = num_classes

        self.encoder = MODELS.build(encoder_cfg)
        self.decoder = MODELS.build(decoder_cfg)
        self.class_embeds = nn.Embedding(num_classes, expansion)

        if vqvae_cfg:
            self.vqvae = MODELS.build(vqvae_cfg)
        self.use_vq = vqvae_cfg is not None
    
    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        sigma = torch.exp(z[:, dim:] / 2)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, sigma

    def forward_encoder(self, x):
        # x: bs, F, H, W, D
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x) # bs, F, H, W, D, c
        x = x.reshape(bs*F, H, W, D * self.expansion).permute(0, 3, 1, 2)

        z, shapes = self.encoder(x)
        return z, shapes
        
    def forward_decoder(self, z, shapes, input_shape):
        logits = self.decoder(z, shapes)

        bs, F, H, W, D = input_shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)
        template = self.class_embeds.weight.T.unsqueeze(0) # 1, expansion, cls
        similarity = torch.matmul(logits, template) # -1, D, cls
        # pred = similarity.argmax(dim=-1) # -1, D
        # pred = pred.reshape(bs, F, H, W, D)
        return similarity.reshape(bs, F, H, W, D, self.num_cls)

    def forward(self, x, **kwargs):
        # xs = self.forward_encoder(x)
        # logits = self.forward_decoder(xs)
        # return logits, xs[-1]
        # x :torch.Size([1, 10, 200, 200, 16])
        output_dict = {}
        z, shapes = self.forward_encoder(x) # torch.Size([10, 128, 50, 50])
        if self.use_vq:
            z_sampled, loss, info = self.vqvae(z, is_voxel=False)
            output_dict.update({'embed_loss': loss})
        else:
            z_sampled, z_mu, z_sigma = self.sample_z(z)
            output_dict.update({
                'z_mu': z_mu,
                'z_sigma': z_sigma})
        
        logits = self.forward_decoder(z_sampled, shapes, x.shape) #torch.Size([1, 10, 200, 200, 16, 18])
        
        output_dict.update({'logits': logits})
    
        if not self.training:
            pred = logits.argmax(dim=-1).detach().cuda()
            output_dict['sem_pred'] = pred
            pred_iou = deepcopy(pred)
            
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou
            
        return output_dict
        # loss, kl, rec = self.loss(logits, x, z_mu, z_sigma)
        # return loss, kl, rec
        
    def generate(self, z, shapes, input_shape):
        logits = self.forward_decoder(z, shapes, input_shape)
        return {'logits': logits}


@MODELS.register_module()
class VAERes3D(BaseModule):
    def __init__(
            self, 
            encoder_cfg, 
            decoder_cfg,
            num_classes=18,
            expansion=8, 
            vqvae_cfg=None,
            scaling_factor=0.18215,
            init_cfg=None):
        super().__init__(init_cfg)

        self.expansion = expansion #8
        self.num_cls = num_classes

        self.encoder = MODELS.build(encoder_cfg)
        self.decoder = MODELS.build(decoder_cfg)
        self.class_embeds = nn.Embedding(num_classes, expansion)

        if vqvae_cfg:
            self.vqvae = MODELS.build(vqvae_cfg)
        self.use_vq = vqvae_cfg is not None
    
    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        sigma = torch.exp(z[:, dim:] / 2)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, sigma

    def forward_encoder(self, x):
        # x: bs, F, H, W, D
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x) # bs, F, H, W, D, c # 16*8->128
        if isinstance(self.encoder,Encoder3D):
            x = rearrange(x,'b f h w d c -> b (d c) f h w')
            z, shapes = self.encoder(x)
        else:
            x = rearrange(x,'b f h w d c-> (b f) (d c) h w')
            z, shapes = self.encoder(x) #TODO check shape
            z = rearrange(z,'(b f) d h w -> b d f h w',b=bs)

        return z, shapes
        
    def forward_decoder(self, z, shapes, input_shape):
        assert z.dim()==5 and len(input_shape)==5
        logits = self.decoder(z, shapes)

        bs, F, H, W, D = input_shape
        # logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)
        logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, D, self.expansion)
        template = self.class_embeds.weight.T.unsqueeze(0) # 1, expansion, cls
        similarity = torch.matmul(logits, template) # -1, D, cls
        # pred = similarity.argmax(dim=-1) # -1, D
        # pred = pred.reshape(bs, F, H, W, D)
        return similarity.reshape(bs, F, H, W, D, self.num_cls)

    def forward(self, x, **kwargs):

        # xs = self.forward_encoder(x)
        # logits = self.forward_decoder(xs)
        # return logits, xs[-1]
        # x :torch.Size([4, 12, 200, 200, 16])
        output_dict = {}
        z, shapes = self.forward_encoder(x) # torch.Size([10, 128, 50, 50])
        if self.use_vq:
            z_sampled, loss, info = self.vqvae(z, is_voxel=False)
            output_dict.update({'embed_loss': loss})
        else:
            z_sampled, z_mu, z_sigma = self.sample_z(z)
            output_dict.update({
                'z_mu': z_mu,
                'z_sigma': z_sigma})
        
        logits = self.forward_decoder(z_sampled, shapes, x.shape) #torch.Size([1, 10, 200, 200, 16, 18])
        
        output_dict.update({'logits': logits})
    
        if not self.training:
            pred = logits.argmax(dim=-1).detach().cuda()
            output_dict['sem_pred'] = pred
            pred_iou = deepcopy(pred)
            
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou
            
        return output_dict
        # loss, kl, rec = self.loss(logits, x, z_mu, z_sigma)
        # return loss, kl, rec
        
    def generate(self, z, shapes, input_shape):
        logits = self.forward_decoder(z, shapes, input_shape)
        return {'logits': logits}

@MODELS.register_module()
class Encoder2D(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Enc has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        # x: bs, F, H, W, D
        shapes = []
        temb = None

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions-1:
                shapes.append(h.shape[-2:])
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        # middle
        # h = hs[-1]
        #
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, shapes

@MODELS.register_module()
class Decoder2D(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align with encoder
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Dec has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, shapes):
        # z: bs*F, C, H, W
        shapes=shapes.copy()
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align encoder
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, shapes.pop())

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


Module=str
@MODELS.register_module()
class Encoder3D(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, 
                    conv_in: Module = "Conv3d",
                    conv_out: Module = "Conv3d",
                    resnet_blocks: Tuple[Module] = (
                        "ResnetBlock3D",
                        "ResnetBlock3D",
                        "ResnetBlock3D",
                        "ResnetBlock3D",
                    ),
                    # temporal_downsample: Tuple[Module] = ("", "", "TimeDownsampleRes2x", ""),
                    temporal_downsample: Tuple[Module] = ("", "", "", ""),
                mid_resnet: Module = "ResnetBlock3D",

                **ignore_kwargs,
                 ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        assert len(resnet_blocks) == len(ch_mult)
        # downsampling
        self.conv_in = eval(conv_in)(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    # ResnetBlock
                    eval(resnet_blocks[i_level])(in_channels=block_in,
                                         out_channels=block_out,
                                        #  temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Enc has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock3D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            if temporal_downsample[i_level]:
                down.time_downsample = eval(temporal_downsample[i_level])(
                    block_in, block_in
                )
            self.down.append(down)


        # middle
        self.mid = nn.Module()
        self.mid.block_1 = eval(mid_resnet)(in_channels=block_in,
                                       out_channels=block_in,
                                    #    temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = eval(mid_resnet)(in_channels=block_in,
                                       out_channels=block_in,
                                    #    temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = eval(conv_out)(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        # x: bs, F, H, W, D
        shapes = []
        temb = None

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions-1:
                shapes.append(h.shape[-2:])
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)
            if hasattr(self.down[i_level], "time_downsample"):
                h = self.down[i_level].time_downsample(h)


        # middle
        # h = hs[-1]
        #
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, shapes

@MODELS.register_module()
class Decoder3D(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False,
                conv_in: Module = "Conv3d",
                conv_out: Module = "Conv3d",
                resnet_blocks: Tuple[Module] = (
                    "ResnetBlock3D",
                    "ResnetBlock3D",
                    "ResnetBlock3D",
                    "ResnetBlock3D",
                ),
                # temporal_upsample: Tuple[Module] = ("", "", "", "TimeUpsampleRes2x"),
                temporal_upsample: Tuple[Module] = ("", "", "", ""),
                mid_resnet: Module = "ResnetBlock3D",
                  **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        assert len(resnet_blocks) == len(ch_mult)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = eval(conv_in)(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = eval(mid_resnet)(in_channels=block_in,
                                       out_channels=block_in,
                                    #    temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = eval(mid_resnet)(in_channels=block_in,
                                       out_channels=block_in,
                                    #    temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align with encoder
                block.append(eval(resnet_blocks[i_level])(in_channels=block_in,
                                         out_channels=block_out,
                                        #  temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Dec has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock3D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            
            if temporal_upsample[i_level]:
                up.time_upsample = eval(temporal_upsample[i_level])(
                    block_in, block_in
                )
            self.up.insert(0, up) # prepend to get consistent order
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = eval(conv_out)(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, shapes):
        # z: bs*F, C, H, W
        shapes=shapes.copy()
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align encoder
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, shapes.pop())
            if hasattr(self.up[i_level], "time_upsample"):
                h = self.up[i_level].time_upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



if __name__ == "__main__":
    # test encoder
    import torch
    # encoder = Encoder2D(in_channels=3, ch=64, out_ch=64, ch_mult=(1,2,4,8), num_res_blocks=2, resolution=200,attn_resolutions=(100,50), z_channels=64, double_z=True)
    # #decoder = Decoder3D()
    # decoder = Decoder2D(in_channels=3, ch=64, out_ch=3, ch_mult=(1,2,4,8), num_res_blocks=2, resolution=200,attn_resolutions=(100,50), z_channels=64, give_pre_end=False)
    base_channel = 64
    _dim_ = 16
    expansion = 8 # class embedding 维度

    # encoder_cfg1=dict(
    #     type='Encoder2D',
    #     ch = base_channel, 
    #     out_ch = base_channel, 
    #     ch_mult = (1,2,4,8), 
    #     num_res_blocks = 2,
    #     attn_resolutions = (50,), 
    #     dropout = 0.0, 
    #     resamp_with_conv = True, 
    #     in_channels = _dim_ * expansion,
    #     resolution = 200, 
    #     z_channels = base_channel * 2, 
    #     double_z = False,
    # )
    # encoder1=MODELS.build(encoder_cfg1)

    # encoder_cfg2=dict(
    #     type='Encoder3D',
    #     ch = base_channel, 
    #     out_ch = base_channel, 
    #     ch_mult = (1,2,4,8), 
    #     num_res_blocks = 2,
    #     attn_resolutions = (50,), 
    #     dropout = 0.0, 
    #     resamp_with_conv = True, 
    #     in_channels = _dim_ * expansion,
    #     resolution = 200, 
    #     z_channels = base_channel * 2, 
    #     double_z = False,

    #     resnet_blocks= ("ResnetBlock",
    #                 "ResnetBlock",
    #                 "ResnetBlock",
    #                 "ResnetBlock3D",
    #                 ),
    #     # temporal_downsample= ("", "", "TimeDownsampleRes2x", ""),
    #     temporal_downsample= ("", "", "", ""),
    #     mid_resnet="ResnetBlock3D"
    # )
    # encoder2=MODELS.build(encoder_cfg2)
    decoder_cfg1=dict(
        type='Decoder2D',
        ch = base_channel, 
        out_ch = _dim_ * expansion, 
        ch_mult = (1,2,4,8), 
        num_res_blocks = 2,
        attn_resolutions = (50,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = _dim_ * expansion,
        resolution = 200, 
        z_channels = base_channel, 
        give_pre_end = False
    )
    decoder1=MODELS.build(decoder_cfg1)
    decoder_cfg2=dict(
        type='Decoder3D',
        ch = base_channel, 
        out_ch = _dim_ * expansion, 
        ch_mult = (1,2,4,8), 
        num_res_blocks = 2,
        attn_resolutions = (50,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = _dim_ * expansion,
        resolution = 200, 
        z_channels = base_channel, 
        give_pre_end = False,
        conv_in = "Conv3d",
        conv_out = "Conv3d",
        resnet_blocks= (
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        # temporal_upsample: Tuple[Module] = ("", "", "", "TimeUpsampleRes2x"),
        temporal_upsample = ("", "", "", ""),
        mid_resnet = "ResnetBlock3D",
    )
    decoder2=MODELS.build(decoder_cfg2)


    pass
    # import pdb; pdb.set_trace()