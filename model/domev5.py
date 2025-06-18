# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import numpy as np
import random
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed
from mmengine.registry import MODELS
from mmengine.model import BaseModule



# for i in sys.path:
#     print(i)

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert
from .bev_cod import BEV_concat_net_s, BEV_condition_net


def downsample_visible_mask(input_tensor, downsample_size=8):
    B, T, L, W, H = input_tensor.shape
    input_tensor = input_tensor.view(B * T, L, W, H)
    # 定义降采样后的尺寸
    output_height = int(L/downsample_size)
    output_width = int(W/downsample_size)

    # 计算每个 pillar 的尺寸
    pillar_height = input_tensor.size(1) // output_height
    pillar_width = input_tensor.size(2) // output_width
    pillar_depth = input_tensor.size(3)

    # 将输入张量重塑为 [12, 50, 4, 50, 4, 16]
    reshaped_tensor = input_tensor.view(B*T, output_height, pillar_height, output_width, pillar_width, pillar_depth)

    # 统计每个 pillar 中 True（不可见）的数量
    true_count = reshaped_tensor.sum(dim=(2, 4, 5))

    # 计算每个 pillar 中 False（可见）的数量
    false_count = (pillar_height * pillar_width * pillar_depth) - true_count

    # 判断每个 pillar 是否应该被标记为 True（不可见）
    output_tensor = true_count > false_count

    output_tensor = output_tensor.view(B, T, output_height, output_width)

    # # 输出结果
    # print(output_tensor.shape)  # 应该是 [12, 50, 50]
    return output_tensor

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BEVDropout_layer_map(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, dropout_prob, use_3d=False, stride=8):
        super().__init__()
        # use_cfg_embedding = dropout_prob > 0
        # self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        # self.num_classes = num_classes
        if use_3d:
            self.maxpool = nn.MaxPool3d(kernel_size=(1, stride, stride), stride=(1, stride, stride), padding=(0, 0, 0))
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0)
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            print("Use BEV Dropout!")

    def token_drop(self, BEV_layout):
        """Drops labels to enable classifier-free guidance."""
        if torch.rand(1) < self.dropout_prob:
            BEV_null = -torch.ones_like(BEV_layout, device=BEV_layout.device)
            BEV_layout = BEV_null
        # drop_mask = torch.rand_like(BEV_layout,device=BEV_layout.device) < self.dropout_prob
        # BEV_layout[drop_mask] =-1
        return BEV_layout

    def forward(self, BEV_layout):
        use_dropout = self.dropout_prob > 0

        BEV_layout = self.maxpool(BEV_layout)
        if self.training and use_dropout:
            BEV_layout = self.token_drop(BEV_layout)
        # embeddings = self.embedding_table(labels)
        return BEV_layout

class BEVDropout_layer(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, dropout_prob, use_3d=False, stride=8):
        super().__init__()
        # use_cfg_embedding = dropout_prob > 0
        # self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        # self.num_classes = num_classes
        if use_3d:
            self.maxpool = nn.MaxPool3d(kernel_size=(16, stride, stride), stride=(16, stride, stride), padding=(0, 0, 0))
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0)
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            print("Use BEV Dropout!")

    def token_drop(self, BEV_layout):
        """Drops labels to enable classifier-free guidance."""
        if torch.rand(1) < self.dropout_prob:
            BEV_null = -torch.ones_like(BEV_layout, device=BEV_layout.device)
            BEV_layout = BEV_null
        # drop_mask = torch.rand_like(BEV_layout,device=BEV_layout.device) < self.dropout_prob
        # BEV_layout[drop_mask] =-1
        return BEV_layout

    def forward(self, BEV_layout):
        use_dropout = self.dropout_prob > 0

        BEV_layout = self.maxpool(BEV_layout)
        if self.training and use_dropout:
            BEV_layout = self.token_drop(BEV_layout)
        # embeddings = self.embedding_table(labels)
        return BEV_layout

class BEVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=3, stride=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=False, padding=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        return x


class MLP_meta(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        print(f"MLP_meta: {input_size}, {hidden_size}, {output_size}")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        if self.training and self.dropout_prob > 0:
            if torch.rand(1) < self.dropout_prob:
                x_null = torch.zeros_like(x, device=x.device)
                x = x_null
        return x
#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
            q_xf = q.transpose(1,2).contiguous()
            k_xf = k.transpose(1,2).contiguous()
            v_xf = v.transpose(1,2).contiguous()
            x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Dome Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A Dome tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        if skip:
            self.skip_norm = nn.LayerNorm(
                2 * hidden_size, elementwise_affine=True, eps=1e-6
            )
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

    def forward(self, x, c, skip = None):
        # Long Skip Connection
        if self.skip_linear is not None and skip is not None:
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1) # x:torch.Size([22, 625, 768])
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class InterLayerBlock(nn.Module):
    """
    A Dome tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of Dome.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

@MODELS.register_module()
class DomeV5(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        extras=1,
        attention_mode='math',
        pose_encoder=None,
        delta_input=False,
        pose_key_in_meta='rel_poses',
        bev_dropout_prob=0,
        control_dropout_prob=0.1,
        bev_in_ch=1,
        bev_out_ch=1,
        use_bev_concat=False,
        direct_concat=False,
        use_label=False,
        use_meta=False,
    ):
        super().__init__()
        self.control_dropout_prob = control_dropout_prob
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels + bev_out_ch if use_bev_concat else in_channels
        self.use_bev_concat = use_bev_concat
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames

        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # if self.extras == 2:
        #     self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # if self.extras == 78: # timestep + text_embedding
        #     self.text_embedding_projection = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(77 * 768, hidden_size, bias=True)
        # )

        if use_label:
            self.y_embedder = BEV_condition_net()
        if use_meta:
            print(f"Use Meta embedding! meta_num:{meta_num}")
            # self.meta_embedder = nn.Linear(meta_num,hidden_size)
            # self.meta_embedder = MLP_meta(meta_num,64,hidden_size,bev_dropout_prob)
            self.meta_embedder = MLP_meta(meta_num, int(hidden_size / 2), hidden_size, bev_dropout_prob)
        
        
        if self.use_bev_concat:
            if direct_concat == False:
                # self.bev_concat = BEV_concat_net(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
                self.bev_concat = BEV_concat_net_s(BEV_in_ch=bev_in_ch, BEV_out_ch=bev_out_ch)
            else:
                print("Use MaxPool3d!")
                # self.bev_concat = nn.MaxPool3d(kernel_size=(1,4,4), stride=(1,4,4), padding=(0,0,0))
                self.bev_concat = BEVDropout_layer_map(bev_dropout_prob, use_3d=True)

        self.mask_invisible = BEVDropout_layer(0, use_3d=True)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.hidden_size =  hidden_size

        self.depth = depth
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode, skip=layer >= depth // 2 ) for layer in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        if pose_encoder is not None:
            self.pose_encoder = MODELS.build(pose_encoder)
        self.initialize_weights()

        self.delta_input=delta_input
        self.pose_key_in_meta=pose_key_in_meta
    
    def set_trainable(self):
        for name, param in self.named_parameters():
            if "skip_" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.extras == 2:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in Dome blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
        
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, 
                x, 
                t, 
                y=None, 
                text_embedding=None, 
                use_fp16=False,
                metas=None,
                pose_st_offset=0,
                controls=None,
                condition=None,
                bev_layout=None,
                invisible_mask=None,
                ):
        """
        Forward pass of Dome.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)
        
        if self.use_bev_concat:
            x = torch.cat([x, self.bev_concat(bev_layout)], dim=2)
        

        invisible_mask_downsampled = downsample_visible_mask(invisible_mask.float()).unsqueeze(2)
        
        if len(controls)>0:
            # invisible_mask_downsampled = self.mask_invisible(invisible_mask.float().permute(0,1,4,2,3))
            B, T, C, H, W = invisible_mask_downsampled.shape
            invisible_mask_downsampled_past = torch.zeros([invisible_mask_downsampled.shape[0], int(controls[0].shape[0] / invisible_mask_downsampled.shape[0]) - invisible_mask_downsampled.shape[1], invisible_mask_downsampled.shape[2], invisible_mask_downsampled.shape[3], invisible_mask_downsampled.shape[4]], dtype=torch.bool, device=invisible_mask_downsampled.device)
            invisible_mask_downsampled = torch.cat([invisible_mask_downsampled_past, invisible_mask_downsampled], dim=1)
            invisible_mask_downsampled = rearrange(invisible_mask_downsampled, 'b f c h w -> (b f) (c h w)').unsqueeze(-1).repeat(1,1,controls[0].shape[2])
            for control in controls:
                control[invisible_mask_downsampled.bool()] = 0


        batches, frames, channels, high, weight = x.shape 
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed  
        t = self.t_embedder(t, use_fp16=use_fp16)                  
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=frames)#self.temp_embed.shape[1]) 
        timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1])

        if self.extras == 2:
            assert False
            y = self.y_embedder(y, self.training)
            y_spatial = repeat(y, 'n d -> (n c) d', c=frames)#self.temp_embed.shape[1]) 
            y_temp = repeat(y, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        elif self.extras == 78:
            assert False
            text_embedding = self.text_embedding_projection(text_embedding.reshape(batches, -1))
            text_embedding_spatial = repeat(text_embedding, 'n d -> (n c) d', c=frames)#self.temp_embed.shape[1])
            text_embedding_temp = repeat(text_embedding, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        if hasattr(self,'pose_encoder') and metas is not None:
            try:
                rel_poses, _ = self._get_pose_feature(metas, frames,st_offset=pose_st_offset) #b f 128
                # # pose_embedding_spatial = rearrange(rel_poses, 'n f d -> (n f) 1 d')
                # # x+=pose_embedding_spatial
                if getattr(self.pose_encoder,'do_proj'):
                    ### new version
                    pose_embedding_spatial = repeat(rel_poses, 'n d -> (n c) d', c=frames)#self.temp_embed.shape[1])
                    pose_embedding_temp = repeat(rel_poses, 'n d -> (n c) d', c=self.pos_embed.shape[1])
                else:
                    ## old version
                    pose_embedding_spatial = rearrange(rel_poses, 'n f d -> (n f) d')
                    pose_embedding_temp = repeat(rel_poses.mean(dim=1), 'n d -> (n c) d', c=self.pos_embed.shape[1])
            except:
                print('@'*50,'too long, generation w/o pose')
                metas=None

        skips = []

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            
            layer = i

            if layer >= self.depth // 2 :
                if controls is not None and len(controls) > 0:
                    # skip = controls.pop()
                    skip = skips.pop() + controls.pop()
                else:
                    skip = skips.pop()
            else:
                skip = None

            if self.extras == 2:
                c = timestep_spatial + y_spatial
            elif self.extras == 78:
                c = timestep_spatial + text_embedding_spatial
            elif hasattr(self,'pose_encoder') and metas is not None:
                c = timestep_spatial + pose_embedding_spatial
            else:
                c = timestep_spatial
            # c=c[:frames]#.to(x.dtype) #debug
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(spatial_block), x, c, skip, use_reentrant=False)       # (N, T, D)
            # x  = spatial_block(x, c)

            if layer < (self.depth // 2 ):
                skips.append(x)
            layer = i + 1


            if layer >= self.depth // 2 :
                if controls is not None and len(controls) > 0:
                    # skip = controls.pop()
                    skip = skips.pop() + controls.pop()
                else:
                    skip = skips.pop()

                skip = rearrange(skip, '(b f) t d -> (b t) f d', b=batches)

            else:
                skip = None

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed[:,:frames]

            if self.extras == 2:
                c = timestep_temp + y_temp
            elif self.extras == 78:
                c = timestep_temp + text_embedding_temp 
            elif hasattr(self,'pose_encoder') and metas is not None:
                c = timestep_temp + pose_embedding_temp
            else:
                c = timestep_temp
            
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(temp_block), x, c, skip, use_reentrant=False)       # (N, T, D)
            # x = temp_block(x, c)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)

            if layer < (self.depth // 2 ):
                skips.append(x)

        if self.extras == 2:
            c = timestep_spatial + y_spatial
        else:
            c = timestep_spatial
        # c=c[:frames]#.to(x.dtype) #debug
        x = self.final_layer(x, c)               
        x = self.unpatchify(x)                  
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        return x

    def forward_with_cfg(self, x, t, y=None, cfg_scale=7.0, use_fp16=False, text_embedding=None):
        """
        Forward pass of Dome, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)
        model_out = self.forward(combined, t, y=y, use_fp16=use_fp16, text_embedding=text_embedding)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :4, ...], model_out[:, :, 4:, ...] 
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0) 
        return torch.cat([eps, rest], dim=2)
    

    def _get_pose_feature(self, metas=None, F=None,st_offset=0):
        rel_poses, output_metas = None, None
        if not hasattr(self, 'pose_encoder'):
            return rel_poses, output_metas
        assert metas is not None
        output_metas = []
        pose_key_in_meta=self.pose_key_in_meta #'rel_poses'
        for meta in metas:
            # record them for loss
            output_meta = dict()
            output_meta[pose_key_in_meta] = meta[pose_key_in_meta]#[self.offset:]
            output_metas.append(output_meta)

        rel_poses = np.array([meta[pose_key_in_meta] for meta in metas]) #(2, 11, 2) bf2
        
        rel_poses = torch.tensor(rel_poses).float().cuda()# list of (num_frames+offsets, 2)
        if self.delta_input:
            rel_poses_pre = torch.cat([torch.zeros_like(rel_poses[:, :1]), rel_poses[:, :-1]], dim=1)
            rel_poses = rel_poses - rel_poses_pre
        assert st_offset+F<=rel_poses.shape[1]
        rel_poses = rel_poses[:, st_offset:st_offset+F, :]
        b,f=rel_poses.shape[:2]
        rel_poses = self.pose_encoder(rel_poses)
        return rel_poses, output_metas

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   Dome Configs                                  #
#################################################################################

def Dome_XL_2(**kwargs):
    return DomeV2(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def Dome_XL_4(**kwargs):
    return DomeV2(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def Dome_XL_8(**kwargs):
    return DomeV2(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def Dome_L_2(**kwargs):
    return DomeV2(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def Dome_L_4(**kwargs):
    return DomeV2(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def Dome_L_8(**kwargs):
    return DomeV2(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def Dome_B_2(**kwargs):
    return DomeV2(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def Dome_B_4(**kwargs):
    return DomeV2(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def Dome_B_8(**kwargs):
    return DomeV2(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def Dome_S_2(**kwargs):
    return DomeV2(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def Dome_S_4(**kwargs):
    return DomeV2(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def Dome_S_8(**kwargs):
    return DomeV2(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


Dome_models = {
    'Dome-XL/2': Dome_XL_2,  'Dome-XL/4': Dome_XL_4,  'Dome-XL/8': Dome_XL_8,
    'Dome-L/2':  Dome_L_2,   'Dome-L/4':  Dome_L_4,   'Dome-L/8':  Dome_L_8,
    'Dome-B/2':  Dome_B_2,   'Dome-B/4':  Dome_B_4,   'Dome-B/8':  Dome_B_8,
    'Dome-S/2':  Dome_S_2,   'Dome-S/4':  Dome_S_4,   'Dome-S/8':  Dome_S_8,
}

if __name__ == '__main__':

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(3, 16, 4, 32, 32).to(device)
    t = torch.tensor([1, 2, 3]).to(device)
    y = torch.tensor([1, 2, 3]).to(device)
    network = Dome_XL_2().to(device)
    from thop import profile 
    flops, params = profile(network, inputs=(img, t))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # y_embeder = LabelEmbedder(num_classes=101, hidden_size=768, dropout_prob=0.5).to(device)
    # lora.mark_only_lora_as_trainable(network)
    # out = y_embeder(y, True)
    # out = network(img, t, y)
    # print(out.shape)
