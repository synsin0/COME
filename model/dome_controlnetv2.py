import math
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed
from mmengine.registry import MODELS
from mmengine.model import BaseModule

from .domev2 import *

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


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



@MODELS.register_module()
class DomeControlNetV2(nn.Module):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Inherit PeftAdapterMixin to be compatible with the PEFT training pipeline.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=14,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        extras=1,
        attention_mode='math',
        bev_dropout_prob=0,
        bev_in_ch=1,
        bev_out_ch=1,
        use_bev_concat=True,
        direct_concat=True,
        pose_encoder=None,
        delta_input=False,
        pose_key_in_meta='rel_poses'
    ):
        super().__init__()
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
        # self.control_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        if self.extras == 2:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if self.extras == 78: # timestep + text_embedding
            self.text_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(77 * 768, hidden_size, bias=True)
        )

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.hidden_size =  hidden_size

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode) for _ in range(depth)
        ])

        if pose_encoder is not None:
            self.pose_encoder = MODELS.build(pose_encoder)
        # self.initialize_weights()

        self.delta_input=delta_input
        self.pose_key_in_meta=pose_key_in_meta

        # Input zero linear for the first block
        self.before_proj = zero_module(nn.Linear(self.hidden_size, self.hidden_size))

        # Output zero linear for the every block
        self.after_proj_list = nn.ModuleList(
            [
                zero_module(nn.Linear(self.hidden_size, self.hidden_size))
                for _ in range(len(self.blocks))
            ]
        )

        
        if self.use_bev_concat:
            if direct_concat == False:
                # self.bev_concat = BEV_concat_net(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
                self.bev_concat = BEV_concat_net_s(BEV_in_ch=bev_in_ch, BEV_out_ch=bev_out_ch)
            else:
                print("Use MaxPool3d!")
                # self.bev_concat = nn.MaxPool3d(kernel_size=(1,4,4), stride=(1,4,4), padding=(0,0,0))
                self.bev_concat = BEVDropout_layer(bev_dropout_prob, use_3d=True)


        self.fix_weight_modules = [
            "y_embedder",
            "x_embedder",
            "t_embedder",
            "extra_embedder",
            "pose_encoder",
        ]

    def from_dit(self, dit):
        """
        Load the parameters from a pre-trained HunYuanDiT model.

        Parameters
        ----------
        dit: HunYuanDiT
            The pre-trained HunYuanDiT model.
        """

        # self.text_embedding_projection.data = dit.text_embedding_projection.data
        # if self.args.use_style_cond:
        #     self.style_embedder.load_state_dict(dit.style_embedder.state_dict())
        if self.extras == 2:
            self.y_embedder.load_state_dict(dit.y_embedder.state_dict())
        if self.extras == 78: # timestep + text_embedding
            self.text_embedding_projection.data = dit.text_embedding_projection.data
        self.pos_embed.data = dit.pos_embed.data
        self.temp_embed.data = dit.temp_embed.data

        self.x_embedder.load_state_dict(dit.x_embedder.state_dict())
        self.t_embedder.load_state_dict(dit.t_embedder.state_dict())
        # self.extra_embedder.load_state_dict(dit.extra_embedder.state_dict())
        self.pose_encoder.load_state_dict(dit.pose_encoder.state_dict())

        for i, block in enumerate(self.blocks):
            block.load_state_dict(dit.blocks[i].state_dict())

        if self.use_bev_concat:
            self.bev_concat.load_state_dict(dit.bev_concat.state_dict())

    def set_trainable(self):

        # self.text_embedding_projection.requires_grad_(False)

        self.pose_encoder.requires_grad_(False)
        self.x_embedder.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        self.temp_embed.requires_grad_(False)
        # self.y_embedder.requires_grad_(False)
        self.t_embedder.requires_grad_(False)
        # self.extra_embedder.requires_grad_(False)

        self.blocks.requires_grad_(True)
        self.before_proj.requires_grad_(True)
        self.after_proj_list.requires_grad_(True)

        self.blocks.train()
        self.before_proj.train()
        self.after_proj_list.train()

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward


    def forward(
        self,
        x, 
        t, 
        condition,
        y=None, 
        text_embedding=None, 
        use_fp16=False,
        metas=None,
        pose_st_offset=0,
        return_dict=True,
        invisible_mask=None,
        bev_layout=None,
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
            condition = torch.cat([condition, self.bev_concat(bev_layout)], dim=2)
       

        batches, frames, channels, high, weight = x.shape 
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        condition = rearrange(condition, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed  
        t = self.t_embedder(t, use_fp16=use_fp16)                  
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=frames)#self.temp_embed.shape[1]) 
        timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1])

        condition = self.x_embedder(condition)

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

        
        controls = []
        x = x + self.before_proj(condition)  # add condition
        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]

            layer = i
            if self.extras == 2:
                c = timestep_spatial + y_spatial
            elif self.extras == 78:
                c = timestep_spatial + text_embedding_spatial
            elif hasattr(self,'pose_encoder') and metas is not None:
                c = timestep_spatial + pose_embedding_spatial
            else:
                c = timestep_spatial
            # c=c[:frames]#.to(x.dtype) #debug
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(spatial_block), x, c,use_reentrant=False)       # (N, T, D)
            # x  = spatial_block(x, c)
            
            controls.append(self.after_proj_list[layer](x)) # zero linear for output


            layer = i + 1


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
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(temp_block), x, c,use_reentrant=False)       # (N, T, D)
            # x = temp_block(x, c)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)

            controls.append(self.after_proj_list[layer](x)) # zero linear for output

        if return_dict:
            return {"controls": controls}
        return controls


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
