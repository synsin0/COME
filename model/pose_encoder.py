import torch.nn as nn
import torch
from mmengine.registry import MODELS
from mmengine.model import BaseModule
from einops import rearrange, repeat
import torch.nn.functional as F

@MODELS.register_module()
class PoseEncoder(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1
        
        pose_encoder = []

        for _ in range(num_layers - 1):
            pose_encoder.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU(True)])
            in_channels = out_channels
        pose_encoder.append(nn.Linear(out_channels, out_channels))
        self.pose_encoder = nn.Sequential(*pose_encoder)
    
    def forward(self,x):
        # x: N*2,
        pose_feat = self.pose_encoder(x)
        return pose_feat

      
@MODELS.register_module()
class PoseEncoder_fourier(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
        fourier_freqs=8,
        init_cfg=None,
        do_proj=False,
        max_length=77,
        # zero_init=False
        **kwargs
    ):
        super().__init__(init_cfg)
        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1
        # assert in_channels==2,"only support 2d coordinates for now, include gt_mode etc later"
        self.fourier_freqs=fourier_freqs
        self.position_dim = fourier_freqs * 2 * in_channels  # 2: sin/cos, 2: xy
        in_channels=self.position_dim

    
        pose_encoder = []
        for _ in range(num_layers - 1):
            pose_encoder.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU(True)])
            in_channels = out_channels
        pose_encoder.append(nn.Linear(out_channels, out_channels))
        self.pose_encoder = nn.Sequential(*pose_encoder)

        # self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
        self.do_proj=do_proj
        self.max_length=max_length
        if do_proj:
            # proj b*f*c -> b*c
            self.embedding_projection =nn.Linear(max_length * out_channels, out_channels, bias=True)
            self.null_position_feature = torch.nn.Parameter(torch.zeros([out_channels]))
        # if zero_init:
        #     self.zero_module()
    
    def zero_module(self):
        """
        Zero out the parameters of a module and return it.
        """
        for p in self.parameters():
            p.detach().zero_()

    def forward(self,x,mask=None):
        # x: N*2,
        b,f=x.shape[:2]
        x = rearrange(x, 'b f d -> (b f) d')
        x=get_fourier_embeds_from_coordinates(self.fourier_freqs,x) # N*dim (bf)*32 # 2,11,32
        # if mask is not None: #TODO
        #     # learnable null embedding
        #     xyxy_null = self.null_position_feature.view(1, 1, -1)
        #     # replace padding with learnable null embedding
        #     xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null
        x = self.pose_encoder(x) #([2, 11, 768])
        if self.do_proj:
            x = rearrange(x, '(b f) d -> b f d', b=b) # 2,11,32
            x_pad=F.pad(x,(0,0,0,self.max_length-f)) # 2,77,32
            xyxy_null = self.null_position_feature.view(1, 1, -1)
            mask=torch.zeros(b,self.max_length,1).to(x.device)
            mask[:,:f]=1
            x = x_pad * mask + (1 - mask) * xyxy_null
            x = rearrange(x, 'b f d -> b (f d)')
            x=self.embedding_projection(x) # b d
        else:
            x = rearrange(x, '(b f) d -> b f d', b=b)

        return x
        
@MODELS.register_module()
class PoseEncoder_fourier_yaw(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
        fourier_freqs=8,
        init_cfg=None,
        do_proj=False,
        max_length=77,
        # zero_init=False
        **kwargs
    ):
        super().__init__(init_cfg)
        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1
        # assert in_channels==2,"only support 2d coordinates for now, include gt_mode etc later"
        self.fourier_freqs=fourier_freqs
        self.position_dim = fourier_freqs * 2 * (in_channels-1)  # 2: sin/cos, 2: xy
        self.position_dim_yaw = fourier_freqs * 2 * (1)  # 2: sin/cos, 2: xy

    
        in_channels=self.position_dim
        pose_encoder = []
        for _ in range(num_layers - 1):
            pose_encoder.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU(True)])
            in_channels = out_channels
        pose_encoder.append(nn.Linear(out_channels, out_channels))
        self.pose_encoder = nn.Sequential(*pose_encoder)

        in_channels=self.position_dim_yaw
        pose_encoder_yaw = []
        for _ in range(num_layers - 1):
            pose_encoder_yaw.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU(True)])
            in_channels = out_channels
        pose_encoder_yaw.append(nn.Linear(out_channels, out_channels))
        self.pose_encoder_yaw = nn.Sequential(*pose_encoder_yaw)

        # self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
        self.do_proj=do_proj
        self.max_length=max_length
        if do_proj:
            # proj b*f*c -> b*c
            self.embedding_projection =nn.Linear(max_length * out_channels, out_channels, bias=True)
            self.null_position_feature = torch.nn.Parameter(torch.zeros([out_channels]))
            self.embedding_projection_yaw =nn.Linear(max_length * out_channels, out_channels, bias=True)
            self.null_position_feature_yaw = torch.nn.Parameter(torch.zeros([out_channels]))
        # if zero_init:
        #     self.zero_module()
    
    def zero_module(self, zero_params=None):
        """
        Zero out the parameters of a module based on the given list of parameter names and return it.
        
        Args:
            zero_params (list): List of parameter names to zero. If None, all parameters will be zeroed.
        """
        for name, p in self.named_parameters():
            if zero_params is None or name in zero_params:
                p.detach().zero_()

    def forward(self,x,mask=None):
        # x: N*2,
        b,f=x.shape[:2]
        x = rearrange(x, 'b f d -> (b f) d')
        x,x_yaw=x[:,:-1],x[:,-1:]
        x=get_fourier_embeds_from_coordinates(self.fourier_freqs,x) # N*dim (bf)*32 # 2,11,32
        x_yaw=get_fourier_embeds_from_coordinates(self.fourier_freqs,x_yaw) # N*dim (bf)*32 # 2,11,32
        # if mask is not None: #TODO
        #     # learnable null embedding
        #     xyxy_null = self.null_position_feature.view(1, 1, -1)
        #     # replace padding with learnable null embedding
        #     xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null
        x = self.pose_encoder(x) #([2, 11, 768])
        x_yaw = self.pose_encoder_yaw(x_yaw) #([2, 11, 768])
        if self.do_proj:
            x = rearrange(x, '(b f) d -> b f d', b=b) # 2,11,32
            x_pad=F.pad(x,(0,0,0,self.max_length-f)) # 2,77,32
            xyxy_null = self.null_position_feature.view(1, 1, -1)
            mask=torch.zeros(b,self.max_length,1).to(x.device)
            mask[:,:f]=1
            x = x_pad * mask + (1 - mask) * xyxy_null
            x = rearrange(x, 'b f d -> b (f d)')
            x=self.embedding_projection(x) # b d
        else:
            x = rearrange(x, '(b f) d -> b f d', b=b)
        
        if self.do_proj:
            x_yaw = rearrange(x_yaw, '(b f) d -> b f d', b=b) # 2,11,32
            x_pad=F.pad(x_yaw,(0,0,0,self.max_length-f)) # 2,77,32
            xyxy_null = self.null_position_feature.view(1, 1, -1)
            mask=torch.zeros(b,self.max_length,1).to(x_yaw.device)
            mask[:,:f]=1
            x_yaw = x_pad * mask + (1 - mask) * xyxy_null
            x_yaw = rearrange(x_yaw, 'b f d -> b (f d)')
            x_yaw=self.embedding_projection(x_yaw) # b d
        else:
            x_yaw = rearrange(x_yaw, '(b f) d -> b f d', b=b)

        x=x+x_yaw

        return x
        
def get_fourier_embeds_from_coordinates(embed_dim, xys):
    """
    Args:
        embed_dim: int
        xys: a 3-D tensor [B x N x 4] representing the bounding boxes for GLIGEN pipeline
    Returns:
        [B x N x embed_dim] tensor of positional embeddings
    """

    batch_size = xys.shape[0]
    ch= xys.shape[-1]

    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None].to(device=xys.device, dtype=xys.dtype)
    emb = emb * xys.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 2, 3, 1).reshape(batch_size, embed_dim * 2 * ch)

    return emb


