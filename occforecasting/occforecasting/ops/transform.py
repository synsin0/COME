import torch
import torch.nn.functional as F


def transform_3d_grid(self, voxels, src_matrix, tgt_matrix):
    Z, H, W = voxels.shape[-3:]
    x = (torch.arange(0, W, device=voxels.device).float() + 0.5) / W - 0.5
    y = (torch.arange(0, H, device=voxels.device).float() + 0.5) / H - 0.5
    z = (torch.arange(0, Z, device=voxels.device).float() + 0.5) / Z - 0.5

    xx = x[None, None, :].expand(Z, H, W)
    yy = y[None, :, None].expand(Z, H, W)
    zz = z[:, None, None].expand(Z, H, W)
    coors = torch.stack([xx, yy, zz], dim=-1)
    
    offsets = []
    for src_mat, tgt_mat in zip(src_matrix, tgt_matrix):
        src_mat = coors.new_tensor(src_mat)
        dst_mat = coors.new_tensor(tgt_mat)
        coors_ = F.pad(coors.reshape(-1, 3), (0, 1), 'constant', 1)
        coors_ = coors_ @ dst_mat.T @ torch.inverse(src_mat).T
        offset = coors_[:, :3] * 2
        offsets.append(offset.reshape(Z, H, W, 3))
    offsets = torch.stack(offsets)
    voxels = F.grid_sample(voxels, offsets, mode='nearest', align_corners=False)
    return voxels, (offsets.abs() > 1).any(-1)


def transform_2d_grid(grids, src_matrix, tgt_matrix):
    H, W = feat.shape[-2:]
    x = (torch.arange(0, W, device=feat.device).float() + 0.5) / W - 0.5
    y = (torch.arange(0, H, device=feat.device).float() + 0.5) / H - 0.5

    xx = x[None, :].expand(H, W)
    yy = y[:, None].expand(H, W)
    coors = torch.stack([xx, yy], dim=-1)
    
    offsets = []
    for src_mat, dst_mat in zip(src_matrix, dst_matrix):
        src_mat = coors.new_tensor(src_mat)
        dst_mat = coors.new_tensor(dst_mat)

        coors_ = F.pad(coors.reshape(-1, 2), (0, 2), 'constant', 0)
        coors_[:, -1] = 1
        coors_ = coors_ @ dst_mat.T @ torch.inverse(src_mat).T
        offset = coors_[:, :2] * 2
        offsets.append(offset.reshape(H, W, 2))
    offsets = torch.stack(offsets)
    feat = F.grid_sample(feat, offsets, mode='nearest', align_corners=False)
    return feat, (offsets.abs() > 1).any(-1)