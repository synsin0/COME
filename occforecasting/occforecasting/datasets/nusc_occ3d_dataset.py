import os
import torch
import torch.nn.functional as F
import os.path as osp
import pickle
from torch.utils.data import Dataset
from pyquaternion import Quaternion
import numpy as np
import mmcv
from scipy.spatial.transform import Rotation

from occforecasting.registry import DATASETS


@DATASETS.register_module(force=__name__=='__main__')
class NuscOcc3DDataset(Dataset):
    CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
               'vegetation', 'free')
    MOVING_CLASSES = ('bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                      'pedestrian', 'trailer', 'truck')
    PALETTE = ([255, 158, 0], [255, 99, 71], [255, 140, 0], [255, 69, 0], [233, 150, 70],
               [220, 20, 60], [255, 61, 99], [0, 0, 230], [47, 79, 79], [112, 128, 144],
               [0, 207, 191], [175, 0, 75], [75, 0, 75], [112, 180, 60], [222, 184, 135],
               [0, 175, 0], [0, 255, 0], [0, 0, 0])
    SCENE_RANGE = (-40, -40, -1, 40, 40, 5.4)
    VOXEL_SIZE = (0.4, 0.4, 0.4)
    
    def __init__(self, 
                ann_file,
                data_path,
                occ_prefix='gts',
                moving_occ_prefix='moving_occs',
                load_moving_occ=True,
                source_frames=6,
                target_frames=6,
                source='gt',
                occpred_path=None,
                with_bev_layout=False,
                bev_root=None,
                bev_ch_use = [0, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],):
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        self.bev_root = bev_root
        self.with_bev_layout = with_bev_layout
        self.ch_use = bev_ch_use

        self.source = source
        self.use_mask_camera = True if self.source != 'gt' else False
        self.occpred_path = occpred_path


        self.scene_names = list(data.keys())
        self.scene_lens = [len(data[sn]) for sn in self.scene_names]
        for scene_name in self.scene_names:
            data[scene_name] = sorted(data[scene_name], key=lambda x: x['timestamp'])
        self.scene_data = data

        self.data_path = data_path
        self.occ_prefix = occ_prefix
        self.moving_occ_prefix = moving_occ_prefix
        self.load_moving_occ = load_moving_occ

        # process source_frames
        assert isinstance(source_frames, (int, list))
        if isinstance(source_frames, int):
            source_frames = list(range(source_frames))
        self.source_frames = source_frames
        assert min(self.source_frames) >= 0

        # process target_frames
        assert isinstance(target_frames, (int, list))
        if isinstance(target_frames, int):
            target_frames = list(range(max(self.source_frames) + 1, 
                                       max(self.source_frames) + 1 + target_frames))
        self.target_frames = target_frames
        # assert min(self.target_frames) >= 0
        if len(self.target_frames) > 0:
            self.total_frames = max(self.target_frames) + 1
        else:
            self.total_frames = max(self.source_frames) + 1

    def __len__(self):
        return sum([max(_len - self.total_frames + 1, 0) for _len in self.scene_lens])
    
    def __getitem__(self, idx):
        _idx = idx
        for scene_name, _len in zip(self.scene_names, self.scene_lens):
            if _idx < max(_len - self.total_frames + 1, 0):
                break
            _idx -= max(_len - self.total_frames + 1, 0)
            
        source_occs = []
        source_sample_idxes = []
        source_moving_inst = []
        source_moving_sem = []
        source_moving_obj = []
        
        if self.use_mask_camera:
            source_mask_cameras = []

        for i in self.source_frames:
            sample = self.scene_data[scene_name][_idx + i]
            source_sample_idxes.append(_idx + i)

            if self.source=='gt':
                # load occ
                occ_file = osp.join(self.data_path, self.occ_prefix, 
                                    scene_name, sample['token'], 'labels.npz')
                source_occs.append(np.load(occ_file)['semantics'])
            else:
                label_file = os.path.join(self.occpred_path, f'{scene_token}/{token}/pred.npz')
                label = np.load(label_file)
                source_occs.append(label['pred'])
                source_mask_cameras.append(label['mask_camera'])      
            # load moving inst id
            # moving_occ_file = osp.join(self.data_path, self.moving_occ_prefix,
            #                            scene_name, sample['token'], 'labels.npz')
            # inst = np.load(moving_occ_file)['instance']
            # source_moving_inst.append(inst)
            # load moving obj
            infos = dict()
            infos['gt_boxes'] = sample['moving_instance']['gt_boxes']
            infos['gt_names'] = sample['moving_instance']['gt_names']
            infos['gt_ids'] = sample['moving_instance']['gt_ids']
            infos['gt_index'] = sample['moving_instance']['gt_index']
            infos['gt_labels'] = np.array(
                [self.MOVING_CLASSES.index(n) for n in infos['gt_names']])
            source_moving_obj.append(infos)
            # load moving inst sem
            # sem = np.full(inst.shape, -1)
            # for idx, label in zip(infos['gt_index'], infos['gt_labels']):
            #     sem[inst==idx] = label
            # source_moving_sem.append(sem)

        source_occs = np.stack(source_occs).astype(np.int64)
        source_occs = source_occs.transpose(0, 3, 2, 1) # (T, W, H, Z) to (T, Z, H, W)
        # source_moving_inst = np.stack(source_moving_inst).astype(np.int64)
        # source_moving_inst = source_moving_inst.transpose(0, 3, 2, 1)
        # source_moving_sem = np.stack(source_moving_sem).astype(np.int64)
        # source_moving_sem = source_moving_sem.transpose(0, 3, 2, 1)
        source_metas = self.get_meta_infos(
            self.scene_data[scene_name], [_idx + i for i in self.source_frames])

        target_occs = []
        target_moving_inst = []
        target_moving_sem = []
        target_moving_obj = []
        target_sample_idxes = []

        if self.use_mask_camera:
            target_mask_cameras = []

        if len(self.target_frames) > 0:
            for i in self.target_frames:
                sample = self.scene_data[scene_name][_idx + i]
                target_sample_idxes.append(_idx + i)

                # load occ
                occ_file = osp.join(self.data_path, self.occ_prefix, 
                                    scene_name, sample['token'], 'labels.npz')
                target_occs.append(np.load(occ_file)['semantics'])

                if self.use_mask_camera:
                    label_file = os.path.join(self.occpred_path, f'{scene_token}/{token}/pred.npz')
                    label = np.load(label_file)
                    target_mask_cameras.append(label['mask_camera']) 
                # load moving inst id
                # moving_occ_file = osp.join(self.data_path, self.moving_occ_prefix,
                #                         scene_name, sample['token'], 'labels.npz')
                # inst = np.load(moving_occ_file)['instance']
                # target_moving_inst.append(inst)
                # load moving obj
                infos = dict()
                infos['gt_boxes'] = sample['moving_instance']['gt_boxes']
                infos['gt_names'] = sample['moving_instance']['gt_names']
                infos['gt_ids'] = sample['moving_instance']['gt_ids']
                infos['gt_index'] = sample['moving_instance']['gt_index']
                infos['gt_labels'] = np.array(
                    [self.MOVING_CLASSES.index(n) for n in infos['gt_names']])
                target_moving_obj.append(infos)
                # load moving inst sem
                # sem = np.full(inst.shape, -1)
                # for idx, label in zip(infos['gt_index'], infos['gt_labels']):
                #     sem[inst==idx] = label
                # target_moving_sem.append(sem)

            target_occs = np.stack(target_occs).astype(np.int64)
            target_occs = target_occs.transpose(0, 3, 2, 1)


            # target_moving_inst = np.stack(target_moving_inst).astype(np.int64)
            # target_moving_inst = target_moving_inst.transpose(0, 3, 2, 1)
            # target_moving_sem = np.stack(target_moving_sem).astype(np.int64)
            # target_moving_sem = target_moving_sem.transpose(0, 3, 2, 1)
            target_metas = self.get_meta_infos(
                self.scene_data[scene_name], [_idx + i for i in self.target_frames])
        
        if self.use_mask_camera:
            source_mask_cameras = np.stack(source_mask_cameras).astype(np.int64)
            source_mask_cameras = source_mask_cameras.transpose(0, 3, 2, 1)
            target_mask_cameras = np.stack(target_mask_cameras).astype(np.int64)
            target_mask_cameras = target_mask_cameras.transpose(0, 3, 2, 1)

        # produce traject input/output
        cur_pose = source_metas['ego2global'][-1]
        cur_pose_inv = np.linalg.inv(cur_pose)
        source_traj = [np.dot(cur_pose_inv, pose)[:3, :] for pose in source_metas['ego2global']]
        source_traj = np.array(source_traj).reshape(-1, 12).astype(np.float32)

        if len(self.target_frames) > 0:
            target_traj = [np.dot(cur_pose_inv, pose)[:3, :] for pose in target_metas['ego2global']]
            target_traj = np.array(target_traj).reshape(-1, 12).astype(np.float32)
        
        result_dict = dict(
            source_occs=source_occs,
            source_metas=source_metas,
            # source_moving_inst=source_moving_inst,
            # source_moving_sem=source_moving_sem,
            # source_moving_obj=source_moving_obj,
            source_traj=source_traj,
            scene_name=scene_name,
            source_sample_idxes=source_sample_idxes,
            target_sample_idxes=target_sample_idxes,
            # target_occs=target_occs,
            # target_metas=target_metas,
            # target_moving_inst=target_moving_inst,
            # target_moving_sem=target_moving_sem,
            # target_moving_obj=target_moving_obj,
            # target_traj=target_traj,
            metas = dict(
                classes=self.CLASSES,
                moving_classes=self.MOVING_CLASSES,
                scene_range=self.SCENE_RANGE,
                voxel_size=self.VOXEL_SIZE,
                idx=idx
            )
        )

        if len(self.target_frames) > 0:
            result_dict.update(
                target_occs=target_occs,
                target_metas=target_metas,
                target_moving_inst=target_moving_inst,
                target_moving_sem=target_moving_sem,
                target_moving_obj=target_moving_obj,
                target_traj=target_traj,
            )
        
        if self.use_mask_camera:
            result_dict.update(  
                source_mask_cameras=source_mask_cameras,
                target_mask_cameras=target_mask_cameras,
            )


       
        result_dict.update(self.get_meta_data(scene_name, _idx))


        return result_dict
    
    def get_meta_infos(self, scene_data, indices):
        metas = dict()
        # calculate the position matrix
        metas['ego2global'] = []
        for i in indices:
            data = scene_data[i]
            ego2global = np.eye(4)
            ego2global_t, ego2global_r = data['ego2global_t'], data['ego2global_r']
            ego2global[:3, :3] = Quaternion(ego2global_r).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_t)
            metas['ego2global'].append(ego2global)

        metas['token'] = [scene_data[i]['token'] for i in indices]
        metas['timestamp'] = [scene_data[i]['timestamp'] for i in indices]
        return metas

    def get_meta_data(self, scene_name, idx, return_len=None):
        # gt_modes = []
        # xys = []
        e2g_t=[]
        e2g_r=[]

        for i in self.source_frames:
            # xys.append(self.scene_data[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2 #[array([-0.0050938,  3.8259335], dtype=float32)]
            e2g_t.append(self.scene_data[scene_name][idx+i]['ego2global_t'])
            e2g_r.append(self.scene_data[scene_name][idx+i]['ego2global_r'])
            # gt_modes.append(self.scene_data[scene_name][idx+i]['pose_mode'])# [0,0,1] #maybe type selection bewteen (angle,speed,trajectory) #may 直行左右转

        for i in self.target_frames:
            # xys.append(self.scene_data[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2 #[array([-0.0050938,  3.8259335], dtype=float32)]
            e2g_t.append(self.scene_data[scene_name][idx+i]['ego2global_t'])
            e2g_r.append(self.scene_data[scene_name][idx+i]['ego2global_r'])
            # gt_modes.append(self.scene_data[scene_name][idx+i]['pose_mode'])# [0,0,1] #maybe type selection bewteen (angle,speed,trajectory) #may 直行左右转

        # xys = np.asarray(xys)
        # gt_modes = np.asarray(gt_modes)
        e2g_t=np.array(e2g_t)
        e2g_r=np.array(e2g_r)
        # use max mode as the whole traj mode 0: right 1: left 2:straight
        # traj_mode=np.argmax(gt_modes.sum(0)).item()

        #get traj (not velocity)  relative to first frame
        e2g_rel0_t=e2g_t.copy()
        e2g_rel0_r=e2g_r.copy()
        for i in self.source_frames + self.target_frames:
            r0=Quaternion(e2g_r[0]).rotation_matrix
            ri=Quaternion(e2g_r[i]).rotation_matrix
            e2g_rel0_t[i]=np.linalg.inv(r0)@(e2g_t[i]-e2g_t[0])
            e2g_rel0_r[i]=Quaternion(matrix=np.linalg.inv(r0)@ri).elements
        poses=[]
        for tt,rr in zip(e2g_t,e2g_r):
            pose=np.eye(4)
            pose[:3,3]=tt
            pose[:3,:3]=Quaternion(rr).rotation_matrix
            poses.append(pose)
        poses=np.stack(poses,axis=0)

        meta_data2=get_meta_data(poses)

        rel_poses_yaws=meta_data2['rel_poses_yaws']
        self.new_rel_pose=True
        if self.new_rel_pose:
            xys=meta_data2['rel_poses']
        
        return {'rel_poses': xys, 'e2g_t':e2g_t,'e2g_r':e2g_r,
                'e2g_rel0_t':e2g_rel0_t,'e2g_rel0_r':e2g_rel0_r,
                'rel_poses_yaws':rel_poses_yaws,
        }

    def get_class_counts(self):
        class_counts = []
        for idx in range(len(self)):
            for scene_name, _len in zip(self.scene_names, self.scene_lens):
                if idx < max(_len - self.total_frames + 1, 0):
                    break
                idx -= max(_len - self.total_frames + 1, 0)
            
            counts = []
            if len(self.target_frames) > 0:
                for i in self.target_frames:
                    counts.append(self.scene_data[scene_name][idx + i]['class_counts'])
            else:
                for i in self.source_frames:
                    counts.append(self.scene_data[scene_name][idx + i]['class_counts'])
            class_counts.append(np.stack(counts).sum(0))
        return class_counts
    
    @staticmethod
    def collate_fn(batch):
        collated_dict = dict()
        for key in batch[0].keys():
            if key in ['source_occs', 'source_moving_inst', 'source_moving_sem',
                       'source_traj', 'target_occs', 'target_moving_inst',
                       'target_moving_sem', 'target_traj']:
                collated_dict[key] = torch.from_numpy(
                    np.stack([item[key] for item in batch], axis=0))
            else:
                collated_dict[key] = [item[key] for item in batch]
        return collated_dict


def cal_moving_instance(args, occ, infos):
    scene_range = np.array(NuscOcc3DDataset.SCENE_RANGE)
    voxel_size = np.array(NuscOcc3DDataset.VOXEL_SIZE)
    W = int((scene_range[3] - scene_range[0]) / voxel_size[0])
    H = int((scene_range[4] - scene_range[1]) / voxel_size[1])
    Z = int((scene_range[5] - scene_range[2]) / voxel_size[2])
    moving_occ_prefix = osp.join(
        args.data_path, args.moving_occ_prefix, infos['scene_name'], infos['token'])
    os.makedirs(moving_occ_prefix, exist_ok=True)

    boxes = infos['instance']['gt_boxes']
    names = infos['instance']['gt_names']
    ids = infos['instance']['gt_ids']
    moving_mask = np.array([n in NuscOcc3DDataset.MOVING_CLASSES for n in names])
    if len(boxes) == 0 or not moving_mask.any():
        infos['moving_instance'] = dict(
            gt_boxes=np.zeros((0, 9), dtype=np.float32),
            gt_names=np.array([], dtype=np.dtype('<U32')),
            gt_ids=np.array([], dtype=np.dtype('<U32')),
            gt_index=np.array([], dtype=np.int64))
        np.savez_compressed(
            osp.join(moving_occ_prefix, 'labels.npz'),
            instance=np.full((W, H, Z), -1, dtype=np.int16))
        return None

    boxes, names, ids = boxes[moving_mask], names[moving_mask], ids[moving_mask]
    order = np.argsort(boxes[:, 3] * boxes[:, 4])[::-1]
    boxes, names, ids = boxes[order], names[order], ids[order]

    xx = np.arange(0, W)[: , None, None].repeat(H, axis=1).repeat(Z, axis=2)
    yy = np.arange(0, H)[None, :, None].repeat(W, axis=0).repeat(Z, axis=2)
    zz = np.arange(0, Z)[None, None, :].repeat(W, axis=0).repeat(H, axis=1)
    occ_index = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    occ_points = (occ_index + 0.5) * voxel_size + scene_range[:3]

    index = 0
    eps = 1
    mv_occ = np.full((W, H, Z), -1, dtype=np.int16)
    mv_boxes, mv_names, mv_ids, mv_index = [], [], [], []
    for box, name, gt_id in zip(boxes, names, ids):
        # filter outside points according to occ category
        label = NuscOcc3DDataset.CLASSES.index(name)
        mask = (occ==label).reshape(-1)
        _occ_points, _occ_index = occ_points[mask], occ_index[mask]

        # filter outside points according to the box.
        L = max(box[3], box[4]) / 2
        x, y, z = _occ_points[:, 0], _occ_points[:, 1], _occ_points[:, 2]
        mask = (x > box[0] - L - eps) & (x < box[0] + L + eps) & \
               (y > box[1] - L - eps) & (y < box[1] + L + eps) & \
               (z > box[2] - box[5] / 2 - eps) & (z < box[2] + box[5] / 2 + eps)
        _occ_points, _occ_index = _occ_points[mask], _occ_index[mask]

        # judge if occ is inside the box
        _box = box.copy()[:7]
        _box[3:6] += voxel_size # expand box to include the points on the boundary
        _box[2] -= _box[5] / 2
        _box = torch.from_numpy(_box[None, None, ...])
        _occ_points = torch.from_numpy(_occ_points[None, ...])
        mask = mmcv.ops.points_in_boxes_cpu(_occ_points, _box)
        mask = mask.reshape(-1).numpy() != 0
        if not mask.any():
            continue

        _occ_index = _occ_index[mask]
        mv_occ[_occ_index[:, 0], _occ_index[:, 1], _occ_index[:, 2]] = index
        mv_boxes.append(box.tolist())
        mv_names.append(name)
        mv_ids.append(gt_id)
        mv_index.append(index)
        index += 1
    
    np.savez_compressed(osp.join(moving_occ_prefix, 'labels.npz'), instance=mv_occ)
    mv_boxes = np.array(mv_boxes, dtype=np.float32) if mv_boxes else \
        np.zeros((0, 9), dtype=np.float32)
    mv_names = np.array(mv_names, dtype=np.dtype('<U32')) if mv_names else \
        np.array([], dtype=np.dtype('<U32'))
    mv_ids = np.array(mv_ids, dtype=np.dtype('<U32')) if mv_ids else \
        np.array([], dtype=np.dtype('<U32'))
    mv_index = np.array(mv_index, dtype=np.int64) if mv_index else \
        np.array([], dtype=np.int64)
    infos['moving_instance'] = dict(
        gt_boxes=mv_boxes,
        gt_names=mv_names,
        gt_ids=mv_ids,
        gt_index=mv_index
    )

   


def get_meta_data(poses):
    rel_pose = np.linalg.inv(poses[:-1]) @ poses[1:]
    rel_pose=  np.concatenate([rel_pose,rel_pose[-1:]], axis=0)
    xyzs = rel_pose[:, :3, 3]

    xys = xyzs[:, :2]
    e2g_t = poses[:, :3, 3]
    # rot 2 quat
    e2g_r = np.array([Quaternion(matrix=pose[:3, :3],atol=1e-7).elements for pose in poses])
    rel_yaws = Rotation.from_matrix(rel_pose[:,:3,:3]).as_euler('zyx', degrees=False)[:,0]

    #get traj (not velocity)  relative to first frame
    e2g_rel0_t = e2g_t.copy()
    # Convert rotations to rotation matrices
    e2g_r_w_last = e2g_r.copy()
    e2g_r_w_last[:, [0, 1, 2 ,3]] = e2g_r_w_last[:, [1, 2,3, 0]] 
    r0 = Rotation.from_quat(e2g_r_w_last[0]).as_matrix()  # First rotation matrix
    rotations = Rotation.from_quat(e2g_r_w_last).as_matrix()  # All rotation matrices
    e2g_rel0_t = np.linalg.inv(r0) @ ( e2g_t - e2g_t[0]).T
    e2g_rel0_t = e2g_rel0_t.T

    rr=np.array([
        [0,-1],
        [1,0],]
    )
    xys=xys@rr.T
    rel_poses_yaws=np.concatenate([xys,rel_yaws[:,None]],axis=1)
    
    return {
        'rel_poses': xys,
        'rel_poses_xyz': xyzs,
        'e2g_t': e2g_t,
        'e2g_r': e2g_r,
        'rel_poses_yaws':rel_poses_yaws,
        'e2g_rel0_t':e2g_rel0_t
    }

def data_preprocess(args):
    import pickle
    from tqdm import tqdm
    from collections import defaultdict
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits

    assert args.version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    nusc = NuScenes(version=args.version, dataroot=args.data_path, verbose=True)
    if args.version == 'v1.0-trainval':
        tasks = [
            ('train', splits.train, defaultdict(list)),
            ('val', splits.val, defaultdict(list))
        ]
    elif args.version == 'v1.0-test':
        tasks = [
            ('test', splits.test, defaultdict(list))
        ]
    else:
        tasks = [
            ('mini-train', splits.mini_train, defaultdict(list)),
            ('mini-val', splits.mini_val, defaultdict(list))
        ]
    
    for sample in tqdm(nusc.sample):
        scene_name = nusc.get('scene', sample['scene_token'])['name']
        for task_name, task_split, collector in tasks:
            if scene_name in task_split:
                break
        else:
            continue

        from .misc import get_nusc_lidar_infos, get_nusc_box_infos
        occ = np.load(osp.join(args.data_path, args.occ_prefix,
                               scene_name, sample['token'], 'labels.npz'))['semantics']
        counts = np.bincount(occ.reshape(-1), minlength=len(NuscOcc3DDataset.CLASSES))
        infos = dict(token=sample['token'], timestamp=sample['timestamp'], class_counts=counts)
        get_nusc_lidar_infos(nusc, sample, infos)
        get_nusc_box_infos(nusc, sample, infos)
        cal_moving_instance(args, occ, infos)
        collector[scene_name].append(infos)

    for task_name, _, collector in tasks:
        data = dict()
        for key, value in collector.items():
            data[key] = sorted(value, key=lambda x: x['timestamp'])
        
        save_file = osp.join(args.data_path, f'nuscenes_{task_name}_occ3d_infos.pkl')
        print(f'Saving data in `nuscenes_{task_name}_occ3d_infos.pkl`')
        with open(save_file, 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Preprocess nuscenes occ3d dataset.')
    parser.add_argument('--data-path', type=str, default='data/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--occ-prefix', type=str, default='gts')
    parser.add_argument('--moving-occ-prefix', type=str, default='moving_occs')
    data_preprocess(parser.parse_args())