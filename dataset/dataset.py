import os, numpy as np, pickle
from pyquaternion import Quaternion
from copy import deepcopy
try:
    from . import OPENOCC_DATASET
except:
    from mmengine.registry import Registry
    OPENOCC_DATASET = Registry('openocc_dataset')
import torch
import torch.nn.functional as F
import numpy as np
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, Box3DMode
from pathlib import Path
from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation
import re
import mmengine
import copy
def nBEV1(data_b, ch_use):  # 18,200,200 -> 1,200,200
    data_b = data_b[ch_use]
    mask = data_b > 0.01
    cumulative_mask = np.cumsum(mask, axis=0)
    max_index_map = np.argmax(cumulative_mask, axis=0)
    max_index_map = max_index_map / (len(ch_use) - 1)
    all_zero_mask = np.all(mask == 0, axis=0)
    max_index_map[all_zero_mask] = -1
    data_b = np.array([max_index_map])
    return data_b

import numpy as np
from math import atan2, cos, sin, sqrt


def calculate_displacement(prev_position, current_position):
    prev_x, prev_y = prev_position
    current_x, current_y = current_position
    return np.sqrt((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2)

def create_rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def calculate_rotation_matrix(prev_R, delta_theta):
    delta_R = create_rotation_matrix(delta_theta)
    new_R = np.dot(prev_R, delta_R)
    return new_R




def compute_relative_poses_with_differential_yaw(pose_matrixs, poses_2d, traj_mode):
    rotation_matrices = [pose_matrixs[0], pose_matrixs[1], pose_matrixs[2], pose_matrixs[3]]

    prev_position = (pose_matrixs[3][0, 3], pose_matrixs[3][1, 3])
    
    prev_R = pose_matrixs[3]
    for fut_frame, position in enumerate(poses_2d[1:]):

        delta_theta = position[2]
        
        new_R = calculate_rotation_matrix(prev_R, delta_theta)
        new_R[0, 3] = position[0]
        new_R[1, 3] = position[1]
        rotation_matrices.append(new_R)
        prev_position = position

    return np.stack(rotation_matrices)


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            times=5,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            new_rel_pose=False,
            test_index_offset=0,
            source='gt',
            occpred_path=None,
            with_bev_layout=False,
            bev_root=None,
            bev_ch_use = [0, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            planning_results = None,
            re_origin = False,
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        
        self.bev_root = bev_root
        self.with_bev_layout = with_bev_layout
        self.ch_use = bev_ch_use
        if planning_results:
            self.planning_results = mmengine.load(planning_results)['trajs']
        else:
            self.planning_results = None
        self.source = source
        self.use_mask_camera = True if self.source != 'gt' else False
        self.occpred_path = occpred_path
        self.nusc_infos = data['infos']  #TODO
        # self.nusc_infos = dict(list(data['infos'].items())[::50])# data['infos']  #debug #TODO
        # self.nusc_infos = dict(list(data['infos'].items())[::10])# data['infos']  #debug
        # self.nusc_infos = dict(list(data['infos'].items())[:1])# data['infos']  #debug
        self.scene_names = list(self.nusc_infos.keys())

        # self.scene_names = self.scene_names[:10]
        # self.nusc_infos = {key: self.nusc_infos[key] for key in self.scene_names}

        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.test_mode = test_mode
        # assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        # assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.new_rel_pose=new_rel_pose
        self.test_index_offset=test_index_offset

        self.re_origin = re_origin
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)*self.times

    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        return_len_=min(self.return_len,scene_len- self.offset-self.test_index_offset)
        if not self.test_mode:
            # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
            idx = np.random.randint(0, scene_len - return_len_ - self.offset + 1)
            # print('@'*50,index,idx,scene_len - self.return_len - self.offset + 1,len(self.scene_names),self.scene_names[0:5],self.scene_names[-5:])
        else:
            # idx=0
            idx=self.test_index_offset
            # print('@'*10,idx)

        scene_token=self.nusc_infos[scene_name][4]['scene_token']
        occs = []
        mask_cameras = []
        for i in range(return_len_ + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            if self.source=='gt':
                label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
                label = np.load(label_file)
                occ = label['semantics']            
                mask_cameras.append(label['mask_camera']) 

            elif self.source=='fusion':
                label_file = os.path.join(self.occpred_path, f'{scene_token}/{token}/pred.npz')
                label = np.load(label_file)
                occ = label['pred']   
                mask_cameras.append(label['mask_camera']) 


            occs.append(occ)
        input_occs = np.stack(occs, dtype=np.int64)
        occs = []
        
        for i in range(return_len_ + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs, dtype=np.int64)
        metas = {}
        metas.update(scene_token=self.nusc_infos[scene_name][4]['token'])
        metas.update(self.get_meta_data(scene_name, idx,return_len=return_len_))
        if self.use_mask_camera:
            metas.update(mask_camera=torch.from_numpy(np.stack(mask_cameras)))
        
        if self.with_bev_layout:
            Bevs = []
            Bevs_ori = []

            for i in range(return_len_ + self.offset):
                token = self.nusc_infos[scene_name][idx + i]['token']
                Bev_file = os.path.join(self.bev_root, f"{token}.npz")
                Bev = np.load(Bev_file)
                Bev = Bev["arr_0"]
                Bevs_ori.append(Bev)  # 18 200 200
                if self.ch_use:
                    Bev = nBEV1(Bev, self.ch_use)
                Bevs.append(Bev)

            data_Bev = np.stack(Bevs).astype(np.float32)
            data_Bev = torch.from_numpy(data_Bev)
            data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
            data_Bev = torch.flip(data_Bev, dims=[2])

            data_Bev_ori = np.stack(Bevs_ori)
            data_Bev_ori = torch.from_numpy(data_Bev_ori)
            metas.update(data_bev=data_Bev)
            metas.update(data_bev_ori=data_Bev_ori)
        
        if self.planning_results is not None and self.re_origin:
            output_occs, invisible_mask = self.transform_coor_system(output_occs, metas['poses_gt'], metas['poses'])

        
        return input_occs[:self.return_len], output_occs[self.offset:], metas

    def get_meta_data(self, scene_name, idx,return_len=None):
        gt_modes = []
        xys = []
        e2g_t=[]
        e2g_r=[]
        return_len=self.return_len if return_len is None else return_len

        for i in range(return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2 #[array([-0.0050938,  3.8259335], dtype=float32)]
            e2g_t.append(self.nusc_infos[scene_name][idx+i]['ego2global_translation'])
            e2g_r.append(self.nusc_infos[scene_name][idx+i]['ego2global_rotation'])
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])# [0,0,1] #maybe type selection bewteen (angle,speed,trajectory) #may 直行左右转
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        e2g_t=np.array(e2g_t)
        e2g_r=np.array(e2g_r)
        # use max mode as the whole traj mode 0: right 1: left 2:straight
        traj_mode=np.argmax(gt_modes.sum(0)).item()

        #get traj (not velocity)  relative to first frame
        e2g_rel0_t=e2g_t.copy()
        e2g_rel0_r=e2g_r.copy()
        for i in range(return_len + self.offset):
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

        poses = np.stack(poses,axis=0)
        poses_gt = copy.deepcopy(poses)
        
        if self.planning_results is not None:
            
            plan_ts = 6
            his_ts = poses.shape[0] - plan_ts
            filtered_dict = {k: v for k, v in self.planning_results.items() if scene_name in k}
            def extract_number(key):
                return int(key.split('-')[-1]) 

            sorted_keys = sorted(filtered_dict.keys(), key=extract_number)
            poses_2d = filtered_dict[list(sorted_keys)[idx+his_ts-1]]
            
            
            poses = compute_relative_poses_with_differential_yaw(poses, poses_2d, traj_mode)

            e2g_t = poses[:,:3,3]
            # for i in range(len(poses)):
            #     e2g_r[i] = Quaternion(matrix=poses[i,:3,:3])
        e2g_r_matrix = poses[:,:3,:3]

        meta_data2=get_meta_data(poses)

        rel_poses_yaws=meta_data2['rel_poses_yaws']
        if self.new_rel_pose:
            xys=meta_data2['rel_poses']
        
        return {'rel_poses': xys, 'gt_mode': gt_modes, 'e2g_t':e2g_t,'e2g_r':e2g_r, 'e2g_r_matrix':e2g_r_matrix,'traj_mode':traj_mode,
                'e2g_rel0_t':e2g_rel0_t,'e2g_rel0_r':e2g_rel0_r,
                'rel_poses_yaws':rel_poses_yaws, 'poses_gt':poses_gt, 'poses':poses,
        }


    def transform_coor_system(self, occs, src_matrix, dst_matrix):
        occs = torch.from_numpy(occs).permute(0,3,2,1)
        scene_range = (-40, -40, -1, 40, 40, 5.4)
        scene_range = occs.new_tensor(scene_range)
        origin = (scene_range[:3] + scene_range[3:]) / 2
        scene_size = scene_range[3:] - scene_range[:3]

        Z, H, W = occs.shape[1:]
        x = torch.arange(0, W, dtype=occs.dtype, device=occs.device)
        x = (x + 0.5) / W * scene_size[0] + scene_range[0]
        y = torch.arange(0, H, dtype=occs.dtype, device=occs.device)
        y = (y + 0.5) / H * scene_size[1] + scene_range[1]
        z = torch.arange(0, Z, dtype=occs.dtype, device=occs.device)
        z = (z + 0.5) / Z * scene_size[2] + scene_range[2]
        xx = x[None, None, :].expand(Z, H, W)
        yy = y[None, :, None].expand(Z, H, W)
        zz = z[:, None, None].expand(Z, H, W)
        coors = torch.stack([xx, yy, zz], dim=-1)
        
        offsets = []
        for src_mat, dst_mat in zip(src_matrix, dst_matrix):
            src_mat = coors.new_tensor(src_mat)
            dst_mat = coors.new_tensor(dst_mat)

            coors_ = F.pad(coors.reshape(-1, 3), (0, 1), 'constant', 1)
            coors_ = coors_ @ dst_mat.T @ torch.inverse(src_mat).T
            offset = (coors_[:, :3] - origin) / scene_size * 2
            offsets.append(offset.reshape(Z, H, W, 3))
        offsets = torch.stack(offsets)
        occs = occs.unsqueeze(1).float()
        occs = F.grid_sample(occs, offsets, mode='nearest', align_corners=False)
        mask = (offsets.abs() > 1).any(-1)
        occs = occs.squeeze(1).permute(0,3,2,1)
        mask = mask.permute(0,3,2,1)
        occs[mask] = 17
        return occs.long().numpy(), mask.numpy()


    def get_traj_mode(self, scene_name, idx):
        gt_modes = []
        for i in range(self.return_len + self.offset):
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])# [0,0,1] #maybe type selection bewteen (angle,speed,trajectory) #may 直行左右转
        gt_modes = np.asarray(gt_modes)
        # use max mode as the whole traj mode 0: right 1: left 2:straight
        traj_mode=np.argmax(gt_modes.sum(0)).item()
        return traj_mode

    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict
        
@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidarTraverse(nuScenesSceneDatasetLidar):
    def __init__(
        self,
        data_path,
        return_len,
        offset,
        imageset='train',
        nusc=None,
        times=1,
        test_mode=False,
        use_valid_flag=True,
        input_dataset='gts',
        output_dataset='gts',
        **kwargs
    ):
        super().__init__(data_path, return_len, offset, imageset, nusc, times, test_mode, input_dataset, output_dataset,**kwargs)
        self.scene_lens = [l - self.return_len - self.offset for l in self.scene_lens]
        # self.scene_lens=self.scene_lens[:1]        #debug
        self.use_valid_flag = use_valid_flag
        self.CLASSES = [
            'noise', 'animal' ,'human.pedestrian.adult', 'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
            'movable_object.barrier', 'movable_object.debris',
            'movable_object.pushable_pullable', 'movable_object.trafficcone',
            'static_object.bicycle_rack', 'vehicle.bicycle',
            'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
            'vehicle.construction', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.motorcycle',
            'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
            'flat.other', 'flat.sidewalk', 'flat.terrain', 'flat.traffic_marking',
            'static.manmade', 'static.other', 'static.vegetation',
            'vehicle.ego'
        ]
        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
        
    def __len__(self):
        'Denotes the total number of samples'
        return sum(self.scene_lens)
    
    def __getitem__(self, index):
        for i, scene_len in enumerate(self.scene_lens):
            if index < scene_len:
                scene_name = self.scene_names[i]
                idx = index
                break
            else:
                index -= scene_len
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        input_occs = np.stack(occs, dtype=np.int64)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs, dtype=np.int64)
        metas = {}
        metas.update(scene_name=scene_name)
        metas.update(scene_token=self.nusc_infos[scene_name][4]['token'])
        metas.update(self.get_meta_data(scene_name, idx))
        # metas.update(self.get_ego_action_info(scene_name,idx))
        # if self.test_mode:
            # metas.update(self.get_meta_info(scene_name, idx))
        # metas.update(self.get_image_info(scene_name,idx))
        # import pdb; pdb.set_trace()
        return input_occs[:self.return_len], output_occs[self.offset:], metas
    
    def get_ego_action_info(self, scene_name, idx):
        vels=[]
        steers=[]
        accels=[]
        for i in range(self.return_len + self.offset):
            accel = np.linalg.norm(self.nusc_infos[scene_name][idx + i]['can_bus'][7:10]) #todo direction
            vel = np.linalg.norm(self.nusc_infos[scene_name][idx + i]['can_bus'][13: 16])
            steer = self.nusc_infos[scene_name][idx + i]['can_bus'][16]
            accels.append(accel)
            vels.append(vel)
            steers.append(steer)
        accels = np.array(accels)
        vels = np.array(vels)
        steers = np.array(steers)
        return {'vels': vels,'steers': steers, 'accels': accels}

    def get_meta_info(self, scene_name, idx):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        fut_valid_flag = info['valid_flag']
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        '''gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
                print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        '''
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        if self.with_attr:
            gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
            gt_fut_masks = info['gt_agent_fut_masks'][mask]
            gt_fut_goal = info['gt_agent_fut_goal'][mask]
            gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            #gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,)
        
        return anns_results
        
        
        
    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidarResample:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            times=1,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            raw_times=1,
            resample_times=1,
        ):

        self.scene_lens ,self.nusc_infos = self.load_data(data_path,raw_times,resample_times)
        # self.nusc_infos = self.nusc_infos[::10]# data['infos']  #debug #TODO
        # self.scene_lens = self.scene_lens[::10]   # data['infos']  #debug #TODO
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.test_mode = test_mode
    
    def load_data(self, data_path,raw_times=0,resample_times=1):
        cache_path=f'{data_path}/scene_cache.npz'
        if os.path.exists(cache_path):
            data = np.load(cache_path,allow_pickle=True)
            # return data['all_scene_lens'], data['all_occs_path']
            all_scene_lens=data['all_scene_lens'].tolist()
            all_occs_path=data['all_occs_path'].tolist()
            all_scene_lens_raw=data['all_scene_lens_raw'].tolist()
            all_occs_path_raw=data['all_occs_path_raw'].tolist()
        else:
            def process_scene(src_scene,scene_key='scene*'):
                scene_lens = []
                occs_path = []
                for resample_scene in sorted(src_scene.glob(scene_key)):
                    all_traj = list(sorted(resample_scene.glob('traj*')))
                    scene_lens.append(len(all_traj))
                    occs_path_i = [(traj/'labels.npz').as_posix() for traj in all_traj]
                    occs_path.append(occs_path_i)
                return scene_lens, occs_path

            all_src_scenes=sorted(list(Path(data_path).glob('src_scene*')))
            total_scenes = len(all_src_scenes)
            
            results = Parallel(n_jobs=-1)(
                delayed(process_scene)(src_scene,scene_key='scene*') 
                for src_scene in tqdm(all_src_scenes, total=total_scenes, desc="Processing scenes")
            )
            
            all_scene_lens = []
            all_occs_path = []
            for scene_lens, occs_path in results:
                all_scene_lens.extend(scene_lens)
                all_occs_path.extend(occs_path)

            # add raw_scene
            results = Parallel(n_jobs=-1)(
                delayed(process_scene)(src_scene,scene_key='raw_scene') 
                for src_scene in tqdm(all_src_scenes, total=total_scenes, desc="Processing scenes")
            )
            
            all_scene_lens_raw = []
            all_occs_path_raw = []
            for scene_lens, occs_path in results:
                all_scene_lens_raw.extend(scene_lens)
                all_occs_path_raw.extend(occs_path)

            np.savez(cache_path, all_scene_lens=all_scene_lens,all_occs_path=all_occs_path,all_scene_lens_raw=all_scene_lens_raw,all_occs_path_raw=np.array(all_occs_path_raw, dtype="object"))

        all_scene_lens=all_scene_lens*resample_times+all_scene_lens_raw*raw_times
        all_occs_path=all_occs_path*resample_times+all_occs_path_raw*raw_times

        return all_scene_lens, all_occs_path
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)*self.times

    def __getitem__(self, index):
        scene_index = index % len(self.nusc_infos)
        scene_len = self.scene_lens[scene_index]
        if not self.test_mode:
            idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        else:
            idx=0
        occs = []
        poses=[]
        for i in range(self.return_len + self.offset):
            iidx=idx + i
            # if iidx>=scene_len:
            #     iidx=scene_len-1
            #     print(f'warning: {iidx} out of range, scene_len: {scene_len}')
            label_file = self.nusc_infos[scene_index][iidx]
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
            poses.append(label['pose'])
        input_occs = np.stack(occs, dtype=np.int64)
        poses=np.stack(poses, dtype=np.float32)
        output_occs = input_occs.copy()
        metas = {}
        metas.update(get_meta_data(poses))
        metas['src_scenes']=int(re.search(r'src_scene-(\d{4})', label_file).group(1)) #TODO might bug
        return input_occs[:self.return_len], output_occs[self.offset:], metas

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

