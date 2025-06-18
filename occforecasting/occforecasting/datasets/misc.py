import numpy as np
from pyquaternion import Quaternion

nusc_cls_mapper = {
    'animal': 'others',
    'human.pedestrian.personal_mobility':   'others',
    'human.pedestrian.stroller':            'others',
    'human.pedestrian.wheelchair':          'others',
    'movable_object.debris':                'others',
    'movable_object.pushable_pullable':     'others',
    'static_object.bicycle_rack':           'others',
    'vehicle.emergency.ambulance':          'others',
    'vehicle.emergency.police':             'others',
    'noise':                                'others',
    'static.other':                         'others',
    'vehicle.ego':                          'others',
    'movable_object.barrier':               'barrier',
    'vehicle.bicycle':                      'bicycle',
    'vehicle.bus.bendy':                    'bus',
    'vehicle.bus.rigid':                    'bus',
    'vehicle.car':                          'car',
    'vehicle.construction':                 'construction_vehicle',
    'vehicle.motorcycle':                   'motorcycle',
    'human.pedestrian.adult':               'pedestrian',
    'human.pedestrian.child':               'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer':      'pedestrian',
    'movable_object.trafficcone':           'traffic_cone',
    'vehicle.trailer':                      'trailer',
    'vehicle.truck':                        'truck',
    'flat.driveable_surface':               'driveable_surface',
    'flat.other':                           'other_flat',
    'flat.sidewalk':                        'sidewalk',
    'flat.terrain':                         'terrain',
    'static.manmade':                       'manmade',
    'static.vegetation':                    'vegetation',
}


def get_nusc_lidar_infos(nusc, sample, infos):
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_pose = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    scene = nusc.get('scene', sample['scene_token'])
    infos.update(dict(
        scene_name=scene['name'],
        lidar_data_token=lidar_data['token'],
        lidar_data_path=lidar_data['filename'],
        lidar2ego_r = lidar_pose['rotation'],
        lidar2ego_t = lidar_pose['translation'],
        ego2global_r = ego_pose['rotation'],
        ego2global_t = ego_pose['translation']))


def get_nusc_box_infos(nusc, sample, infos):
    lidar_infos = infos if 'lidar2ego_r' in infos else \
        get_nusc_lidar_infos(nusc, sample, {})
    boxes = nusc.get_boxes(sample['data']['LIDAR_TOP'])
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(lidar_infos['ego2global_t']))
        box.rotate(Quaternion(lidar_infos['ego2global_r']).inverse)

    annos = [nusc.get('sample_annotation', box.token) for box in boxes]
    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)[:, [1, 0, 2]]
    rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
    velos = np.array([b.velocity for b in boxes]).reshape(-1, 3)[:, :2]
    gt_boxes = np.concatenate([locs, dims, rots, velos], axis=1)
    gt_boxes = gt_boxes.astype(np.float64)
    gt_names = np.array([nusc_cls_mapper[b.name] for b in boxes], dtype=np.dtype('<U32'))
    gt_ids = np.array([anno['instance_token'] for anno in annos], dtype=np.dtype('<U32'))
    gt_npts = np.array([anno['num_lidar_pts'] for anno in annos], dtype=np.int64)

    infos['instance'] = dict(
        gt_boxes=gt_boxes,
        gt_names=gt_names,
        gt_ids=gt_ids,
        gt_npts=gt_npts
    )
