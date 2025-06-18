_base_ = '../default_settings.py'

class_weights = [
    6,  # others
    4,  # barrier
    34, # bicycle
    3,  # bus
    1,  # car
    7,  # construction_vehicle
    30, # motorcycle
    4,  # pedestrian
    26, # traffic_cone
    3,  # trailer
    1,  # truck
    1,  # driveable_surface
    3,  # other_flat
    1,  # sidewalk
    1,  # terrain
    1,  # manmade
    1,  # vegetation
    1   # free
]

model=dict(
    type='BaseTrajPredictor',
    in_ts=6,
    in_channels=16,
    out_ts=6,
    out_channels=16,
)

train_dataloader=dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='NuscOcc3DDataset',
        ann_file='data/nuscenes_train_occ3d_infos.pkl',
        data_path='data/nuscenes',
        occ_prefix='gts',
        load_moving_occ=False,
        source_frames=6,
        target_frames=6),
    wrappers=dict(
        type='BalanceClassWrapper',
        balance_mode='voxel',
        ratio=0.01),
)

val_dataloader=dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='NuscOcc3DDataset',
        ann_file='data/nuscenes_val_occ3d_infos.pkl',
        data_path='data/nuscenes',
        occ_prefix='gts',
        source_frames=6,
        target_frames=6)
)

test_dataloader=dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='NuscOcc3DDataset',
        ann_file='data/nuscenes_val_occ3d_infos.pkl',
        data_path='data/nuscenes',
        occ_prefix='gts',
        source_frames=6,
        target_frames=6)
)

evaluators=[
    dict(type='ADE', 
         timestamps=['0.5s', '1s', '1.5s', '2s', '2.5s', '3s']),
]

lr = 1e-3
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))

param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=10*lr,
        begin=0,
        end=12,
        pct_start=0.4,
        div_factor=10,
        convert_to_iter_based=True
    )
]

max_epoch = 12