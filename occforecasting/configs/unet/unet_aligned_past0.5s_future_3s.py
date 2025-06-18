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

source_frames = 2
target_frames = 6

model=dict(
    type='UNet',
    source_seq_size=(source_frames, 16, 200, 200),
    target_seq_size=(target_frames, 16, 200, 200),
    num_classes=18,
    align_source_coors=True,
    align_target_coors=True,
    recover_target_coors=True,
    sem_encode_type='embedding',
    sem_embedding_dim=32,
    size_divisor=16,
    base_channels=256,
    num_stages=5,
    block_type='BottleNeck',
    enc_channels=(256, 512, 1024, 1024, 1024),
    strides=(1, 1, 1, 1, 1),
    enc_num_convs=(2, 2, 2, 2, 2),
    dec_num_convs=(2, 2, 2, 2),
    downsamples=(True, True, True, True),
    enc_dilations=(1, 1, 1, 1, 1),
    dec_dilations=(1, 1, 1, 1),
    temporal_num_convs=2,
    conv_cfg=None,
    norm_cfg=dict(type='BN'),
    act_cfg=dict(type='ReLU'),
    upsample_cfg=dict(type='InterpConv'),
    losses=dict(
        ce_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=class_weights,
            loss_weight=1.0
        )
    )
)

train_dataloader=dict(
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type='NuscOcc3DDataset',
        ann_file='data/nuscenes_train_occ3d_infos.pkl',
        data_path='data/nuscenes',
        occ_prefix='gts',
        source_frames=source_frames,
        target_frames=target_frames),
    wrappers=dict(
        type='BalanceClassWrapper',
        balance_mode='voxel',
        ratio=0.01),
)

val_dataloader=dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='NuscOcc3DDataset',
        ann_file='data/nuscenes_val_occ3d_infos.pkl',
        data_path='data/nuscenes',
        occ_prefix='gts',
        source_frames=source_frames,
        target_frames=target_frames)
)

test_dataloader=dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='NuscOcc3DDataset',
        ann_file='data/nuscenes_val_occ3d_infos.pkl',
        data_path='data/nuscenes',
        occ_prefix='gts',
        source_frames=source_frames,
        target_frames=target_frames)
)

evaluators=[
    dict(type='MIoU', ignore_label=-1),
    dict(type='SeqMIoU', ignore_label=-1,
         timestamps=['0.5s', '1s', '1.5s', '2s', '2.5s', '3s'])
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


resume = 'auto'
