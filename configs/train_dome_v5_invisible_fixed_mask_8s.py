

work_dir = './work_dir/dome_v2'

start_frame = 0
mid_frame = 4
end_frame = 20
eval_length = end_frame-mid_frame

return_len_train = 20
return_len_ = 20
grad_max_norm = 1
print_freq = 10
max_epochs = 200
warmup_iters = 50
ema = True

load_from = ''
vae_load_from = 'ckpts/occvae_latest.pth'
port = 25098
revise_ckpt = 3
eval_every_epochs = 1
save_every_epochs = 5

multisteplr = True
multisteplr_config = dict(
    decay_rate=1,
    decay_t=[
        0,
    ],
    t_in_epochs=False,
    warmup_lr_init=1e-06,
    warmup_t=0)
optimizer = dict(optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.0001))

schedule = dict(
    beta_end=0.02,
    beta_schedule='linear',
    beta_start=0.0001,
    variance_type='learned_range')

sample = dict(
    enable_temporal_attentions=True,
    enable_vae_temporal_decoder=True,
    guidance_scale=7.5,
    n_conds=4,
    num_sampling_steps=20,
    run_time=0,
    sample_method='ddpm',
    seed=None)
p_use_pose_condition = 0.9

replace_cond_frames = True
cond_frames_choices = [
    [],
    [0],
    [0,1],
    [0,1,2],
    [0,1,2,3],
]
data_path = 'data/nuscenes/'

train_dataset_config = dict(
    type='nuScenesSceneDatasetLidar',
    data_path = data_path,
    return_len = return_len_train, 
    offset = 0,
    times=50,
    imageset = 'data/nuscenes_infos_train_temporal_v3_scene.pkl',
)

val_dataset_config = dict(
    data_path='data/nuscenes/',
    imageset='data/nuscenes_infos_val_temporal_v3_scene.pkl',
    new_rel_pose=True,
    offset=0,
    return_len=return_len_,
    test_mode=True,
    times=1,
    type='nuScenesSceneDatasetLidar')
train_wrapper_config = dict(phase='train', type='tpvformer_dataset_nuscenes')
val_wrapper_config = dict(phase='val', type='tpvformer_dataset_nuscenes')
train_loader = dict(batch_size=32, num_workers=8, shuffle=True)
val_loader = dict(batch_size=1, num_workers=8, shuffle=False)
loss = dict(
    loss_cfgs=[
        dict(
            input_dict=dict(ce_inputs='ce_inputs', ce_labels='ce_labels'),
            type='CeLoss',
            weight=1.0),
    ],
    type='MultiLoss')
loss_input_convertion = dict()

_dim_ = 16
base_channel = 64
expansion = 8
n_e_ = 512
num_heads=12
hidden_size=768

model = dict(
    delta_input=False,
    world_model=dict(
        attention_mode='xformers',
        class_dropout_prob=0.1,
        depth=28,
        extras=1,
        hidden_size=hidden_size,
        in_channels=64,
        input_size=25,
        learn_sigma=True,
        mlp_ratio=4.0,
        num_classes=1000,
        num_frames=return_len_train,
        num_heads=num_heads,
        patch_size=1,
        pose_encoder=dict(
            do_proj=True,
            in_channels=2,
            num_fut_ts=1,
            num_layers=2,
            num_modes=3,
            out_channels=hidden_size,
            type='PoseEncoder_fourier',
            zero_init=False),
        type='DomeV5'),
    sampling_method='SAMPLE',
    topk=10,
    vae=dict(
        encoder_cfg=dict(
            attn_resolutions=(50, ),
            ch=base_channel,
            ch_mult=(
                1,
                2,
                4,
                8,
            ),
            double_z=False,
            dropout=0.0,
            in_channels=128,
            num_res_blocks=2,
            out_ch=base_channel,
            resamp_with_conv=True,
            resolution=200,
            type='Encoder2D',
            z_channels=base_channel*2),
        decoder_cfg=dict(
            attn_resolutions=(50, ),
            ch=base_channel,
            ch_mult=(
                1,
                2,
                4,
                8,
            ),
            dropout=0.0,
            give_pre_end=False,
            in_channels=_dim_ * expansion,
            num_res_blocks=2,
            out_ch=_dim_ * expansion,
            resamp_with_conv=True,
            resolution=200,
            type='Decoder3D',
            z_channels=base_channel),
        expansion=expansion,
        num_classes=18,
        scaling_factor=0.18215,
        type='VAERes3D'))
shapes = [[200,200],[100,100],[50,50],[25,25]]

unique_label = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
label_mapping = './configs/label_mapping/nuscenes-occ.yaml'

# load_from = "ckpts/dome_latest.pth"

find_unused_parameters=True