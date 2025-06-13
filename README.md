
# COME: Adding Scene-Centric Forecasting Control to Occupancy World Model

# Demo Videos

The comparison of ground-truth, DOME generation with official checkpoint and COME. The task setting is to use 4-frame 3D-Occ sequences as input and predict the next 6-frame (3-s prediction) sequences. 
 
<video src="assets/gt_dome_come_3s_results.mp4" controls="controls" width="500" height="300"></video>



The comparison of ground-truth, DOME generation with reproduced checkpoint and COME. The task setting is to use 4-frame 3D-Occ sequences as input and predict the next 16-frame (8-s prediction) sequences. 
 
<video src="assets/gt_dome_come_8s_results.mp4" controls="controls" width="500" height="300"></video>


The COME generation with BEV layouts. The task setting is to use 2-frame 3D-Occ sequences and 8-frame BEV layouts as input and predict the next 6-frame (3-s) sequences. 
 
<video src="assets/come_with_bev_layout_3s_results.mp4" controls="controls" width="500" height="300"></video>


# Overview
COME = Forecasting Guided Generation

![method](assets/method.png)

# Results

![method](assets/result.png)




## 🚀 Setup
### environment setup
```
conda env create --file environment.yml
pip install einops tabulate 
cd occforecasting 
python setup.py develop
cd ..
```

### data preparation
1. Create soft link from `data/nuscenes` to your_nuscenes_path

2. Prepare the gts semantic occupancy introduced in [Occ3d](https://github.com/Tsinghua-MARS-Lab/Occ3D)

3. Download generated train/val pickle files from OccWorld or DOME.

4. Prepare the train/val pickle files for scene-centric forecasting.

```
python -m occforecasting.datasets.nusc_occ3d_dataset
```

  The dataset should be organized as follows:



```
.
└── data/
    ├── nuscenes            # downloaded from www.nuscenes.org/
    │   ├── lidarseg
    │   ├── maps
    │   ├── samples
    │   ├── sweeps
    │   ├── v1.0-trainval
    │   └── gts             # download from Occ3d
    ├── nuscenes_infos_train_temporal_v3_scene.pkl
    └── nuscenes_infos_val_temporal_v3_scene.pkl
    ├── nuscenes_train_occ3d_infos.pkl
    └── nuscenes_val_occ3d_infos.pkl
```

### Model Zoos


| Task Setting | Inputs | Method | Config |
| --- | --- | --- |  --- |
| Input-4frame-Output-6frame | 3DOcc + GT Traj | Stage1-COME-World Model | [Config](Code_COME/configs/train_dome_v2.py)
| Input-4frame-Output-6frame | 3DOcc + GT Traj| Stage2-COME-Scene-Centric-Forecasting | [Config](Code_COME/occforecasting/configs/unet/unet_aligned_past2s_future_3s.py)
| Input-4frame-Output-6frame | 3DOcc + GT Traj| Stage3-COME-ControlNet | [Config](Code_COME/configs/train_dome_controlnet_mask_invisible_v2.py)
| Input-4frame-Output-6frame | 3DOcc + Pred Traj| Stage3-COME-ControlNet | [Config](Code_COME/configs/inference_configs/inference_dome_controlnet_mask_invisible_v2_3docc_input_pred_traj.py)
| Input-4frame-Output-6frame | BEVDet + Pred Traj| Stage3-COME-ControlNet | [Config](CCode_COME/configs/inference_configs/inference_dome_controlnet_mask_invisible_v2_effocc_input_pred_traj.py)
| Input-4frame-Output-6frame | BEVDet + GT Traj| Stage3-COME-ControlNet | [Config](Code_COME/configs/inference_configs/inference_dome_controlnet_mask_invisible_v2_bevdet_input_gt_traj.py)
| Input-4frame-Output-6frame | EFFOcc + Pred Traj| Stage3-COME-ControlNet | [Config](Code_COME/configs/inference_configs/inference_dome_controlnet_mask_invisible_v2_effocc_input_pred_traj.py)
| Input-4frame-Output-6frame | EFFOcc + GT Traj| Stage3-COME-ControlNet | [Config](Code_COME/configs/inference_configs/inference_dome_controlnet_mask_invisible_v2_effocc_input_gt_traj.py)
| Input-4frame-Output-16frame | 3DOcc + GT Traj | Stage1-COME-World Model | [Config](Code_COME/configs/train_dome_v2_8s.py)
| Input-4frame-Output-16frame | 3DOcc + GT Traj| Stage2-COME-Scene-Centric-Forecasting | [Config](Code_COME/occforecasting/configs/unet/unet_aligned_past2s_future_8s.py)
| Input-4frame-Output-16frame | 3DOcc + GT Traj| Stage3-COME-ControlNet | [Config](Code_COME/configs/train_dome_controlnet_8s.py)
| Input-2frame-Output-6frame | 3DOcc + GT Traj + BEV Layouts | Stage1-COME-World Model | [Config](Code_COME/configs/train_dome_v3_with_bev_layout.py)
| Input-2frame-Output-6frame | 3DOcc + GT Traj + BEV Layouts | Stage2-COME-Scene-Centric-Forecasting | [Config](Code_COME/occforecasting/configs/unet/unet_aligned_past0.5s_future_3s.py)
| Input-2frame-Output-6frame | 3DOcc + GT Traj + BEV Layouts | Stage3-COME-ControlNet | [Config](Code_COME/configs/train_dome_controlnet_bev_layout_masked.py)
| Input-4frame-Output-6frame | 3DOcc + GT Traj | Stage1-COME-Small-World Model | [Config](Code_COME/configs/train_dome_v5_small.py)
| Input-4frame-Output-6frame | 3DOcc + GT Traj| Stage2-COME-Scene-Centric-Forecasting | [Config](Code_COME/occforecasting/configs/unet/unet_aligned_past2s_future_3s.py)
| Input-4frame-Output-6frame | 3DOcc + GT Traj| Stage3-COME-Small-ControlNet | [Config](Code_COME/configs/train_dome_controlnet_small_masked.py)

## 🏃 Run the code
### OCC-VAE
```shell
# train 
python tools/train_vae.py --py-config ./configs/train_occvae.py --work-dir ./work_dir/occ_vae 

# eval
python tools/train_vae.py --py-config ./configs/train_occvae.py --work-dir ./work_dir/occ_vae --resume-from ckpts/occvae_latest.pth

# visualize
python tools/visualize_demo_vae.py \
    --py-config ./configs/train_occvae.py \
    --work-dir ./work_dir/occ_vae \
    --resume-from ckpts/occvae_latest.pth \
    --export_pcd \
    --skip_gt
```

### Scene-Centric Forecasting
```shell
cd occforecasting
# train 
bash train.sh occforecasting/configs/unet/unet_aligned_past2s_future_3s.py

# eval
bash test.sh occforecasting/configs/unet/unet_aligned_past2s_future_3s.py

```

### COME World Model
```shell
# train 
python tools/train_diffusion.sh --py-config ./configs/train_dome_v2.py --work-dir ./work_dir/dome_v2

# eval
python tools/eval_metric.py --py-config ./configs/train_dome_v2.py --work-dir ./work_dir/dome_v2 --resume-from ./work_dir/dome_v2/best_miou.pth --vae-resume-from ckpts/occvae_latest.pth 


# visualize
python tools/visualize_demo.py --py-config ./configs/train_dome_v2.py --work-dir ./work_dir/dome_v2 --resume-from ./work_dir/dome_v2/best_miou.pth --vae-resume-from ckpts/occvae_latest.pth 
```

### COME ControlNet
```shell
# train 
python tools/train_diffusion_control_ddp.py --py-config configs/train_dome_controlnet_mask_invisible_v2.py --work-dir work_dir/train_dome_controlnet_mask_invisible_v2 

# eval
python tools/test_diffusion_control.py --py-config configs/train_dome_controlnet_mask_invisible_v2.py --work-dir work_dir/train_dome_controlnet_mask_invisible_v2 


# visualize
python tools/visualize_demo_control_mask_invisible.py --py-config configs/train_dome_controlnet_mask_invisible_v2.py  --work-dir work_dir/train_dome_controlnet_mask_invisible_v2  --vae-resume-from ckpts/occvae_latest.pth  --skip_gt 
```


### Acknoweldgement
Thanks for the excellent works!

[DOME](https://github.com/gusongen/DOME)

[OccWorld](https://github.com/wzzheng/OccWorld)

[ControlNet](https://github.com/lllyasviel/ControlNet)

[HuyuanDiT](https://github.com/Tencent-Hunyuan/HunyuanDiT)
