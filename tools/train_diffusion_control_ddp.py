import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmengine.registry import MODELS
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.load_save_util import revise_ckpt, revise_ckpt_1, load_checkpoint
from utils.ema import update_ema
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
from diffusion import create_diffusion
from einops import rearrange
from diffusion.gaussian_diffusion import ModelMeanType
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from pyquaternion import Quaternion

# build model
import model
from dataset import get_dataloader, get_nuScenes_label_name
from loss import OPENOCC_LOSS
from utils.metric_util import MeanIoU, multi_step_MeanIou
from utils.freeze_model import freeze_model
from utils.misc import downsample_visible_mask
from mmengine.registry import METRICS
import torch.nn.functional as F

from utils.metric_util import multi_step_fid_mmd, multi_step_MeanIou, multi_step_TemporalConsistency


def build_evaluators(cfg, dataloader):
    evaluators = []
    eval_cfgs = cfg.evaluators if isinstance(cfg.evaluators, list) else [cfg.evaluators]
    for eval_cfg in eval_cfgs:
        evaluators.append(
            METRICS.build(eval_cfg, default_args=dict(dataset=dataloader.dataset)))
    return evaluators

def init_ddp(local_rank):
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", 29500)
    hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.environ.get("RANK", 0))  # node id
    ngpus = torch.cuda.device_count()
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", 
        world_size=hosts * ngpus, rank=rank * ngpus + local_rank)
    torch.cuda.set_device(local_rank)


def build_model(cfg, logger, ckpt_path):
    model = MODELS.build(cfg.model)
    model.init_weights()
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {nparams}')

    model = model.cuda()
    if is_distributed():
        if cfg.get('syncBN', True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info('converted sync bn.')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
    return model

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    cfg.ema=args.ema

    # init DDP
    if args.gpus > 0:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", cfg.get("port", 29500))
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')
    tb_dir=args.tb_dir if args.tb_dir  else osp.join(args.work_dir, 'tb_log')
    writer = SummaryWriter(tb_dir)


    my_model = MODELS.build(cfg.model.world_model)
    # my_model.init_weights()
    vae=MODELS.build(cfg.model.vae)

    stage1_cfg = Config.fromfile(cfg.stage_one_config)
    stage1_model = MODELS.build(stage1_cfg.model)

    wm_cfg = Config.fromfile(cfg.world_model_config)
    world_model = MODELS.build(wm_cfg.model.world_model)

    from model.VAE.AE_eval import Autoencoder_2D  # FID MMD
    ae_eval = Autoencoder_2D(num_classes=18, expansion=4)
    ae_ckpt_path = 'ckpts/Occupancy_Generation_ckpt_AE_eval_epoch_196.pth'
    ae_ckpt = torch.load(ae_ckpt_path, map_location="cpu")
    ae_eval.load_state_dict(ae_ckpt["state_dict"], strict=True)
    ae_eval = ae_eval.cuda()
    ae_eval.eval()

    # if cfg.stage_one_config is not None and cfg.stage_one_ckpt is not None:
    #     # stage one model loading
    #     stage1_cfg = Config.fromfile(cfg.stage_one_config)
    #     stage1_model = MODELS.build(stage1_cfg.model).cuda()
    #     _stage1_model = stage1_model.module if hasattr(stage1_model, 'module') else stage1_model
    #     stage1_ckpt_dict = torch.load(cfg.stage_one_ckpt, map_location='cpu')
    #     logger.info(_stage1_model.load_state_dict(stage1_ckpt_dict['state_dict'], strict=True))
    #     stage1_model.requires_grad_(False)
    #     stage1_model.eval()

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    # if cfg.get('freeze_dict', False):
    #     logger.info(f'Freezing model according to freeze_dict:{cfg.freeze_dict}')
    #     freeze_model(my_model, cfg.freeze_dict)

    if args.ema:
        ema=deepcopy(my_model).to('cuda')
        ema.requires_grad_(False)

    if False:
        if cfg.get('syncBN', True):
            vae = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vae)
            # vae = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vae)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        vae = ddp_model_module(
            vae.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        vae = vae.cuda()


    if False:
        if cfg.get('syncBN', True):
            stage1_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(stage1_model)
            # vae = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vae)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        stage1_model = ddp_model_module(
            stage1_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        stage1_model = stage1_model.cuda()



    # load vae
    assert cfg.vae_load_from and osp.exists(cfg.vae_load_from)
    _vae = vae if hasattr(vae, 'module') else vae
    ckpt = torch.load(cfg.vae_load_from, map_location='cpu')
    logger.info(_vae.load_state_dict(ckpt['state_dict'], strict=True))
    vae.eval()
    vae.requires_grad_(False)

    # load stage1 model
    assert cfg.stage_one_ckpt and osp.exists(cfg.stage_one_ckpt)
    _stage1_model = stage1_model.module if hasattr(stage1_model, 'module') else stage1_model
    ckpt = torch.load(cfg.stage_one_ckpt, map_location='cpu')
    logger.info(_stage1_model.load_state_dict(ckpt['state_dict'], strict=True))
    stage1_model.eval()
    stage1_model.requires_grad_(False)

    # load occupancy world model
    assert cfg.world_model_ckpt and osp.exists(cfg.world_model_ckpt)
    _world_model = world_model.module if hasattr(world_model, 'module') else world_model

    if osp.exists(osp.join(args.work_dir, 'best_miou_world_model.pth')):
        world_model_ckpt = osp.join(args.work_dir, 'best_miou_world_model.pth')
        ckpt = torch.load(world_model_ckpt, map_location='cpu')
        load_key='state_dict' 
        logger.info(_world_model.load_state_dict(ckpt[load_key], strict=False))
    else:
        ckpt = torch.load(cfg.world_model_ckpt, map_location='cpu')
        load_key='state_dict' if not cfg.get('ema',False) else 'ema'
        logger.info(_world_model.load_state_dict(ckpt[load_key], strict=False))

    world_model.set_trainable()
    my_model.from_dit(world_model)
    my_model.set_trainable()

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params after freezed: {n_parameters}')

    diffusion = create_diffusion(timestep_respacing="",
        beta_start=cfg.schedule.beta_start,
        beta_end=cfg.schedule.beta_end,
        replace_cond_frames=cfg.replace_cond_frames,
        cond_frames_choices=cfg.cond_frames_choices,
        predict_xstart=cfg.schedule.get('predict_xstart',False),
    )  # default: 1000 steps, linear noise schedule
    diffusion_eval = create_diffusion(
        timestep_respacing=str(cfg.sample.num_sampling_steps),
        beta_start=cfg.schedule.beta_start,
        beta_end=cfg.schedule.beta_end,
        replace_cond_frames=cfg.replace_cond_frames,
        cond_frames_choices=cfg.cond_frames_choices,
        predict_xstart=cfg.schedule.get('predict_xstart',False),
    )

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume,
        )
    
    learnable_params = []
    for model in [my_model, world_model]:
        for param in model.parameters():
            if param.requires_grad:
                learnable_params.append(param)

    optimizer = torch.optim.AdamW(learnable_params, lr=cfg.optimizer.optimizer.lr, weight_decay=cfg.optimizer.optimizer.weight_decay)

    # get optimizer, loss, scheduler
    # optimizer = build_optim_wrapper(torch.nn.Sequential(my_model, world_model), cfg.optimizer)


    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            # vae = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vae)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        # vae = vae.cuda()
        raw_model = my_model


    if distributed:
        if cfg.get('syncBN', True):
            world_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(world_model)
            # vae = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vae)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        world_model = ddp_model_module(
            world_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        world_model = world_model.cuda()



    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('multisteplr', False):
        scheduler = MultiStepLRScheduler(
            optimizer,
            **cfg.multisteplr_config)
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False)


    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0
    best_val_iou = [0]*cfg.get('return_len_', 10)
    best_val_miou = [0]*cfg.get('return_len_', 10)

    best_val_mean_iou = 0
    best_val_mean_miou = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'best_miou_controlnet.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'best_miou_controlnet.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if args.load_from:
        cfg.load_from = args.load_from
    if args.vae_load_from:
        cfg.vae_load_from = args.vae_load_from

    logger.info('resume from: ' + cfg.resume_from)
    logger.info('load from: ' + cfg.load_from)
    logger.info('vae_load_from: ' + cfg.vae_load_from)
    logger.info('work dir: ' + args.work_dir)

    evaluators = build_evaluators(cfg, val_dataset_loader)


    is_resume=False
    # load DiT
    if cfg.resume_from and osp.exists(cfg.resume_from):
        # assert False
        is_resume=True
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
        if args.ema and 'ema' in ckpt:
            print(ema.load_state_dict(ckpt['ema']))
            
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        # assert False,'Only use for fintune'
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        # raw_model.load_state_dict(state_dict, strict=False) #TODO may need to remove moudle.xxx
        load_checkpoint(raw_model,state_dict, strict=False) #TODO may need to remove moudle.xxx

    # zero init pose encoder
    # if not is_resume and 'pose_encoder' in cfg.model.world_model and cfg.model.world_model.pose_encoder.get('zero_init', False):
    #     zero_params=cfg.model.world_model.pose_encoder.get('zero_params', [])
    #     raw_model.pose_encoder.zero_module(zero_params)
        

        
    # training
    print_freq = cfg.print_freq
    first_run = True
    grad_norm = 0
    
    label_name = get_nuScenes_label_name(cfg.label_mapping)
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', times=cfg.get('eval_length', 10))
    CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', times=cfg.get('eval_length', 10))
    Cal_fid_mmd = multi_step_fid_mmd()
    Cal_TC = multi_step_TemporalConsistency("TC", times=cfg.get('eval_length', 10))

    # logger.info('compiling model')
    # my_model = torch.compile(my_model)
    # logger.info('done compile model')
    best_plan_loss = 100000
    # max_num_epochs=1 #debug

    scaler = GradScaler()

    vae_scale_factor = 2 ** (len(cfg.model.vae.encoder_cfg.ch_mult) - 1)
    channel_factor=cfg.model.vae.encoder_cfg.in_channels / cfg.model.vae.decoder_cfg.z_channels
    if args.ema and not is_resume:
        update_ema(ema, raw_model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode

    while epoch < max_num_epochs:
        
        my_model.train()
        os.environ['eval'] = 'false'
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()

        start_frame=cfg.get('start_frame', 0)
        mid_frame=cfg.get('mid_frame', 3)
        # mid_frame=cfg.get('mid_frame', 4)
        end_frame=cfg.get('end_frame', 10)

        for i_iter, (input_occs, target_occs, metas) in enumerate(train_dataset_loader):
            # if i_iter>3: #debug
            #     break
            SCENE_RANGE = (-40, -40, -1, 40, 40, 5.4)

            inputs_dict = dict()
            inputs_dict["source_occs"] = input_occs[:,start_frame: mid_frame].clone().permute(0,1,4,3,2)
            inputs_dict["target_occs"] = input_occs[:,mid_frame: end_frame].clone().permute(0,1,4,3,2)
            inputs_dict["source_metas"] = dict()
            inputs_dict["target_metas"] = dict()

            inputs_dict["source_metas"] = []
            inputs_dict["target_metas"] = []
            inputs_dict["metas"] = []

            for bs in range(len(metas)):
                inputs_dict["source_metas"].append(dict(ego2global=[]))
                inputs_dict["target_metas"].append(dict(ego2global=[]))
                inputs_dict["source_metas"]
                inputs_dict["target_metas"]
                inputs_dict["metas"].append(dict())
                inputs_dict["metas"][bs]['scene_range'] = SCENE_RANGE
                for frame_idx in range(end_frame - start_frame):
                    e2g_t = metas[bs]['e2g_t'][frame_idx]
                    e2g_r = metas[bs]['e2g_r'][frame_idx]
                    ego2global = np.eye(4)
                    ego2global[:3,:3] = Quaternion(e2g_r).rotation_matrix
                    ego2global[:3, 3] = e2g_t
                    if frame_idx < mid_frame:
                        inputs_dict["source_metas"][bs]["ego2global"].append(ego2global)
                    else:
                        inputs_dict["target_metas"][bs]["ego2global"].append(ego2global)
                       
            with torch.no_grad():
                stage1_outputs_dict = stage1_model(inputs_dict)

            if first_run:
                i_iter = i_iter + last_iter
            
            input_occs = input_occs.cuda() #torch.Size([1, 11, 200, 200, 16])
            # target_occs = target_occs.cuda()
            data_time_e = time.time()
            use_pose_condition = torch.rand(1) < cfg.p_use_pose_condition
            x=input_occs
            bs, _, _, _, _ = x.shape
            x, shape = vae.forward_encoder(x) #16 128 50 50
            # vae sample
            x,_,_=vae.sample_z(x) #16 64 50 50
            x*=cfg.model.vae.scaling_factor
            if x.dim()==4:
                x = rearrange(x, '(b f) c h w -> b f c h w', b=bs).contiguous()
            elif x.dim()==5:
                x = rearrange(x, 'b c f h w -> b f c h w', b=bs).contiguous()
            else:
                raise NotImplementedError
            # z = self.vae.vqvae.quant_conv(z)
            # z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
            future_occs_pred = stage1_outputs_dict["sem_preds"].clone().permute(0,1,4,3,2) # torch.Size([2, 6, 200, 200, 16])
            # stage1_outputs_dict['invisible_mask']


            input_occs_condition = input_occs[:,start_frame: mid_frame]
            input_occs_condition = torch.cat([input_occs_condition, future_occs_pred], dim=1)
            bs,f,_,_,_ = input_occs_condition.shape
            encoded_latent, shape = vae.forward_encoder(input_occs_condition)
            encoded_latent,_,_ = vae.sample_z(encoded_latent) #bchw
            input_latents = encoded_latent * cfg.model.vae.scaling_factor
            if input_latents.dim()==4:
                input_latents = rearrange(input_latents, '(b f) c h w -> b f c h w', b=bs).contiguous()
            elif input_latents.dim()==5:
                input_latents = rearrange(input_latents, 'b c f h w -> b f c h w', b=bs).contiguous()
            else:
                raise NotImplementedError


            stage1_invisible_mask = stage1_outputs_dict['invisible_mask'].permute(0,1,4,3,2)  # True: Invisible; False: Visible
            
            
            stage1_invisible_mask_downsampled = downsample_visible_mask(stage1_invisible_mask)
            # stage1_invisible_mask_downsampled_v2 = downsample_visible_mask_v2(stage1_invisible_mask)
            
            stage1_invisible_mask_downsampled = stage1_invisible_mask_downsampled.unsqueeze(2).repeat(1,1,input_latents.shape[2],1,1)
            stage1_invisible_mask_downsampled_past = torch.zeros([stage1_invisible_mask_downsampled.shape[0], mid_frame-start_frame, stage1_invisible_mask_downsampled.shape[2], stage1_invisible_mask_downsampled.shape[3], stage1_invisible_mask_downsampled.shape[4]], dtype=torch.bool, device=stage1_invisible_mask_downsampled.device)
            stage1_invisible_mask_downsampled = torch.cat([stage1_invisible_mask_downsampled_past, stage1_invisible_mask_downsampled], dim=1)
            input_latents[stage1_invisible_mask_downsampled] = 0
            model_kwargs=dict(
                condition = input_latents,
                # metas=metas,
            )
            if use_pose_condition:
                model_kwargs['metas'] = metas
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],)).cuda()
            with torch.autocast(device_type="cuda"):#,dtype=torch.float16):
                # x: torch.Size([8, 11, 64, 25, 25])
                # y: torch.Size([8])
                loss_dict = diffusion.training_losses(world_model, x, t, model_kwargs=model_kwargs, controlnet=my_model)
                loss = loss_dict["loss"].mean()
                # loss.backward()
            optimizer.zero_grad()

            # loss.backward()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()


            if args.ema:
                update_ema(ema, raw_model)

            loss_list.append(loss.detach().cpu().item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and local_rank == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.3f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader), 
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s))
                writer.add_scalar(f'train/loss', loss.item(), global_iter)
                writer.add_scalar(f'train/lr', lr, global_iter)
                writer.add_scalar(f'train/grad_norm', grad_norm, global_iter)
                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    loss_value=loss_value.mean().item()
                    detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                    writer.add_scalar(f'train/{loss_name}', loss_value, global_iter)
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)
                loss_list = []
                # exit(0) #debug
            data_time_s = time.time()
            time_s = time.time()

            if args.iter_resume:
                if (i_iter + 1) % 50 == 0 and local_rank == 0:
                    dict_to_save = {
                        'state_dict': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_iter': global_iter,
                        'last_iter': i_iter + 1,
                    }
                    if args.ema:
                        dict_to_save['ema'] = ema.state_dict()
                    save_file_name = os.path.join(os.path.abspath(args.work_dir), 'iter.pth')
                    torch.save(dict_to_save, save_file_name)
                    dst_file = osp.join(args.work_dir, 'latest.pth')
                    # # symlink(save_file_name, dst_file) #bug in cluster
                    logger.info(f'iter ckpt {i_iter + 1} saved!')
            # break #debug
        
        # save checkpoint
        if local_rank == 0 and (epoch+1) % cfg.get('save_every_epochs', 1) == 0:
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
            }
            if args.ema:
                dict_to_save['ema'] = ema.state_dict()
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            # symlink(save_file_name, dst_file) #bug in cluster

        if local_rank == 0 and (epoch+1) % cfg.get('save_every_epochs', 1) == 0:
            dict_to_save = {
                'state_dict': _world_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
            }
            # if args.ema:
            #     dict_to_save['ema'] = ema.state_dict()
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}_world_model.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest_world_model.pth')
            # symlink(save_file_name, dst_file) #bug in cluster


        epoch += 1
        first_run = False
        
        # eval
        if epoch % cfg.get('eval_every_epochs', 1) == 0:
            # torch.cuda.empty_cache()
            my_model.eval()
            eval_model = world_model
            controlnet = my_model if not args.ema else ema
            os.environ['eval'] = 'true'
            val_loss_list = []
            CalMeanIou_sem.reset()
            CalMeanIou_vox.reset()
            Cal_TC.reset()


    

            for evaluator in evaluators:
                evaluator.clean()
            plan_loss = 0
            
            start_frame=cfg.get('start_frame', 0)
            mid_frame=cfg.get('mid_frame', 3)
            # mid_frame=cfg.get('mid_frame', 4)
            end_frame=cfg.get('end_frame', 10)

            with torch.no_grad():
                for i_iter_val, (input_occs, target_occs, metas) in tqdm(enumerate(val_dataset_loader),total=len(val_dataset_loader)):
                    # if i_iter_val>3: #debug
                    #     break
                    # assert (input_occs==target_occs).all()

                    SCENE_RANGE = (-40, -40, -1, 40, 40, 5.4)

                    inputs_dict = dict()
                    inputs_dict["source_occs"] = input_occs[:,start_frame: mid_frame].clone().permute(0,1,4,3,2)
                    inputs_dict["target_occs"] = input_occs[:,mid_frame: end_frame].clone().permute(0,1,4,3,2)
                    inputs_dict["source_metas"] = dict()
                    inputs_dict["target_metas"] = dict()

                    inputs_dict["source_metas"] = []
                    inputs_dict["target_metas"] = []
                    inputs_dict["metas"] = []

                    for bs in range(len(metas)):
                        inputs_dict["source_metas"].append(dict(ego2global=[]))
                        inputs_dict["target_metas"].append(dict(ego2global=[]))
                        inputs_dict["source_metas"]
                        inputs_dict["target_metas"]
                        inputs_dict["metas"].append(dict())
                        inputs_dict["metas"][bs]['scene_range'] = SCENE_RANGE
                        for frame_idx in range(end_frame - start_frame):
                            e2g_t = metas[bs]['e2g_t'][frame_idx]
                            e2g_r = metas[bs]['e2g_r'][frame_idx]
                            ego2global = np.eye(4)
                            ego2global[:3,:3] = Quaternion(e2g_r).rotation_matrix
                            ego2global[:3, 3] = e2g_t
                            if frame_idx < mid_frame:
                                inputs_dict["source_metas"][bs]["ego2global"].append(ego2global)
                            else:
                                inputs_dict["target_metas"][bs]["ego2global"].append(ego2global)
                            
                    with torch.no_grad():
                        stage1_outputs_dict = stage1_model(inputs_dict)


                    future_occs_pred = stage1_outputs_dict["sem_preds"].clone().permute(0,1,4,3,2)

                    input_occs = input_occs.cuda()
                    input_occs[:,mid_frame:end_frame] = future_occs_pred

                    target_occs=target_occs.cuda()
                    data_time_e = time.time()
                    # encode the input occ
                    
                    x=input_occs
                    bs, F, H, W, D = x.shape
                    x, shape = vae.forward_encoder(x) #16 128 50 50
                    # vae sample
                    x,_,_=vae.sample_z(x) #16 64 50 50
                    input_latents=x*cfg.model.vae.scaling_factor

                    if input_latents.dim()==4:
                        input_latents = rearrange(input_latents, '(b f) c h w -> b f c h w', b=bs).contiguous()
                    elif input_latents.dim()==5:
                        input_latents = rearrange(input_latents, 'b c f h w -> b f c h w', b=bs).contiguous()
                    else:
                        raise NotImplementedError

                    w=h=cfg.model.vae.encoder_cfg.resolution
                    # vae_scale_factor = 2 ** (len(cfg.model.vae.encoder_cfg.ch_mult) - 1)
                    vae_docoder_shapes=cfg.shapes[:len(cfg.model.vae.encoder_cfg.ch_mult) - 1]
                    w//=vae_scale_factor
                    h//=vae_scale_factor


                    stage1_invisible_mask = stage1_outputs_dict['invisible_mask'].permute(0,1,4,3,2)  # True: Invisible; False: Visible
                    stage1_invisible_mask_downsampled = downsample_visible_mask(stage1_invisible_mask)
                    # stage1_invisible_mask_downsampled_v2 = downsample_visible_mask_v2(stage1_invisible_mask)
                    stage1_invisible_mask_downsampled = stage1_invisible_mask_downsampled.unsqueeze(2).repeat(1,1,input_latents.shape[2],1,1)
                    stage1_invisible_mask_downsampled_past = torch.zeros([stage1_invisible_mask_downsampled.shape[0], mid_frame-start_frame, stage1_invisible_mask_downsampled.shape[2], stage1_invisible_mask_downsampled.shape[3], stage1_invisible_mask_downsampled.shape[4]], dtype=torch.bool, device=stage1_invisible_mask_downsampled.device)
                    stage1_invisible_mask_downsampled = torch.cat([stage1_invisible_mask_downsampled_past, stage1_invisible_mask_downsampled], dim=1)
                    input_latents[stage1_invisible_mask_downsampled] = 0
                    model_kwargs=dict(
                        condition = input_latents,
                        # cfg_scale=cfg.sample.guidance_scale
                        metas=metas
                    )
                    noise_shape=(bs, end_frame,cfg.base_channel+cfg.get('len_additonal_channel',0), w,h,)
                    initial_cond_indices=None
                    n_conds=cfg.sample.get('n_conds',0)
                    if n_conds:
                        initial_cond_indices=[index for index in range(n_conds)]
                    
                    # Sample images:
                    if cfg.sample.sample_method == 'ddim':
                        latents = diffusion_eval.ddim_sample_loop(
                            eval_model,  noise_shape, None, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device='cuda'
                        )
                    elif cfg.sample.sample_method == 'ddpm':
                        latents = diffusion_eval.p_sample_loop(
                            eval_model,  noise_shape, None, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device='cuda',
                            initial_cond_indices=initial_cond_indices,
                            initial_cond_frames=input_latents, controlnet=controlnet,
                        )
                    latents = 1 / cfg.model.vae.scaling_factor * latents

                    if cfg.model.vae.decoder_cfg.type=='Decoder3D':
                        latents = rearrange(latents,'b f c h w-> b c f h w')
                    else:
                        assert False #debug
                        latents = rearrange(latents,'b f c h w -> (b f) c h w')
                                # post process for two stream

                    logits = vae.forward_decoder(
                            latents , shapes=vae_docoder_shapes,input_shape=[bs,end_frame,*cfg.shapes[0],cfg._dim_]
                        )
                    z_q_predict=logits
                    
                    target_occs_d=target_occs.clone()
                    result_dict={
                        'target_occs':target_occs[:, mid_frame:end_frame],
                    }
                    # z_q_predict=z_q_predict[:,mid_frame:end_frame]
                    pred = z_q_predict.argmax(dim=-1).detach().cuda()
                    pred_d=pred.clone()
                    pred=pred[:,mid_frame:end_frame]
                    result_dict['sem_pred'] = pred
                    pred_iou = deepcopy(pred)
                    
                    pred_iou[pred_iou!=17] = 1
                    pred_iou[pred_iou==17] = 0
                    result_dict['iou_pred'] = pred_iou
                    
                    loss_input = {
                        'inputs': input_occs,
                        'target_occs': target_occs,
                        # 'metas': metas
                    }
                    for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                        loss_input.update({
                            loss_input_key: result_dict[loss_input_val]
                        })
                    # loss, loss_dict = loss_func(loss_input)
                    loss_dict={}
                    loss=torch.zeros(1)
                    if result_dict.get('target_occs', None) is not None:
                        target_occs = result_dict['target_occs']
                    target_occs_iou = deepcopy(target_occs)
                    target_occs_iou[target_occs_iou != 17] = 1
                    target_occs_iou[target_occs_iou == 17] = 0
                    
                    val_miou, _ = CalMeanIou_sem._after_step(result_dict['sem_pred'], target_occs,log_current=True)
                    val_iou, _ = CalMeanIou_vox._after_step(result_dict['iou_pred'], target_occs_iou,log_current=True)


                    ae_feature_ori = ae_eval.forward_eval(target_occs)  # B*T,2048
                    ae_feature_gen = ae_eval.forward_eval(result_dict['sem_pred'])

                    Cal_fid_mmd._after_step(ae_feature_ori, ae_feature_gen)
                    Cal_TC._after_step(ae_feature_gen.reshape(bz, p_time, -1))


                    for i, evaluator in enumerate(evaluators):
                        evaluator.update(result_dict, result_dict)

                    # if distributed:
                    #     val_miou=dist.all_reduce(val_miou)
                    #     val_iou=dist.all_reduce(val_iou)
                    val_loss_list.append(loss.detach().cpu().numpy())
                    if i_iter_val % print_freq == 0 and local_rank == 0:
                        logger.info('[EVAL] Epoch %d Iter %5d/%5d: Loss: %.3f (%.3f)'%(
                            epoch, i_iter_val,len(val_dataset_loader), loss.item(), np.mean(val_loss_list)))
                        detailed_loss = []
                        for loss_name, loss_value in loss_dict.items():
                            detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                        detailed_loss = ', '.join(detailed_loss)
                        logger.info(detailed_loss)

                    ####################### debug vis
                    if False:
                        # debug
                        # val_miou, _ = CalMeanIou_sem._after_epoch()
                        # val_iou, _ = CalMeanIou_vox._after_epoch()
                        # logger.info(f'PlanRegLoss is {plan_loss/len(val_dataset_loader)}')
                        # logger.info(f'{i_iter_val:06d}'+f'{i_iter_val:06d}')
                        logger.info(f'rank:{local_rank}_{i_iter_val:06d}_'+f'Current val iou is {val_iou}')
                        logger.info(f'rank:{local_rank}_{i_iter_val:06d}_'+f'Current val miou is {val_miou}')
                        logger.info(f'rank:{local_rank}_{i_iter_val:06d}_'+f'avg val iou is {(val_iou[1]+val_iou[3]+val_iou[5])/3}')
                        logger.info(f'rank:{local_rank}_{i_iter_val:06d}_'+f'avg val miou is {(val_miou[1]+val_miou[3]+val_miou[5])/3}')

                        logger.info(f'iou:rank:{local_rank}_{i_iter_val:06d}::total_seen: {CalMeanIou_sem.total_seen.sum()}')
                        logger.info(f'iou:rank:{local_rank}_{i_iter_val:06d}::total_correct: {CalMeanIou_sem.total_correct.sum()}')
                        logger.info(f'iou:rank:{local_rank}_{i_iter_val:06d}::total_positive: {CalMeanIou_sem.total_positive.sum()}')
                        logger.info(f'miou:rank:{local_rank}_{i_iter_val:06d}::total_seen: {CalMeanIou_vox.total_seen.sum()}')
                        logger.info(f'miou:rank:{local_rank}_{i_iter_val:06d}::total_correct: {CalMeanIou_vox.total_correct.sum()}')
                        logger.info(f'miou:rank:{local_rank}_{i_iter_val:06d}::total_positive: {CalMeanIou_vox.total_positive.sum()}')

                        
            val_miou, _ = CalMeanIou_sem._after_epoch()
            val_iou, _ = CalMeanIou_vox._after_epoch()
            del target_occs, input_occs
            
            best_val_iou = [max(best_val_iou[i], val_iou[i]) for i in range(min(len(best_val_iou),len(val_iou)))]
            best_val_miou = [max(best_val_miou[i], val_miou[i]) for i in range(min(len(best_val_miou),len(val_miou)))]
            
            # logger.info(f'Current val iou is {val_iou}')
            # logger.info(f'Current val miou is {val_miou}')
            logger.info(f'Current val iou is {val_iou} while the best val iou is {best_val_iou}')
            logger.info(f'Current val miou is {val_miou} while the best val miou is {best_val_miou}')
            logger.info(f'avg val iou is {(val_iou[1]+val_iou[3]+val_iou[5])/3}')
            logger.info(f'avg val miou is {(val_miou[1]+val_miou[3]+val_miou[5])/3}')
            writer.add_scalar(f'val/iou', (val_iou[1]+val_iou[3]+val_iou[5])/3, global_iter)
            writer.add_scalar(f'val/miou', (val_miou[1]+val_miou[3]+val_miou[5])/3, global_iter)
            val_TC = Cal_TC._after_epoch()
            logger.info(f"Avg TC: %.4f" % (val_TC))
            if local_rank == 0:
                fid, mmd = Cal_fid_mmd._after_epoch()
                logger.info(f"FID: %.4f" % (fid))
                logger.info(f"MMD: %.6f" % (mmd))

            # evaluate metrics
            logger.info(f'Evaluate results!')
            for evaluator in evaluators:
                metric_name = evaluator.name
                evaluator.eval()
                logger.info(f'{metric_name}: \n' + evaluator.format_string())


            save_best_iou_flag = False
            save_best_miou_flag = False

            if best_val_mean_iou < (val_iou[1]+val_iou[3]+val_iou[5])/3:
                best_val_mean_iou = (val_iou[1]+val_iou[3]+val_iou[5])/3
                save_best_iou_flag = True

            if best_val_mean_miou < (val_miou[1]+val_miou[3]+val_miou[5])/3:
                best_val_mean_miou = (val_miou[1]+val_miou[3]+val_miou[5])/3
                save_best_miou_flag = True


            # save checkpoint
            if local_rank == 0 and save_best_iou_flag:
                dict_to_save = {
                    'state_dict': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_iter': global_iter,
                }
                if args.ema:
                    dict_to_save['ema'] = ema.state_dict()
                save_file_name = os.path.join(os.path.abspath(args.work_dir), f'best_iou_controlnet.pth')
                torch.save(dict_to_save, save_file_name)
                dst_file = osp.join(args.work_dir, 'latest.pth')

            if local_rank == 0 and save_best_iou_flag:
                dict_to_save = {
                    'state_dict': _world_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_iter': global_iter,
                }

                save_file_name = os.path.join(os.path.abspath(args.work_dir), f'best_iou_world_model.pth')
                torch.save(dict_to_save, save_file_name)


            # save checkpoint
            if local_rank == 0 and save_best_miou_flag:
                dict_to_save = {
                    'state_dict': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_iter': global_iter,
                }
                if args.ema:
                    dict_to_save['ema'] = ema.state_dict()
                save_file_name = os.path.join(os.path.abspath(args.work_dir), f'best_miou_controlnet.pth')
                torch.save(dict_to_save, save_file_name)
                dst_file = osp.join(args.work_dir, 'latest.pth')

            if local_rank == 0 and save_best_miou_flag:
                dict_to_save = {
                    'state_dict': _world_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_iter': global_iter,
                }
                # if args.ema:
                #     dict_to_save['ema'] = ema.state_dict()
                save_file_name = os.path.join(os.path.abspath(args.work_dir), f'best_miou_world_model.pth')
                torch.save(dict_to_save, save_file_name)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='configs/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--tb-dir', type=str, default=None)
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--vae_load_from', type=str, default=None)
    parser.add_argument('--ema', type=bool, default=True)
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)


