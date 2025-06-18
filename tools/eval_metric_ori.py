import cv2
from tqdm import tqdm
import pdb
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


from utils.load_save_util import revise_ckpt, revise_ckpt_1
import warnings
warnings.filterwarnings("ignore")
from diffusion import create_diffusion
from einops import rearrange

TIMESTAMP = time.strftime('%Y%m%d_%H%M%S', time.localtime())



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

    # init DDP
    if args.gpus > 1:
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
    if args.dst_dir is not None:
        log_file = osp.join(args.dst_dir, f'eval_stp3_{cfg.start_frame}_{cfg.mid_frame}_{cfg.end_frame}_{timestamp}.log')
    else:
        log_file = osp.join(args.work_dir, f'eval_stp3_{cfg.start_frame}_{cfg.mid_frame}_{cfg.end_frame}_{timestamp}.log')
    logger = MMLogger('genocc', log_file=log_file,distributed=distributed)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader, get_nuScenes_label_name
    from loss import OPENOCC_LOSS
    from utils.metric_util import MeanIoU, multi_step_MeanIou
    from utils.freeze_model import freeze_model

    my_model = MODELS.build(cfg.model.world_model)
    # my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if cfg.get('freeze_dict', False):
        logger.info(f'Freezing model according to freeze_dict:{cfg.freeze_dict}')
        freeze_model(my_model, cfg.freeze_dict)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params after freezed: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
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
        raw_model = my_model
    vae=MODELS.build(cfg.model.vae).cuda()
    vae.requires_grad_(False)
    vae.eval()
    
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume)

    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0
    best_val_iou = [0]*cfg.get('return_len_', 10)
    best_val_miou = [0]*cfg.get('return_len_', 10)

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    else:
        ckpts=[i for i in os.listdir(args.work_dir) if 
            i.endswith('.pth') and i.replace('.pth','').replace('epoch_','').isdigit()]
        if len(ckpts)>0:
            ckpts.sort(key=lambda x:int(x.replace('.pth','').replace('epoch_','')))
            cfg.resume_from = osp.join(args.work_dir, ckpts[-1])

    if args.resume_from:
        cfg.resume_from = args.resume_from
    cfg.vae_resume_from=args.vae_resume_from
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('vae resume from: ' + cfg.vae_resume_from)
    logger.info('work dir: ' + args.work_dir)
     
    assert cfg.resume_from and osp.exists(cfg.resume_from)
    if True:
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        load_key='state_dict' if not cfg.get('ema',False) else 'ema'
        print(raw_model.load_state_dict(ckpt[load_key], strict=False))
        epoch = ckpt['epoch']
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
            
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            assert False
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    print(vae.load_state_dict(torch.load(cfg.vae_resume_from)['state_dict']))
    # training
    print_freq = cfg.print_freq

    label_name = get_nuScenes_label_name(cfg.label_mapping)
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', times=cfg.get('eval_length'))
    CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', times=cfg.get('eval_length'))
    
    my_model.eval()
    os.environ['eval'] = 'true'

    diffusion = create_diffusion(
        timestep_respacing=str(cfg.sample.num_sampling_steps),
        beta_start=cfg.schedule.beta_start,
        beta_end=cfg.schedule.beta_end,
        replace_cond_frames=cfg.replace_cond_frames,
        cond_frames_choices=cfg.cond_frames_choices,
        predict_xstart=cfg.schedule.get('predict_xstart',False),
    )
    val_loss_list = []
    CalMeanIou_sem.reset()
    CalMeanIou_vox.reset()
    metric_stp3 = {}
    time_used = {
        'encode':0,
        'mid':0,
        'autoreg':0,
        'total':0,
        'per_frame':0,
    }
    start_frame=cfg.get('start_frame', 0)
    mid_frame=cfg.get('mid_frame', 3)
    end_frame=cfg.get('end_frame', 9)
    assert cfg.sample.n_conds==mid_frame

    with torch.no_grad():
        for i_iter_val, (input_occs, target_occs, metas) in tqdm(enumerate(val_dataset_loader)):
        #     pass
        # exit()
        # if True:
            # if i_iter_val>50: #debug
            #     break
            input_occs = input_occs.cuda()
            target_occs = target_occs.cuda()
            # assert (input_occs==target_occs).all()

            input_occs[:, mid_frame:end_frame] = 0
            data_time_e = time.time()
            # encode the input occ
            bs,f,_,_,_=input_occs.shape
            encoded_latent, shape=vae.forward_encoder(input_occs)
            encoded_latent,_,_=vae.sample_z(encoded_latent) #bchw
            input_latents=encoded_latent*cfg.model.vae.scaling_factor

            if input_latents.dim()==4:
                input_latents = rearrange(input_latents, '(b f) c h w -> b f c h w', b=bs).contiguous()
            elif input_latents.dim()==5:
                input_latents = rearrange(input_latents, 'b c f h w -> b f c h w', b=bs).contiguous()
            else:
                raise NotImplementedError

            w=h=cfg.model.vae.encoder_cfg.resolution
            vae_scale_factor = 2 ** (len(cfg.model.vae.encoder_cfg.ch_mult) - 1)
            w//=vae_scale_factor
            h//=vae_scale_factor

            model_kwargs=dict(
                # cfg_scale=cfg.sample.guidance_scale
                metas=metas
            )
            noise_shape=(bs, end_frame,cfg.base_channel, w,h,)
            initial_cond_indices=None
            n_conds=cfg.sample.get('n_conds',0)
            if n_conds:
                initial_cond_indices=[index for index in range(n_conds)]
            
            # Sample images:
            if cfg.sample.sample_method == 'ddim':
                latents = diffusion.ddim_sample_loop(
                    my_model,  noise_shape, None, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device='cuda'
                )
            elif cfg.sample.sample_method == 'ddpm':
                latents = diffusion.p_sample_loop(
                    my_model,  noise_shape, None, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device='cuda',
                    initial_cond_indices=initial_cond_indices,
                    initial_cond_frames=input_latents,
                )
            latents = 1 / cfg.model.vae.scaling_factor * latents

            if cfg.model.vae.decoder_cfg.type=='Decoder3D':
                latents = rearrange(latents,'b f c h w-> b c f h w')
            else:
                assert False #debug
                latents = rearrange(latents,'b f c h w -> (b f) c h w')

            z_q_predict = vae.forward_decoder(
                latents , shapes=[[200, 200], [100, 100],[50,50]],input_shape=[bs,end_frame,200,200,cfg._dim_]
            )
            
            target_occs_d=target_occs.clone()
            result_dict={
                'target_occs':target_occs[:, mid_frame:end_frame],
                'metric_stp3':metric_stp3,
                'time':{'encode': 0, 'mid': 0, 'autoreg': 0, 'total': 0, 'per_frame': 0} #TODO
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


            for key in metric_stp3.keys():
                metric_stp3[key] += result_dict['metric_stp3'][key]
            for key in time_used.keys():
                time_used[key] += result_dict['time'][key]
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
            if val_iou[0]>30:
                continue
            ####################### debug vis
            if True:
                # debug
                # val_miou, _ = CalMeanIou_sem._after_epoch()
                # val_iou, _ = CalMeanIou_vox._after_epoch()
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

    metric_stp3 = {key:metric_stp3[key]/len(val_dataset_loader) for key in metric_stp3.keys()}
    time_used = {key:time_used[key]/len(val_dataset_loader) for key in time_used.keys()}
    # reduce for distributed
    if distributed:
        metric_stp3 = {key:torch.tensor(metric_stp3[key],dtype=torch.float64).cuda() for key in metric_stp3.keys()}
        for key in metric_stp3.keys():
            dist.all_reduce(metric_stp3[key])
            metric_stp3[key] /= world_size
        time_used = {key:torch.tensor(time_used[key],dtype=torch.float64).cuda() for key in time_used.keys()}
        for key in time_used.keys():
            dist.all_reduce(time_used[key])
            time_used[key] /= world_size
    for key in metric_stp3.keys():
        try:
            metric_stp3[key] = metric_stp3[key].item()
        except:
            pass
        logger.info(f'{key} is {metric_stp3[key]}')
    #logger.info(f'metric_stp3 is {metric_stp3}')
    logger.info(f'time_used is {time_used}')
    logger.info(f'FPS is {1/(time_used["per_frame"]+1e-6)}')
                
    val_miou, _ = CalMeanIou_sem._after_epoch()
    val_iou, _ = CalMeanIou_vox._after_epoch()
    del target_occs, input_occs
    
    #best_val_iou = [max(best_val_iou[i], val_iou[i]) for i in range(len(best_val_iou))]
    #best_val_miou = [max(best_val_miou[i], val_miou[i]) for i in range(len(best_val_miou))]
    
    logger.info(f'Current val iou is {val_iou}')
    logger.info(f'Current val miou is {val_miou}')
    logger.info(f'avg val iou is {(val_iou[1]+val_iou[3]+val_iou[5])/3}')
    logger.info(f'avg val miou is {(val_miou[1]+val_miou[3]+val_miou[5])/3}')
    #logger.info(f'Current val iou is {val_iou} while the best val iou is {best_val_iou}')
    #logger.info(f'Current val miou is {val_miou} while the best val miou is {best_val_miou}')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='configs/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--dst-dir', type=str, default=None)
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--vae-resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
        