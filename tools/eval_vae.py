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
from mmengine.registry import MODELS, METRICS
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.load_save_util import revise_ckpt, revise_ckpt_1, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
import warnings
from einops import rearrange
warnings.filterwarnings("ignore")


def build_evaluators(cfg, dataloader):
    evaluators = []
    eval_cfgs = cfg.evaluators if isinstance(cfg.evaluators, list) else [cfg.evaluators]
    for eval_cfg in eval_cfgs:
        evaluators.append(
            METRICS.build(eval_cfg, default_args=dict(dataset=dataloader.dataset)))
    return evaluators

def pass_print(*args, **kwargs):
    pass

@torch.no_grad()
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
        port = os.environ.get("MASTER_PORT", cfg.get("port", 29510))
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
    log_file = osp.join(args.work_dir, f'{timestamp}_eval.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')
    tb_dir=args.tb_dir if args.tb_dir  else osp.join(args.work_dir, 'tb_log')
    writer = SummaryWriter(tb_dir)

    # build model
    import model
    from dataset import get_dataloader, get_nuScenes_label_name
    from loss import OPENOCC_LOSS
    from utils.metric_util import MeanIoU, multi_step_MeanIou
    from utils.freeze_model import freeze_model

    my_model = MODELS.build(cfg.model)
    my_model.init_weights()
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

        find_unused_parameters = cfg.get('find_unused_parameters', True)
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
    
    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
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

    cfg.resume_from = ''
    # if osp.exists(osp.join(args.work_dir, 'latest.pth')):
    #     cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if args.load_from:
        cfg.load_from = args.load_from

    logger.info('resume from: ' + cfg.resume_from)
    logger.info('load from: ' + cfg.load_from)
    logger.info('work dir: ' + args.work_dir)

    evaluators = build_evaluators(cfg, val_dataset_loader)


    if cfg.resume_from and osp.exists(cfg.resume_from):
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
            
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        if cfg.get('revise_ckpt', False):
            if cfg.revise_ckpt == 1:
                print('revise_ckpt')
                print(raw_model.load_state_dict(revise_ckpt(state_dict), strict=False))
            elif cfg.revise_ckpt == 2:
                print('revise_ckpt_1')
                print(raw_model.load_state_dict(revise_ckpt_1(state_dict), strict=False))
            elif cfg.revise_ckpt == 3:
                print('revise_ckpt_2')
                print(raw_model.vae.load_state_dict(state_dict, strict=False))
        else:
            # print(raw_model.load_state_dict(state_dict, strict=False))
            load_checkpoint(raw_model,state_dict, strict=False) #TODO may need to remove moudle.xxx
        
    # training
    print_freq = cfg.print_freq
    grad_norm = 0
    
    label_name = get_nuScenes_label_name(cfg.label_mapping)
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    # CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', times=cfg.get('return_len_', 10))
    # CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', times=cfg.get('return_len_', 10))
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', times=1)#cfg.get('return_len_', 10))
    CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', times=1)#cfg.get('return_len_', 10))

    for evaluator in evaluators:
        evaluator.clean()

    # logger.info('compiling model')
    # my_model = torch.compile(my_model)
    # logger.info('done compile model')
    best_plan_loss = 100000
    # max_num_epochs=1 #debug
    if True:
        my_model.eval()
        os.environ['eval'] = 'true'
        val_loss_list = []
        CalMeanIou_sem.reset()
        CalMeanIou_vox.reset()
        plan_loss = 0
        
        with torch.no_grad():
            for i_iter_val, (input_occs, target_occs, metas) in enumerate(val_dataset_loader):
                # input_occs=rearrange(input_occs,'b (f1 f) h w d-> (b f) (f1 1) h w d',f1=2)
                # target_occs=rearrange(target_occs,'b f h w d-> (b f) 1 h w d')
                
                input_occs = input_occs.cuda()
                target_occs = target_occs.cuda()
                data_time_e = time.time()
                
                result_dict = my_model(x=input_occs, metas=metas)

                loss_input = {
                    'inputs': input_occs,
                    'target_occs': target_occs,
                    # 'metas': metas
                    **result_dict
                }
                for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                    loss_input.update({
                        loss_input_key: result_dict[loss_input_val]
                    })
                loss, loss_dict = loss_func(loss_input)
                plan_loss += loss_dict.get('PlanRegLoss', 0)
                plan_loss += loss_dict.get('PlanRegLossLidar', 0)
                if result_dict.get('target_occs', None) is not None:
                    target_occs = result_dict['target_occs']
                target_occs_iou = deepcopy(target_occs)
                target_occs_iou[target_occs_iou != 17] = 1
                target_occs_iou[target_occs_iou == 17] = 0
                
                CalMeanIou_sem._after_step(
                    rearrange(result_dict['sem_pred'],'b f h w d-> (b f) 1 h w d'),
                    rearrange(target_occs,'b f h w d-> (b f) 1 h w d'))
                CalMeanIou_vox._after_step(
                    rearrange(result_dict['iou_pred'],'b f h w d-> (b f) 1 h w d'), 
                    rearrange(target_occs_iou,'b f h w d-> (b f) 1 h w d'))
                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0 and local_rank == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))
                    writer.add_scalar(f'val/loss', loss.item(), global_iter)
                    detailed_loss = []
                    for loss_name, loss_value in loss_dict.items():
                        detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                        writer.add_scalar(f'val/{loss_name}', loss_value, global_iter)
                    detailed_loss = ', '.join(detailed_loss)
                    logger.info(detailed_loss)

                for i, evaluator in enumerate(evaluators):
                    evaluator.update(inputs_dict, outputs_dict)

                # break #debug
        val_miou, _ = CalMeanIou_sem._after_epoch()
        val_iou, _ = CalMeanIou_vox._after_epoch()



        del target_occs, input_occs
        plan_loss = plan_loss/len(val_dataset_loader)
        if plan_loss < best_plan_loss:
            best_plan_loss = plan_loss
        logger.info(f'PlanRegLoss is {plan_loss} while the best plan loss is {best_plan_loss}')
        #logger.info(f'PlanRegLoss is {plan_loss/len(val_dataset_loader)}')
        best_val_iou = val_iou#[max(best_val_iou[i], val_iou[i]) for i in range(len(best_val_iou))]
        best_val_miou = val_miou#[max(best_val_miou[i], val_miou[i]) for i in range(len(best_val_miou))]
        #logger.info(f'PlanRegLoss is {plan_loss/len(val_dataset_loader)}')
        logger.info(f'Current val iou is {val_iou} while the best val iou is {best_val_iou}')
        logger.info(f'Current val miou is {val_miou} while the best val miou is {best_val_miou}')


        # evaluate metrics
        logger.info(f'Evaluate results!')
        for evaluator in evaluators:
            metric_name = evaluator.name
            evaluator.eval()
            logger.info(f'{metric_name}: \n' + evaluator.format_string())


        torch.cuda.empty_cache()


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
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
