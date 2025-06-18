import re
import os
import sys
import time
import glob
import pickle
import argparse
import numpy as np
import os.path as osp
from copy import deepcopy
from collections import OrderedDict
from collections.abc import Sequence, Mapping

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader, RandomSampler

from mmengine import Config
from mmengine.dist import get_rank, is_distributed, broadcast, collect_results
from mmengine.runner import set_random_seed, find_latest_checkpoint
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmengine.registry import PARAM_SCHEDULERS

from occforecasting.registry import MODELS, DATASETS, EVALUATORS, DATASET_WRAPPERS
from occforecasting.utils import LogProcessor, dump_results


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


def init_work_space(cfg_path, cfg):
    if cfg.get('work_dir', None) is None:
        all_folders = cfg_path.split('/')
        exp_name = osp.splitext(all_folders[-1])[0]
        exp_group = '.' if 'configs' not in all_folders else \
            osp.join(*all_folders[all_folders.index('configs')+1:-1])
        cfg.work_dir = osp.join('./work_dirs', exp_group, exp_name, cfg.extra_tag)
    
    if get_rank() == 0:
        os.makedirs(cfg.work_dir, exist_ok=True)


def init_logger(cfg):
    timestamp = torch.tensor(time.time(), dtype=torch.float64)
    broadcast(timestamp)
    cfg.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
    cfg.logger_dir = osp.join(cfg.work_dir, 'train_'+cfg.timestamp)
    if get_rank() == 0:
        os.makedirs(cfg.logger_dir, exist_ok=True)
        cfg.dump(osp.join(cfg.logger_dir, 'config.py'))

    log_cfg = dict(name='OccForecasting', log_level='INFO', file_mode='a',
                   log_file=osp.join(cfg.logger_dir, 'logfile.log'))
    logger = MMLogger.get_instance(**log_cfg)
    log_processor = LogProcessor()
    logger.info(f'Config:\n{cfg.pretty_text}')
    return logger, log_processor


def build_model(cfg, logger):
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


def build_dataloader(cfg):
    cfg = deepcopy(cfg)
    train_dataset = DATASETS.build(cfg.train_dataloader.pop('dataset'))
    val_dataset = DATASETS.build(cfg.val_dataloader.pop('dataset'))

    if 'wrappers' in cfg.train_dataloader:
        wrapper_cfgs = cfg.train_dataloader.pop('wrappers')
        if not isinstance(wrapper_cfgs, list):
            wrapper_cfgs = [wrapper_cfgs]
        for wrapper_cfg in wrapper_cfgs:
            train_dataset = DATASET_WRAPPERS.build(
                wrapper_cfg, default_args=dict(dataset=train_dataset))

    if is_distributed():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = None

    train_batch_size = cfg.train_dataloader.pop('batch_size')
    train_num_workers = cfg.train_dataloader.pop('num_workers')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        sampler=train_sampler,
        collate_fn=None if not hasattr(train_dataset, 'collate_fn') \
            else train_dataset.collate_fn,
        **cfg.train_dataloader)

    val_batch_size = cfg.val_dataloader.pop('batch_size')
    val_num_workers = cfg.val_dataloader.pop('num_workers')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        sampler=val_sampler,
        collate_fn=None if not hasattr(val_dataset, 'collate_fn') \
            else val_dataset.collate_fn,
        **cfg.val_dataloader)
    return train_dataloader, val_dataloader


def build_param_scheduler(cfg, optim_wrapper, train_dataloader):
    scheduler = deepcopy(cfg.param_scheduler)
    schedulers = scheduler if isinstance(scheduler, Sequence) else [scheduler]

    param_schedulers = []
    for _scheduler in schedulers:
        param_schedulers.append(
            PARAM_SCHEDULERS.build(
                _scheduler, default_args=dict(
                    optimizer=optim_wrapper,
                    epoch_length=len(train_dataloader))))
    return param_schedulers


def build_evaluators(cfg, dataloader):
    evaluators = []
    eval_cfgs = cfg.evaluators if isinstance(cfg.evaluators, list) else [cfg.evaluators]
    for eval_cfg in eval_cfgs:
        evaluators.append(
            EVALUATORS.build(eval_cfg, default_args=dict(dataset=dataloader.dataset)))
    return evaluators


def find_latest_checkpoint(work_dir):
    if not osp.exists(work_dir) or not osp.isdir(work_dir):
        return None

    pattern = re.compile(r'epoch_(\d+).pth')
    def get_epoch_idx(x):
        groups = pattern.match(x)
        if groups is None:
            return -1
        return int(groups.group(1))

    ckpt = sorted(os.listdir(work_dir), key=get_epoch_idx)[-1]
    if get_epoch_idx(ckpt) == -1:
        return None
    return osp.join(work_dir, ckpt)


def load_or_resume(cfg, model, optimizer, schedulers, logger):
    ckpt = find_latest_checkpoint(cfg.work_dir) \
        if cfg.resume == 'auto' else cfg.resume
    resume_or_load = 'resume' if cfg.resume_status else 'load'

    epoch = 0
    if ckpt is not None:
        if not osp.exists(ckpt):
            raise FileNotFoundError(f'No such file: {ckpt} to {resume_or_load} from.')
        logger.info(f'{resume_or_load} from {ckpt}')
        ckpt = torch.load(ckpt, map_location='cpu')
        _model = model.module if hasattr(model, 'module') else model
        _model.load_state_dict(ckpt['state_dict'], strict=False)

        if resume_or_load == 'resume':
            optimizer.load_state_dict(ckpt['optimizer'])
            for scheduler, state_dict in zip(schedulers, ckpt['schedulers']):
                scheduler.load_state_dict(state_dict)
            epoch = ckpt['epoch']
    return epoch


def parse_train_outputs(outputs_dict):

    def is_list_of(data, _type):
        if not isinstance(data, list):
            return False
        return all(isinstance(d, _type) for d in data)
    
    log_vars = []
    for loss_name, loss_value in outputs_dict.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars.append([loss_name, loss_value.mean()])
        elif is_list_of(loss_value, torch.Tensor):
            log_vars.append(
                [loss_name, sum(_loss.mean() for _loss in loss_value)])
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(value for key, value in log_vars if 'loss' in key or 'Loss' in key)
    log_vars.insert(0, ['loss', loss])
    return OrderedDict(log_vars)  # type: ignore


def train_one_epoch(cur_epoch, cfg, model, dataloader, logger, log_processor,
                    optimizer, schedulers):
    model.train()
    for batch_idx, inputs_dict in enumerate(dataloader):
        log_processor.before_iter()

        # forward model
        outputs_dict = model(inputs_dict)
        outputs_dict = parse_train_outputs(outputs_dict)

        if torch.isnan(outputs_dict['loss']):
            print("loss is nan, system exits.")
            sys.exit()

        optimizer.update_params(outputs_dict['loss'])

        # update schedulers
        for scheduler in schedulers:
            if not scheduler.by_epoch:
                scheduler.step()
        
        log_processor.after_iter('train', outputs_dict)

        if (batch_idx + 1) % cfg.log_interval == 0 or batch_idx == len(dataloader) - 1:
            log_str = log_processor.format_train_log_str(
                cur_epoch, cfg.max_epoch, dataloader, optimizer, batch_idx)
            logger.info(log_str)
    
    for scheduler in schedulers:
        if scheduler.by_epoch:
            scheduler.step()


def eval_one_epoch(cfg, ckpt_name, model, dataloader, evaluators, logger, log_processor, 
                   best_metric):
    for evaluator in evaluators:
        evaluator.clean()

    model.eval()
    for batch_idx, inputs_dict in enumerate(dataloader):
        log_processor.before_iter()

        with torch.no_grad():
            outputs_dict = model(inputs_dict)
        for evaluator in evaluators:
            evaluator.update(inputs_dict, outputs_dict)

        log_processor.after_iter('val', outputs_dict.get('disp_dict', None))

        if cfg.save_results_after_eval:
            dump_results(osp.join(cfg.logger_dir, ckpt_name+'_results'), outputs_dict)
        
        if (batch_idx + 1) % cfg.log_interval == 0 or batch_idx == len(dataloader) - 1:
            log_str = log_processor.format_test_val_log_str('val', dataloader, batch_idx)
            logger.info(log_str)
    
    # evaluate metrics
    logger.info(f'Evaluate results!')
    metrics = {}
    for evaluator in evaluators:
        metric_name = evaluator.name
        metrics.update(evaluator.eval())
        logger.info(f'{metric_name}:\n' + evaluator.format_string())

    name, mode = cfg.comparison_mode.split(':')
    if name not in metrics:
        logger.warning(f'{name} not in the metrics, Cannot compare.')
        is_best = None
    else:
        is_best = (metrics[name] > best_metric) if mode == '+' else \
            (metrics[name] < best_metric)
        best_metric = metrics[name] if is_best else best_metric

    return best_metric, is_best


def save_ckpt(cfg, model, optimizer, schedulers, epoch, is_best, logger):
    _model = model.module if hasattr(model, 'module') else model
    dict_to_save = {
        'state_dict': _model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'schedulers': [sch.state_dict() for sch in schedulers],
        'epoch': epoch + 1}
    save_file_name = os.path.join(os.path.abspath(cfg.work_dir), f'epoch_{epoch+1}.pth')
    logger.info(f'Save ckpt to {save_file_name}')
    torch.save(dict_to_save, save_file_name)

    if cfg.save_best_ckpt:
        if is_best is None: # is_best is None
            logger.warning('Cannot receive metric comparison results.')
        elif is_best: # is_best == True
            logger.info(f'Save best epoch {epoch+1} ckpt to `best.pth`')
            torch.save(dict_to_save, osp.join(cfg.work_dir, 'best.pth'))
    
    if cfg.max_ckpt_save_num is not None:
        ckpt_list = glob.glob(str(osp.join(osp.abspath(cfg.work_dir), 'epoch_*.pth')))
        ckpt_list.sort(key=osp.getmtime)
        if len(ckpt_list) >= cfg.max_ckpt_save_num:
            for cur_file_idx in range(0, len(ckpt_list) - cfg.max_ckpt_save_num):
                os.remove(ckpt_list[cur_file_idx])            


def main(local_rank, distributed, args):
    # load config
    cfg = Config.fromfile(args.config)
    ignored = ['config', 'enable_dist']
    for key, value in vars(args).items():
        if (key not in ignored) and (key in cfg) and (value is not None):
            setattr(cfg, key, value)

    # init distributed training
    if distributed:
        init_ddp(local_rank)
    
    # init work 
    init_work_space(args.config, cfg)

    # setup logger
    logger, log_processor = init_logger(cfg)

    # set random seed
    set_random_seed(cfg.seed, cfg.get('deterministic', False),
                    cfg.get('diff_rank_seed', False))
    if cfg.seed is not None:
        logger.info(f'set random seed to {cfg.seed}')
    
    # build training modules
    model = build_model(cfg, logger)
    train_dataloader, val_dataloader = build_dataloader(cfg)
    optimizer = build_optim_wrapper(model, cfg.optim_wrapper)
    schedulers = build_param_scheduler(cfg, optimizer, train_dataloader)
    evaluators = build_evaluators(cfg, val_dataloader)

    # load or resume
    start_epoch = load_or_resume(cfg, model, optimizer, schedulers, logger)

    # Save best eval results
    best_metric = -np.inf if cfg.comparison_mode[-1] == '+' else np.inf

    # Start training
    logger.info(f'Start training from epoch {start_epoch}')
    for epoch in range(start_epoch, cfg.max_epoch):
        train_one_epoch(epoch, cfg, model, train_dataloader, logger, 
                        log_processor, optimizer, schedulers)

        is_best = None
        if (epoch + 1) % cfg.val_interval == 0 or epoch == cfg.max_epoch - 1:
            best_metric, is_best = eval_one_epoch(cfg, f'epoch_{epoch+1}', model, val_dataloader,
                                                  evaluators, logger, log_processor, best_metric)
        
        if (epoch + 1) % cfg.save_ckpt_interval == 0 and get_rank() == 0:
            save_ckpt(cfg, model, optimizer, schedulers, epoch, is_best, logger)

        # sync after each epoch
        if distributed:
            dist.barrier()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Occupancy Forecasting Model Training')
    parser.add_argument('config', type=str, help='train config file path')
    parser.add_argument('--enable-dist', action='store_true', help='distributed training')
    parser.add_argument('--work-dir', type=str, help='the dir to save logs and models')
    parser.add_argument('--extra-tag', type=str, help='extra tag for specific exp')
    parser.add_argument('--seed', type=int, help='Random seed')    
    parser.add_argument('--max-ckpt-save-num', type=int, help='max number of saved checkpoint')    
    parser.add_argument('--save-best-ckpt', type=bool, help='save the best checkpoint or not')    
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    if args.enable_dist and ngpus > 1:
        torch.multiprocessing.spawn(main, args=(True, args), nprocs=ngpus)
    else:
        main(0, False, args)
