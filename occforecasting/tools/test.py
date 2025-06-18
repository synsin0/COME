import re
import os
import time
import pickle
import argparse
import os.path as osp
from copy import deepcopy
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

from mmengine import Config
from mmengine.dist import is_distributed, collect_results, get_rank, broadcast
from mmengine.logging import MMLogger

from occforecasting.registry import MODELS, DATASETS, EVALUATORS
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


def init_work_space(args, cfg):
    if args.ckpt is not None and osp.isdir(args.ckpt):
        cfg.work_dir = args.ckpt
    elif cfg.get('work_dir', None) is None:
        all_folders = args.config.split('/')
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
    cfg.logger_dir = osp.join(cfg.work_dir, 'test_'+cfg.timestamp)
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
    dataset = DATASETS.build(cfg.test_dataloader.pop('dataset'))
    sampler = None if not is_distributed() else \
        DistributedSampler(dataset, shuffle=False, drop_last=False)
    batch_size = cfg.test_dataloader.pop('batch_size')
    num_workers = cfg.test_dataloader.pop('num_workers')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=None if not hasattr(dataset, 'collate_fn') \
            else dataset.collate_fn,
        **cfg.test_dataloader)


def build_evaluators(cfg, dataloader):
    evaluators = []
    eval_cfgs = cfg.evaluators if isinstance(cfg.evaluators, list) else [cfg.evaluators]
    for eval_cfg in eval_cfgs:
        evaluators.append(
            EVALUATORS.build(eval_cfg, default_args=dict(dataset=dataloader.dataset)))
    return evaluators


def find_pretrained_model(args, cfg):
    pattern = re.compile(r'epoch_(\d+).pth')
    def get_epoch_idx(x):
        groups = pattern.match(osp.basename(x))
        if groups is None:
            return -1
        return int(groups.group(1))

    def list_all_pth(work_dir):
        assert osp.isdir(work_dir), 'input should be a valid folder'
        return [osp.join(work_dir, f) for f in os.listdir(work_dir)
                    if f.endswith('.pth') and not osp.islink(osp.join(work_dir, f))]

    assert not (args.eval_latest and args.eval_best), 'Cannot set both eval_latest and eval_best'
    work_dir = cfg.work_dir if args.ckpt is None else args.ckpt
    if args.eval_best:
        best_checkpoint = osp.join(work_dir, 'best.pth')
        if not osp.exists(best_checkpoint):
            raise FileExistsError(f'Cannot find `best.pth` ckpt from work_dir {work_dir}')
        return [best_checkpoint]
    else:
        if osp.isfile(work_dir) and work_dir.endswith('.pth'):
            return [work_dir]
        ckpts = list_all_pth(work_dir)
        if not ckpts:
            return []
            # raise FileExistsError(f'Cannot find any models in {work_dir}')
        ckpts = sorted(ckpts, key=get_epoch_idx)
        if args.eval_latest:
            ckpts = ckpts[-1:]
        return ckpts


def eval_one_epoch(cfg, ckpt_name, model, dataloader, evaluators, logger,
                   log_processor, save_results):
    for evaluator in evaluators:
        evaluator.clean()

    model.eval()
    for batch_idx, inputs_dict in enumerate(dataloader):
        log_processor.before_iter()

        with torch.no_grad():
            outputs_dict = model(inputs_dict)
        for i, evaluator in enumerate(evaluators):
            evaluator.update(inputs_dict, outputs_dict)

        log_processor.after_iter('test', outputs_dict.get('disp_dict', None))

        # save results
        if save_results:
            dump_results(osp.join(cfg.logger_dir, ckpt_name+'_results'), outputs_dict)
        
        if (batch_idx + 1) % cfg.log_interval == 0 or batch_idx == len(dataloader) - 1:
            log_str = log_processor.format_test_val_log_str('test', dataloader, batch_idx)
            logger.info(log_str)
    
    # evaluate metrics
    logger.info(f'Evaluate results!')
    for evaluator in evaluators:
        metric_name = evaluator.name
        evaluator.eval()
        logger.info(f'{metric_name}: \n' + evaluator.format_string())


def main(local_rank, distributed, args):
    # load config
    cfg = Config.fromfile(args.config)
    if args.extra_tag is not None:
        cfg.extra_tag = args.extra_tag

    # init distributed training
    if distributed:
        init_ddp(local_rank)
    
    # init work 
    init_work_space(args, cfg)

    # setup logger
    logger, log_processor = init_logger(cfg)

    # build training modules
    model = build_model(cfg, logger)
    dataloader = build_dataloader(cfg)
    evaluators = build_evaluators(cfg, dataloader)

    # find all ckpt need to test
    ckpts = find_pretrained_model(args, cfg)
    if len(ckpts) == 0:
        ckpt_name = 'no_training'
        eval_one_epoch(cfg, ckpt_name, model, dataloader, evaluators, logger, log_processor,
                       args.save_results)     
    # Start testing
    for ckpt in ckpts:
        ckpt_dict = torch.load(ckpt, map_location='cpu')
        _model = model.module if hasattr(model, 'module') else model
        _model.load_state_dict(ckpt_dict['state_dict'], strict=False)

        logger.info(f'---------Start testing ckpt {ckpt}----------')
        ckpt_name = osp.splitext(osp.basename(ckpt))[0]
        eval_one_epoch(cfg, ckpt_name, model, dataloader, evaluators, logger, log_processor,
                       args.save_results)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Occupancy Forecasting Model Testing')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--enable-dist', action='store_true', help='distributed testing')
    parser.add_argument('--extra-tag', type=str, help='extra tag for specific exp')
    parser.add_argument('--ckpt', type=str, help='the ckpt file or dir to find models')
    parser.add_argument('--eval-latest', action='store_true', help='only eval latest ckpt.')
    parser.add_argument('--eval-best', action='store_true', help='only eval best ckpt.')
    parser.add_argument('--save-results', action='store_true', help='Save outputs')
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    if args.enable_dist and ngpus > 1:
        torch.multiprocessing.spawn(main, args=(True, args), nprocs=ngpus)
    else:
        main(0, False, args)