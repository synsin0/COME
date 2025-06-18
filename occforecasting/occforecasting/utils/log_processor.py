import time
import datetime
import numpy as np
from collections import defaultdict

from mmengine.dist import is_distributed, all_reduce

import torch

class LogProcessor:

    def __init__(self, max_log_length=10000):
        self.train_interval = 0
        self.val_interval = 0
        self.test_interval = 0
        self.max_log_length = max_log_length

        empty_ndarray = lambda: np.empty([0])
        self._train_log_history = defaultdict(empty_ndarray)
        self._val_log_history = defaultdict(empty_ndarray)
        self._test_log_history = defaultdict(empty_ndarray)

    def _select_log_history(self, phase):
        assert phase in ['train', 'val', 'test']
        if phase == 'train':
            log_history = self._train_log_history
        elif phase == 'val':
            log_history = self._val_log_history
        else:
            log_history = self._test_log_history
        return log_history

    def update_log_hitory(self, phase, key, value):
        # validate value
        if isinstance(value, torch.Tensor):
            if is_distributed():
                all_reduce(value, 'mean')
            value = value.detach().cpu().numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array([value], dtype=np.float32)
        
        if len(value.shape) == 0:
            value = value[None]

        # record history
        log_history = self._select_log_history(phase)
        log_value = log_history[key]
        log_value = np.concatenate([log_value, value])
        if log_value.size > self.max_log_length:
            log_value = log_value[-self.max_log_length:]
        log_history[key] = log_value

    def get_log_hitory(self, phase, key, window_size=1, reduction='mean'):
        log_history = self._select_log_history(phase)
        log_value = log_history[key]
        if log_value.size == 0:
            raise RuntimeError(f'{key} has not been logged in history')

        if reduction == 'mean':
            window_size = min(window_size, log_value.size)
            log_value = log_value[-window_size:]
            log_value = log_value.mean().item()
        elif reduction == 'current':
            log_value = log_value[-1].item()
        else:
            raise KeyError('Only support "mean" and "current" reduction, but '
                           f'got {reduction}')
        return log_value

    def format_train_log_str(self, 
                             cur_epoch,
                             max_epoch,
                             dataloader,
                             optimizer,
                             batch_idx):
        log_str_list = []

        # epoch iteration number information
        log_str = '(train) '
        iter_per_epoch = len(dataloader)
        # get the max length of iteration and epoch number
        epoch_len = len(str(max_epoch))
        iter_len = len(str(iter_per_epoch))
        # right just the length
        cur_epoch_str = str(cur_epoch + 1).rjust(epoch_len)
        cur_iter_str = str(batch_idx + 1).rjust(iter_len)
        log_str += f'[{cur_epoch_str}/{max_epoch}]'
        log_str += f'[{cur_iter_str}/{iter_per_epoch}]'
        log_str_list.append(log_str)

        # iter time and etc time
        iter_time = self.get_log_hitory(
            'train', 'time', 1000, reduction='mean')
        past_iter = iter_per_epoch * cur_epoch + batch_idx
        total_iter = iter_per_epoch * max_epoch
        eta_time = iter_time * (total_iter - past_iter)
        eta_time = datetime.timedelta(seconds=int(eta_time))
        mm, ss = divmod(eta_time.seconds, 60)
        hh, mm = divmod(mm, 60)
        format_eta_time = f'{eta_time.days:01d} day {hh:02d}:{mm:02d}:{ss:02d}'
        log_str_list.extend(
            [f'eta: {format_eta_time}', f'time: {iter_time:.4f}'])

        # leanring rate information
        lr = optimizer.get_lr()['lr'][0]
        log_str_list.append(f'lr: {lr:.4e}')

        # other information
        for key in self._train_log_history.keys():
            if key == 'time':
                continue
            if 'loss' in key or 'acc' in key:
                log_value = self.get_log_hitory(
                    'train', key, self.train_interval, reduction='mean')
            else:
                log_value = self.get_log_hitory(
                    'train', key, self.train_interval, reduction='current')
            log_str_list.append(f'{key}: {log_value:.4f}')
        
        # reset train interval
        self.train_interval = 0

        log_str = '  '.join(log_str_list)
        return log_str

    def format_test_val_log_str(self, phase, dataloader, batch_idx):
        assert phase in ['val', 'test']
        log_str_list = []

        # epoch iteration number information
        log_str = '(val) [1/1]' if phase == 'val' else '(test) [1/1]'
        iter_per_epoch = len(dataloader)
        # get the max length of iteration and epoch number
        iter_len = len(str(iter_per_epoch))
        # right just the length
        cur_iter = str(batch_idx + 1).rjust(iter_len)
        log_str += f'[{cur_iter}/{iter_per_epoch}]'
        log_str_list.append(log_str)

        # iter time and etc time
        iter_time = self.get_log_hitory(phase, 'time', 1000, reduction='mean')
        eta_time = iter_time * (iter_per_epoch - batch_idx)
        eta_time = datetime.timedelta(seconds=int(eta_time))
        mm, ss = divmod(eta_time.seconds, 60)
        hh, mm = divmod(mm, 60)
        format_eta_time = f'{eta_time.days:01d} day {hh:02d}:{mm:02d}:{ss:02d}'
        log_str_list.extend(
            [f'eta: {format_eta_time}', f'time: {iter_time:.4f}'])

        # other information
        interval = self.val_interval if phase == 'val' else self.test_interval
        for key in self._select_log_history(phase).keys():
            if key == 'time':
                continue
            log_value = self.get_log_hitory(
                phase, key, interval, reduction='current')
            log_str_list.append(f'{key}: {log_value:.4f}')
        
        # reset interval
        if phase == 'val':
            self.val_interval = 0
        else:
            self.test_interval = 0

        log_str = '  '.join(log_str_list)
        return log_str
    
    def before_iter(self):
        self.iter_start_time = time.time()

    def after_iter(self, phase, outputs=None):
        assert phase in ['train', 'val', 'test']

        if outputs is not None:
            for key, value in outputs.items():
                self.update_log_hitory(phase, key, value)

        iter_time = time.time() - self.iter_start_time
        self.update_log_hitory(phase, 'time', iter_time)

        if phase == 'train':
            self.train_interval += 1
        elif phase == 'val':
            self.val_interval += 1
        else:
            self.test_interval += 1



