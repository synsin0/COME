import numpy as np
from torch.utils.data import Dataset
from tabulate import tabulate

from occforecasting.registry import DATASET_WRAPPERS
from mmengine.utils import ProgressBar
from mmengine.logging import MMLogger


@DATASET_WRAPPERS.register_module()
class BaseWrapper(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.load_dataset_static_attr(dataset)
        if hasattr(dataset, 'collate_fn'):
            self.collate_fn = dataset.collate_fn

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def load_dataset_static_attr(cls, dataset):
        if hasattr(dataset, 'CLASSES'):
            cls.CLASSES = dataset.CLASSES
        if hasattr(dataset, 'PALETTE'):
            cls.PALETTE = dataset.PALETTE
        if hasattr(dataset, 'SCENE_RANGE'):
            cls.SCENE_RANGE = dataset.SCENE_RANGE
        if hasattr(dataset, 'VOXEL_SIZE'):
            cls.VOXEL_SIZE = dataset.VOXEL_SIZE

    def __getitem__(self, idx):
        results = self.dataset[idx]
        results['idx'] = idx
        if 'local_idx' not in results:
            results['local_idx'] = idx 
        return results

@DATASET_WRAPPERS.register_module()
class RepeatWrapper(BaseWrapper):

    def __init__(self, times, dataset=None):
        super().__init__(dataset)
        self.times = times
    
    def __len__(self):
        return len(self.dataset) * self.times
    
    def __getitem__(self, idx):
        results = self.dataset[idx % len(self.dataset)]
        results['idx'] = idx
        if 'local_idx' not in results:
            results['local_idx'] = idx % len(self.dataset)
        return results
        

@DATASET_WRAPPERS.register_module()
class BalanceClassWrapper(BaseWrapper):

    def __init__(self, balance_mode='voxel', ratio=0.01, dataset=None):
        super().__init__(dataset)
        assert balance_mode in ['voxel', 'frame']
        self.class_counts = self.get_class_counts()
        self.balance_mode = balance_mode
        if balance_mode == 'voxel':
            self.total_counts = np.stack(self.class_counts).sum(0)
        else:
            self.total_counts = np.stack(
                [counts != 0 for counts in self.class_counts]).sum(0)
        
        tgt_num = self.total_counts[:-1].max() * ratio # ignore the last class of 'free'
        repeat_times = np.floor(tgt_num / self.total_counts).astype(np.int32)
        self.repeat_times = repeat_times.clip(1, None)

        self.balanced_idx = []
        self.balanced_class_counts = []
        for i, counts in enumerate(self.class_counts):
            times = self.repeat_times[counts != 0].max()
            self.balanced_idx.extend([i] * times)
            self.balanced_class_counts.extend([counts] * times)
        if balance_mode == 'voxel':
            self.balanced_total_counts = np.stack(self.balanced_class_counts).sum(0)
        else:
            self.balanced_total_counts = np.stack(
                [counts != 0 for counts in self.balanced_class_counts]).sum(0)

        logger = MMLogger.get_instance('OccForecasting')
        contents = []
        for i, c in enumerate(self.CLASSES[:-1]):
            contents.append(
                [c, self.total_counts[i], self.repeat_times[i], self.balanced_total_counts[i]])
        log_str = tabulate(
            contents, headers=['Class', 'Count', 'Repeat', 'Balanced'], tablefmt='orgtbl')
        logger.info(f'Class counts before balance:\n' + log_str)
    
    def get_class_counts(self):
        if hasattr(self.dataset, 'get_class_counts'):
            return self.dataset.get_class_counts()
        
        logger = MMLogger.get_instance('OccForecasting')
        logger.info('Collecting class counts by iterate total dataset.')
        class_counts = []
        progress_bar = ProgressBar(len(self.dataset))
        for data_dict in self.dataset:
            class_counts.append(np.bincount(
                data_dict['targets'].reshape(-1), minlength=len(self.CLASSES)))
            progress_bar.update()
        return class_counts
        
    def __len__(self):
        return len(self.balanced_idx)
    
    def __getitem__(self, idx):
        results = self.dataset[self.balanced_idx[idx]]
        results['idx'] = idx
        if 'local_idx' not in results:
            results['local_idx'] = self.balanced_idx[idx]
        return results
