import os.path as osp
import pickle
import shutil
import torch
import json
import tempfile
from abc import ABCMeta, abstractmethod

import mmengine
from mmengine.dist import barrier, get_rank, broadcast, is_distributed


class BaseEvaluator(metaclass=ABCMeta):

    def __init__(self, dataset):
        self._len = len(dataset)
        self._classes = dataset.CLASSES
        self.states = {}

    @property 
    def name(self):
        return self._name

    @abstractmethod
    def clean(self):
        pass

    @abstractmethod
    def update(self, inputs_dict, outputs_dict):
        pass

    @abstractmethod
    def eval(self):
        pass

    def format_string(self):
        formatted_str = ''
        for key, value in self.states.items():
            formatted_str += f'{key}: {value}\n'
        return formatted_str
    
    def dump(self, filepath):
        assert filepath.endswith('.json')
        if get_rank() == 0:
            with open(filepath, 'w') as f:
                json.dump(self.states, f)
    
    def gen_broadcasted_tmpdir(self):
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8)
        if get_rank() == 0:
            mmengine.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8)
            dir_tensor[:len(tmpdir)] = tmpdir
        broadcast(dir_tensor, 0)
        return dir_tensor.numpy().tobytes().decode().rstrip()
    
    def broadcast_states(self, states, src_rank):
        if not is_distributed():
            return states

        tmpdir = self.gen_broadcasted_tmpdir()
        if get_rank() == src_rank:
            with open(osp.join(tmpdir, 'metrics.pkl'), 'wb') as f:
                pickle.dump(states, f, protocol=2)

        barrier()
        with open(osp.join(tmpdir, 'metrics.pkl'), 'rb') as f:
            states = pickle.load(f)
        barrier()

        if get_rank() == 0:
            shutil.rmtree(tmpdir)  # type: ignore
        return states
