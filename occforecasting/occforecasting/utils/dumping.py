import os
import torch
import pickle
import numpy as np
import os.path as osp


def dump_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def dump_occs(data, file):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    assert isinstance(data, np.ndarray)

    with open(file, 'wb') as f:
        np.save(f, data)


def dump_results(root, results):
    os.makedirs(root, exist_ok=True)
    for i, meta in enumerate(results['metas']):
        for key, value in results.items():
            if key in ['sem_preds', 'occ_preds']:
                dump_occs(value[i], osp.join(root, f'{meta["idx"]:04d}_{key}.npy'))
