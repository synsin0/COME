import mayavi.mlab as mlab
import os
import cv2
import os.path as osp
import mayavi
import numpy as np
import argparse
from occforecasting.registry import DATASETS
from mmengine import Config
from collections import namedtuple


Camera = namedtuple(
    'Camera', ['position', 'focal_point', 'view_up', 'view_angle', 'clipping_range'])
Cameras = {
    'back': Camera(
        position=[0.0, -80.0, 80.0],
        focal_point=[0.0, 0.0, -1.0],
        view_up=[0.0, 0.0, 1.0],
        view_angle=30.0,
        clipping_range=[0.01, 190.0]
    ),
    'top': Camera(
        position=[0.0, -0.01, 100.0],
        focal_point=[0.0, 0.0, -1.0],
        view_up=[0.0, 0.0, 1.0],
        view_angle=45.0,
        clipping_range=[0.01, 190.0]
    )
}


def visualize_occ(x, y, z, labels, palette, voxel_size, camera, classes):
    if palette.shape[1] == 3:
        palette = np.concatenate([palette, np.ones((palette.shape[0], 1)) * 255], axis=1)
    
    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    plot = mlab.points3d(x, y, z, 
                         labels,
                         scale_factor=voxel_size[0],
                         mode="cube",
                         scale_mode = "vector",
                         opacity=1.0,
                         vmin=1.0,
                         vmax=len(classes)-1)
    plot.module_manager.scalar_lut_manager.lut.table = palette

    assert camera in Cameras
    fig.scene.camera.position = Cameras[camera].position
    fig.scene.camera.focal_point = Cameras[camera].focal_point
    fig.scene.camera.view_up = Cameras[camera].view_up
    fig.scene.camera.view_angle = Cameras[camera].view_angle
    fig.scene.camera.clipping_range = Cameras[camera].clipping_range
    fig.scene.camera.compute_view_plane_normal()
    fig.scene.render()
    f = mlab.gcf()
    f.scene._lift()
    save_fig = mlab.screenshot()
    mlab.close()

    return save_fig


def main(args):
    cfg = Config.fromfile(args.config)
    if args.phase == 'train':
        dataset_cfg = cfg.train_dataloader.dataset
        wrapper_cfg = cfg.train_dataloader.pop('wrappers', None)
    elif args.phase == 'val':
        dataset_cfg = cfg.val_dataloader.dataset
        wrapper_cfg = cfg.val_dataloader.pop('wrappers', None)
    else:
        dataset_cfg = cfg.test_dataloader.dataset
        wrapper_cfg = cfg.test_dataloader.pop('wrappers', None)
    dataset = DATASETS.build(dataset_cfg)
    if wrapper_cfg is not None:
        wrapper_cfg = wrapper_cfg if isinstance(wrapper_cfg, list) else [wrapper_cfg]
        for cfg in wrapper_cfg:
            dataset = DATASET_WRAPPERS.build(cfg, default_args=dict(dataset=dataset))

    classes = dataset.CLASSES
    palette = np.array(dataset.PALETTE)
    scene_range = dataset.SCENE_RANGE
    voxel_size = dataset.VOXEL_SIZE
    free = len(classes) - 1
    W = int((scene_range[3] - scene_range[0]) / voxel_size[0])
    H = int((scene_range[4] - scene_range[1]) / voxel_size[1])
    Z = int((scene_range[5] - scene_range[2]) / voxel_size[2])

    x = (np.arange(0, W) + 0.5) * voxel_size[0] + scene_range[0]
    y = (np.arange(0, H) + 0.5) * voxel_size[1] + scene_range[1]
    z = (np.arange(0, Z) + 0.5) * voxel_size[2] + scene_range[2]
    xx = x[None, None, :].repeat(Z, axis=0).repeat(H, axis=1)
    yy = y[None, :, None].repeat(Z, axis=0).repeat(W, axis=2)
    zz = z[:, None, None].repeat(H, axis=1).repeat(W, axis=2)

    if osp.isfile(args.result_path):
        save_path = osp.split(args.result_path)[0]
        results = [args.result_path]
    elif osp.isdir(args.result_path):
        save_path = args.result_path
        results = [osp.join(args.result_path, f) for f in os.listdir(args.result_path) \
                       if f.endswith('sem_preds.npy')]
    else:
        raise ValueError('Invalid result path.')
    
    if args.save_path is not None:
        save_path = args.save_path
    
    for result in results:
        vedio_writer = cv2.VideoWriter(
            osp.join(save_path, f'{osp.basename(result)}_vis.mp4'), 
            cv2.VideoWriter_fourcc(*'mp4v'), 2, (1000, 948))
        
        if args.show_input:
            idx = int(osp.basename(result).split('_')[0])
            source_occs = dataset[idx]['source_occs']
            for scene in source_occs:
                x, y, z = xx[scene != free], yy[scene != free], zz[scene != free]
                labels = scene[scene != free]
                vedio_writer.write(visualize_occ(
                    x, y, z, 
                    labels, 
                    palette,
                    voxel_size,
                    args.camera,
                    classes)[..., ::-1])
        
        occ_preds = np.load(result)
        for scene in occ_preds:
            x, y, z = xx[scene != free], yy[scene != free], zz[scene != free]
            labels = scene[scene != free]
            vedio_writer.write(visualize_occ(
                x, y, z,
                labels,
                palette,
                voxel_size,
                args.camera,
                classes)[..., ::-1])

        vedio_writer.release()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path.')
    parser.add_argument('--result-path', type=str, help='Results file or path.')
    parser.add_argument('--phase', type=str, default='test', 
                        choices=['train', 'val', 'test'], help='Dataset phase.')
    parser.add_argument('--show-input', action='store_true', help='Visualize input.')
    parser.add_argument('--save-path', type=str, help='Save path.')
    parser.add_argument('--camera', type=str, default='top', help='Camera type.')
    args = parser.parse_args()
    
    main(args)
