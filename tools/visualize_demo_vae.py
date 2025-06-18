from pyvirtualdisplay import Display
display = Display(visible=False, size=(2560, 1440))
display.start()
import pdb
from mayavi import mlab
import mayavi
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import time, argparse, os.path as osp, os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
import cv2
from vis_gif import create_mp4
import warnings
warnings.filterwarnings("ignore")
from vis_utils import draw



def main(args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    
    os.makedirs(args.work_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.dir_name=f'{args.dir_name}_{timestamp}'

    log_file = osp.join(args.work_dir, f'{cfg.get("data_type", "gts")}_visualize_{timestamp}.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    my_model = MODELS.build(cfg.model)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    my_model = my_model.cuda()
    raw_model = my_model
    logger.info('done ddp model')
    from dataset import get_dataloader
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False)
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
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    epoch = 'last'
    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        epoch = ckpt['epoch']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    # eval
    my_model.eval()
    os.environ['eval'] = 'true'
    recon_dir = os.path.join(args.work_dir, args.dir_name)
    os.makedirs(recon_dir, exist_ok=True)
    with torch.no_grad():
        for i_iter_val, (input_occs, _, metas) in enumerate(val_dataset_loader):
            if i_iter_val not in args.scene_idx:
                continue
            if i_iter_val > max(args.scene_idx):
                break
            input_occs = input_occs.cuda() #torch.Size([1, 16, 200, 200, 16])
            result = my_model(x=input_occs, metas=metas)
            start_frame=cfg.get('start_frame', 0)
            #end_frame=cfg.get('end_frame', 11)
            end_frame=input_occs.shape[1]
            logits = result['logits'] #torch.Size([1, 6, 200, 200, 16, 18])
            dst_dir = os.path.join(recon_dir, str(i_iter_val),'pred')
            input_dir = os.path.join(recon_dir, f'{i_iter_val}','input')
            cmp_dir = os.path.join(recon_dir, f'{i_iter_val}','cmp')
            # input_occs = result['input_occs']
            os.makedirs(dst_dir, exist_ok=True)
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(cmp_dir, exist_ok=True)
            all_pred=[]
            for frame in range(start_frame,end_frame):
                tt=str(i_iter_val) + '_' + str(frame)
                input_occ = input_occs[:, frame, ...].squeeze().cpu().numpy()
                dst_input=draw(input_occ, 
                    None, # predict_pts,
                    [-40, -40, -1], 
                    [0.4] * 3, 
                    None, #  grid.squeeze(0).cpu().numpy(), 
                    None,#  pt_label.squeeze(-1),
                    input_dir,#recon_dir,
                    None, # img_metas[0]['cam_positions'],
                    None, # img_metas[0]['focal_positions'],
                    timestamp=tt,
                    mode=0,
                    sem=False)
                im=mmcv.imread(dst_input)
                cv2.putText(im, f'GT_{frame:02d}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
                mmcv.imwrite(im,dst_input)
                if True:
                    logit = logits[:, frame, ...]
                    pred = logit.argmax(dim=-1).squeeze().cpu().numpy() # 1, 1, 200, 200, 16
                    all_pred.append((pred))
                    
                    dst_wm=draw(pred, 
                        None, # predict_pts,
                        [-40, -40, -1], 
                        [0.4] * 3, 
                        None, #  grid.squeeze(0).cpu().numpy(), 
                        None,#  pt_label.squeeze(-1),
                        dst_dir,#recon_dir,
                        None, # img_metas[0]['cam_positions'],
                        None, # img_metas[0]['focal_positions'],
                        timestamp=tt,
                        mode=0,
                        sem=False)
                    im=mmcv.imread(dst_wm)
                    cv2.putText(im, f'predict_{frame:02d}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
                    mmcv.imwrite(im,dst_wm)
                                # concat 2 img


                im1=mmcv.imread(dst_input)#[:,550:-530,:]
                im2=mmcv.imread(dst_wm)#[:,550:-530,:]
                mmcv.imwrite(np.concatenate([im1, im2], axis=1), os.path.join(cmp_dir, f'vis_{tt}.png'))
            logger.info('[EVAL] Iter %5d / %5d'%(i_iter_val, len(val_dataset_loader)))
            # create_mp4(dst_dir)
            # create_mp4(cmp_dir)
            if args.export_pcd:
                from vis_utils import visualize_point_cloud

                abs_pose=metas[0]['e2g_t']
                abs_rot=metas[0]['e2g_r']
                n_gt=min(len(all_pred),len(abs_pose))
                visualize_point_cloud(all_pred[:n_gt],abs_pose=abs_pose[:n_gt],abs_rot=abs_rot[:n_gt],cmp_dir=dst_dir,key='pred')


            # break #debug

if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='configs/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--dir-name', type=str, default='vis')
    parser.add_argument('--seed', type=int, default=42)#,1023,333,256])
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--frame-idx', nargs='+', type=int, default=[0, 10])
    parser.add_argument('--scene-idx', nargs='+', type=int, default=[6])#,7,16,18,19,87,89,96,101])
    parser.add_argument('--export_pcd', action='store_true', default=False, help='Enable pose (default is True)')

    args = parser.parse_args()

    ngpus = 1
    args.gpus = ngpus
    print(args)
    main(args)

