from pyvirtualdisplay import Display
display = Display(visible=False, size=(2560, 1440))
display.start()
from mayavi import mlab
import mayavi
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import pdb
import time, argparse, os.path as osp, os
import torch, numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
import cv2
from vis_gif import create_mp4
import warnings
warnings.filterwarnings("ignore")
from einops import rearrange
from diffusion import create_diffusion
from vis_utils import draw
from pyquaternion import Quaternion
from utils.misc import downsample_visible_mask

from visualize_nuscenes_occupancy import draw_nusc_occupancy_Bev_Front


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
    # args.dir_name=f'{args.dir_name}_{timestamp}'

    log_file = osp.join(args.work_dir, f'{cfg.get("data_type", "gts")}_visualize_{timestamp}.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    my_model = MODELS.build(cfg.model.world_model)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    my_model = my_model.cuda()
    raw_model = my_model
    vae=MODELS.build(cfg.model.vae).cuda()

    vae.requires_grad_(False)
    vae.eval()

    stage1_cfg = Config.fromfile(cfg.stage_one_config)
    stage1_model = MODELS.build(stage1_cfg.model)
    # load stage1 model
    assert cfg.stage_one_ckpt and osp.exists(cfg.stage_one_ckpt)
    _stage1_model = stage1_model.module if hasattr(stage1_model, 'module') else stage1_model
    ckpt = torch.load(cfg.stage_one_ckpt, map_location='cpu')
    logger.info(_stage1_model.load_state_dict(ckpt['state_dict'], strict=True))
    stage1_model.requires_grad_(False)
    stage1_model.eval()
    stage1_model = stage1_model.cuda()

    wm_cfg = Config.fromfile(cfg.world_model_config)
    world_model = MODELS.build(wm_cfg.model.world_model)
    world_model = world_model.eval()
    world_model.cuda()


    logger.info('done ddp model')
    from dataset import get_dataloader
    cfg.val_dataset_config.test_mode=True
    cfg.val_loader.num_workers=0
    cfg.train_loader.num_workers=0
    
    # cfg.val_dataset_config.new_rel_pose=False ## TODO
    # cfg.train_dataset_config.test_index_offset=args.test_index_offset
    cfg.val_dataset_config.test_index_offset=args.test_index_offset
    if args.return_len is not None: 
        cfg.train_dataset_config.return_len=max(cfg.train_dataset_config.return_len,args.return_len)
        cfg.val_dataset_config.return_len=max(cfg.val_dataset_config.return_len,args.return_len)
        # cfg.val_dataset_config.return_len=60

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False)
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'best_miou_controlnet.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'best_miou_controlnet.pth')
    else:
        ckpts=[i for i in os.listdir(args.work_dir) if 
            i.endswith('.pth') and i.replace('.pth','').replace('epoch_','').isdigit()]
        if len(ckpts)>0:
            ckpts.sort(key=lambda x:int(x.replace('.pth','').replace('epoch_','')))
            cfg.resume_from = osp.join(args.work_dir, ckpts[-1])

    if osp.exists(osp.join(args.work_dir, 'best_miou_world_model.pth')):
        world_model_ckpt = osp.join(args.work_dir, 'best_miou_world_model.pth')
        ckpt = torch.load(world_model_ckpt, map_location='cpu')
        load_key='state_dict' 
        logger.info(world_model.load_state_dict(ckpt[load_key], strict=False))
    else:
        ckpt = torch.load(cfg.world_model_ckpt, map_location='cpu')
        load_key='state_dict' if not cfg.get('ema',False) else 'ema'
        logger.info(world_model.load_state_dict(ckpt[load_key], strict=False))


    if args.resume_from:
        cfg.resume_from = args.resume_from
    if args.vae_resume_from:
        cfg.vae_load_from=args.vae_resume_from
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('vae resume from: ' + cfg.vae_load_from)
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
    print(vae.load_state_dict(torch.load(cfg.vae_load_from)['state_dict']))
        
    # eval
    my_model.eval()
    eval_model = world_model
    controlnet = my_model 
    # if not args.ema else ema
    os.environ['eval'] = 'true'
    recon_dir = os.path.join(args.work_dir, args.dir_name)
    os.makedirs(recon_dir, exist_ok=True)
    os.environ['recon_dir']=recon_dir

    diffusion = create_diffusion(
        # timestep_respacing=str(cfg.sample.num_sampling_steps),
        timestep_respacing=str(args.num_sampling_steps),
        beta_start=cfg.schedule.beta_start,
        beta_end=cfg.schedule.beta_end,
        replace_cond_frames=cfg.replace_cond_frames,
        cond_frames_choices=cfg.cond_frames_choices,
        predict_xstart=cfg.schedule.get('predict_xstart',False),
    )
    if args.pose_control:
        cfg.sample.n_conds=1
    print(len(val_dataset_loader))
    with torch.no_grad():
        for i_iter_val, (input_occs, _, metas) in enumerate(val_dataset_loader):
            # if i_iter_val not in args.scene_idx:
            #     continue
            # if i_iter_val > max(args.scene_idx):
            #     break
            if str(i_iter_val) in os.listdir(recon_dir):
                import glob
                pattern = os.path.join(recon_dir, str(i_iter_val), 'pred', "vis_*.png")
                png_files = glob.glob(pattern)
                count = len(png_files)
                if count == cfg.get('end_frame', 10) - cfg.get('start_frame', 0):
                    print(f'skip iter {i_iter_val} as all images are drawn.')
                    continue
 
            start_frame=cfg.get('start_frame', 0)
            mid_frame=cfg.get('mid_frame', 3)
            # end_frame=cfg.get('end_frame', 9)
            end_frame=input_occs.shape[1] if args.end_frame is None else args.end_frame

            if args.pose_control:
                # start_frame=0
                mid_frame=1
                # end_frame=10
            assert cfg.sample.n_conds==mid_frame




            # __import__('ipdb').set_trace()

            SCENE_RANGE = (-40, -40, -1, 40, 40, 5.4)

            inputs_dict = dict()
            inputs_dict["source_occs"] = input_occs[:,start_frame: mid_frame].clone().permute(0,1,4,3,2)
            inputs_dict["target_occs"] = input_occs[:,mid_frame: end_frame].clone().permute(0,1,4,3,2)
            inputs_dict["source_metas"] = dict()
            inputs_dict["target_metas"] = dict()

            inputs_dict["source_metas"] = []
            inputs_dict["target_metas"] = []
            inputs_dict["metas"] = []

            for bs in range(len(metas)):
                inputs_dict["source_metas"].append(dict(ego2global=[]))
                inputs_dict["target_metas"].append(dict(ego2global=[]))
                inputs_dict["source_metas"]
                inputs_dict["target_metas"]
                inputs_dict["metas"].append(dict())
                inputs_dict["metas"][bs]['scene_range'] = SCENE_RANGE
                for frame_idx in range(end_frame - start_frame):
                    e2g_t = metas[bs]['e2g_t'][frame_idx]
                    e2g_r = metas[bs]['e2g_r'][frame_idx]
                    ego2global = np.eye(4)
                    ego2global[:3,:3] = Quaternion(e2g_r).rotation_matrix
                    ego2global[:3, 3] = e2g_t
                    if frame_idx < mid_frame:
                        inputs_dict["source_metas"][bs]["ego2global"].append(ego2global)
                    else:
                        inputs_dict["target_metas"][bs]["ego2global"].append(ego2global)
                    
            with torch.no_grad():
                stage1_outputs_dict = stage1_model(inputs_dict)


            future_occs_pred = stage1_outputs_dict["sem_preds"].clone().permute(0,1,4,3,2)


            input_occs = input_occs.cuda() #torch.Size([1, 16, 200, 200, 16])
            input_occs[:,mid_frame:end_frame] = future_occs_pred


            bs,f,_,_,_=input_occs.shape
            encoded_latent, shape=vae.forward_encoder(input_occs)
            encoded_latent,_,_=vae.sample_z(encoded_latent) #bchw
            # encoded_latent = self.vae.vqvae.quant_conv(encoded_latent)
            # encoded_latent, _,_ = vae.vqvae(encoded_latent, is_voxel=False)
            input_latents=encoded_latent*cfg.model.vae.scaling_factor
            if input_latents.dim()==4:
                input_latents = rearrange(input_latents, '(b f) c h w -> b f c h w', b=bs).contiguous()
            elif input_latents.dim()==5:
                input_latents = rearrange(input_latents, 'b c f h w -> b f c h w', b=bs).contiguous()
            else:
                raise NotImplementedError
            

            # from debug_vis import visualize_tensor_pca
            # TODO fix dim bug torch.Size([1, 64, 12, 25, 25])
            # visualize_tensor_pca(encoded_latent.permute(0,2,3,1).cpu(), save_dir=recon_dir+'/debug_feature', filename=f'vis_vae_encode_{i_iter_val}.png')
            os.environ.update({'i_iter_val': str(i_iter_val)})
            os.environ.update({'recon_dir': str(recon_dir)})
            # rencon_occs=vae.forward_decoder(encoded_latent, shape, input_occs.shape)

            # gaussian diffusion  pipeline
            w=h=cfg.model.vae.encoder_cfg.resolution
            vae_scale_factor = 2 ** (len(cfg.model.vae.encoder_cfg.ch_mult) - 1)
            vae_docoder_shapes=cfg.shapes[:len(cfg.model.vae.encoder_cfg.ch_mult) - 1]
            w//=vae_scale_factor
            h//=vae_scale_factor

            stage1_invisible_mask = stage1_outputs_dict['invisible_mask'].permute(0,1,4,3,2)  # True: Invisible; False: Visible
            # stage1_invisible_mask_downsampled = downsample_visible_mask(stage1_invisible_mask)
            # # stage1_invisible_mask_downsampled_v2 = downsample_visible_mask_v2(stage1_invisible_mask)
            # stage1_invisible_mask_downsampled = stage1_invisible_mask_downsampled.unsqueeze(2).repeat(1,1,input_latents.shape[2],1,1)
            # stage1_invisible_mask_downsampled_past = torch.zeros([stage1_invisible_mask_downsampled.shape[0], mid_frame-start_frame, stage1_invisible_mask_downsampled.shape[2], stage1_invisible_mask_downsampled.shape[3], stage1_invisible_mask_downsampled.shape[4]], dtype=torch.bool, device=stage1_invisible_mask_downsampled.device)
            # stage1_invisible_mask_downsampled = torch.cat([stage1_invisible_mask_downsampled_past, stage1_invisible_mask_downsampled], dim=1)
            # input_latents[stage1_invisible_mask_downsampled] = 0
            model_kwargs=dict(
                condition = input_latents,
                metas=metas,
                invisible_mask=stage1_invisible_mask,
            #     # cfg_scale=cfg.sample.guidance_scale
                # metas=metas
            )
            if 'data_bev' in metas[0].keys():
                data_bev = torch.stack([meta['data_bev'] for meta in metas])
                model_kwargs['bev_layout'] = data_bev.cuda()
            if 'data_bev_ori' in metas[0].keys():
                bev_ori = torch.stack([meta['data_bev_ori'] for meta in metas])

            if args.pose or args.pose_control:
                # assert False #debug pure gen
                model_kwargs['metas']=metas
            noise_shape=(bs, end_frame,cfg.base_channel, w,h,)
            initial_cond_indices=None
            n_conds=cfg.sample.get('n_conds',0)
            if n_conds:
                initial_cond_indices=[index for index in range(n_conds)]
            
            # Sample images:
            if cfg.sample.sample_method == 'ddim':
                latents = diffusion.ddim_sample_loop(
                    eval_model,  noise_shape, None, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device='cuda'
                )
            elif cfg.sample.sample_method == 'ddpm':
                if args.rolling_sampling_n<2:

                    latents = diffusion.p_sample_loop(
                        eval_model,  noise_shape, None, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device='cuda',
                        initial_cond_indices=initial_cond_indices,
                        initial_cond_frames=input_latents, controlnet=controlnet,
                    )
                else:
                    latents=diffusion.p_sample_loop_cond_rollout(
                        eval_model,  noise_shape, None, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device='cuda',
                        # initial_cond_indices=initial_cond_indices,
                        input_latents=input_latents,
                        rolling_sampling_n=args.rolling_sampling_n,
                        n_conds=n_conds,
                        n_conds_roll=args.n_conds_roll,
                        controlnet=controlnet,
                    )
                    end_frame=latents.shape[1]
            latents = 1 / cfg.model.vae.scaling_factor * latents

            if cfg.model.vae.decoder_cfg.type=='Decoder3D':
                latents = rearrange(latents,'b f c h w-> b c f h w')
            else:
                # assert False #debug
                latents = rearrange(latents,'b f c h w -> (b f) c h w')

            logits = vae.forward_decoder(
                latents , shapes=vae_docoder_shapes,input_shape=[bs,end_frame,*cfg.shapes[0],cfg._dim_]
            )
            dst_dir = os.path.join(recon_dir, str(i_iter_val),'pred')
            input_dir = os.path.join(recon_dir, f'{i_iter_val}','input')
            # input_occs = result['input_occs']
            os.makedirs(dst_dir, exist_ok=True)
            os.makedirs(input_dir, exist_ok=True)


            if True:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.plot(metas[0]['rel_poses'][:,0],metas[0]['rel_poses'][:,1],marker='o',alpha=0.5)
                plt.savefig(os.path.join(dst_dir, f'pose.png'))
                plt.clf()
                # for i, xyz in enumerate(e2g_t):
                #     xy=xyz[:2]
                #     gt_mode=gt_modes[i].astype('int').tolist().index(1)
                #     ax2.annotate(f"{i+1}({gt_mode})", xy=xy, textcoords="offset points", xytext=(0,10), ha='center')
                # ax2.set_title('ego2global_translation (xy) (idx+gt_mode)')

                plt.plot(metas[0]['e2g_rel0_t'][:,0],metas[0]['e2g_rel0_t'][:,1])
                plt.scatter([0],[0],c='r')

                plt.annotate(f"start", xy=(0,0), textcoords="offset points", xytext=(0,10),ha='center') 


                plt.savefig(os.path.join(dst_dir, f'pose_w.png'))
            # exit(0)
            all_pred=[]
            for frame in range(start_frame,end_frame):
            # for frame in range(0,end_frame):
                # if frame >15  and frame%10!=0:
                    # continue
                # tt=str(i_iter_val) + '_' + str(frame)
                tt=str(i_iter_val) + '_' + str(frame+args.test_index_offset)
                # if frame < rencon_occs.shape[1]:
                    # input_occ = rencon_occs[:, frame, ...].argmax(-1).squeeze().cpu().numpy()
                if frame < input_occs.shape[1] and not args.skip_gt:
                # if True:
                    input_occ = input_occs[:, frame, ...].squeeze().cpu().numpy()

                    draw(input_occ, 
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
                        sem=False,
                        show_ego=args.show_ego)
                if True:
                # if frame>=mid_frame:
                    logit = logits[:, frame, ...]
                    pred = logit.argmax(dim=-1).squeeze().cpu().numpy() # 1, 1, 200, 200, 16
                    all_pred.append((pred))

                    # all_pred.append((pred))
                    draw(pred, 
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
                        sem=False,
                        show_ego=args.show_ego)
                    

                    if 'data_bev_ori' in metas[0].keys():
                        bev_layout = bev_ori[0][frame].cpu().numpy()
                        cat_save_file = os.path.join(dst_dir, "cat_vis_{}.png".format(frame))

                        cat_image = draw_nusc_occupancy_Bev_Front(
                            voxels=pred,
                            vox_origin=np.array([-40, -40, -1]),
                            voxel_size=np.array([0.4, 0.4, 0.4]),
                            grid=np.array([200, 200, 16]),
                            pred_lidarseg=None,
                            target_lidarseg=None,
                            save_folder=dst_dir,
                            cat_save_file=cat_save_file,
                            cam_positions=None,
                            focal_positions=None,
                            bev_layout=bev_layout,
                        )
            logger.info('[EVAL] Iter %5d / %5d'%(i_iter_val, len(val_dataset_loader)))
            # create_mp4(dst_dir)
            # create_mp4(cmp_dir)
            if args.export_pcd:
                from vis_utils import visualize_point_cloud

                abs_pose=metas[0]['e2g_t']
                abs_rot=metas[0]['e2g_r']
                n_gt=min(len(all_pred),len(abs_pose))
                visualize_point_cloud(all_pred[:n_gt],abs_pose=abs_pose[:n_gt],abs_rot=abs_rot[:n_gt],cmp_dir=dst_dir,key='pred')


if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='configs/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--vae-resume-from', type=str, default='')
    parser.add_argument('--dir-name', type=str, default='vis')
    parser.add_argument('--num_sampling_steps', type=int, default=20)
    parser.add_argument('--seed', type=int,  default=42)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--n_conds_roll', type=int, default=None)
    parser.add_argument('--return_len', type=int, default=None)
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--frame-idx', nargs='+', type=int, default=[0, 10])
    #########################################
    parser.add_argument('--scene-idx', nargs='+', type=int, default=[6,7,16,18,19,87,89,96,101])
    # parser.add_argument('--scene-idx', nargs='+', type=int, default=[6,7])
    parser.add_argument('--rolling_sampling_n', type=int, default=1)
    parser.add_argument('--pose_control', action='store_true', default=False)
    parser.add_argument('--pose', action='store_true', default=True, help='Enable pose (default is True)')
    parser.add_argument('--no-pose', action='store_false', dest='pose', help='Disable pose')
    parser.add_argument('--test_index_offset',type=int, default=0)
    parser.add_argument('--ts',type=str, default=None)
    parser.add_argument('--skip_gt', action='store_true', default=False, help='Enable pose (default is True)')
    parser.add_argument('--show_ego', action='store_true', default=False, help='Enable pose (default is True)')
    parser.add_argument('--export_pcd', action='store_true', default=False, help='Enable pose (default is True)')
    parser.add_argument('--ema', type=bool, default=True)

    args = parser.parse_args()

    ngpus = 1
    args.gpus = ngpus
    print(args)
    main(args)

