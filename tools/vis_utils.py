from pyvirtualdisplay import Display
display = Display(visible=False, size=(2560, 1440))
display.start()
from mayavi import mlab
import mayavi
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
import numpy as np
import os
from pyquaternion import Quaternion
try:
    import open3d as o3d
except:
    pass
from functools import reduce 
import mmcv

colors = np.array(
    [
        [255, 120,  50, 255],       # barrier              orange
        [255, 192, 203, 255],       # bicycle              pink
        [255, 255,   0, 255],       # bus                  yellow
        [  0, 150, 245, 255],       # car                  blue
        [  0, 255, 255, 255],       # construction_vehicle cyan
        [255, 127,   0, 255],       # motorcycle           dark orange
        [255,   0,   0, 255],       # pedestrian           red
        [255, 240, 150, 255],       # traffic_cone         light yellow
        [135,  60,   0, 255],       # trailer              brown
        [160,  32, 240, 255],       # truck                purple                
        [255,   0, 255, 255],       # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [ 75,   0,  75, 255],       # sidewalk             dard purple
        [150, 240,  80, 255],       # terrain              light green          
        [230, 230, 250, 255],       # manmade              white
        [  0, 175,   0, 255],       # vegetation           green
        # [  0, 255, 127, 255],       # ego car              dark cyan
        # [255,  99,  71, 255],       # ego car
        # [  0, 191, 255, 255]        # ego car
    ]
).astype(np.uint8)

def pass_print(*args, **kwargs):
    pass


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def draw(
    voxels,          # semantic occupancy predictions
    pred_pts,        # lidarseg predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dir=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
    sem=False,
    show_ego=True
):

    w, h, z = voxels.shape

    # assert show_ego
    if show_ego:
        assert voxels.shape==(200, 200, 16)
        voxels[96:104, 96:104, 2:7] = 15
        voxels[104:106, 96:104, 2:5] = 3

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError


    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 17)
    ]
    # print(len(fov_voxels))

    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    voxel_size = sum(voxel_size) / 3


    plt_plot_fov = mlab.points3d(
        # fov_voxels[:, 1],
        # fov_voxels[:, 0],
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        scale_factor=1.0 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=16, # 16
    )
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    dst=os.path.join(save_dir, f'vis_{timestamp}.png')
    mlab.savefig(dst)
    mlab.close()
    # crop
    im3=mmcv.imread(dst)[:,550:-530,:]
    # im3=mmcv.imread(dst)[:,590:-600,230:-230]
    # im3=mmcv.imread(dst)[360:-230,590:-600]
    mmcv.imwrite(im3, dst)
    return dst


def write_pc(pc,dst,c=None):
    # pc=pc
    import open3d as o3d 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if c is not None:
        pcd.colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_point_cloud(dst, pcd)

def merge_mesh(meshes):
    return reduce(lambda x,y:x+y, meshes)


def get_pose_mesh(trans_mat,s=5):

    # Create a coordinate frame with x-axis (red), y-axis (green), and z-axis (blue)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=s, origin=[0, 0, 0])
    mesh_frame.transform(trans_mat)
    # Save the coordinate frame to a file
    return mesh_frame


def visualize_point_cloud(
    all_pred,
    abs_pose,
    abs_rot,
    vox_origin=[-40, -40, -1],
    resolution=0.4,  #voxel size
    cmp_dir="./",
    key='gt'
):
    assert len(all_pred)==len(abs_pose)==len(abs_rot)
    all_occ,all_color=[],[]
    pose_mesh=[]
    for i,(occ,pose,rot) in enumerate(zip(all_pred,abs_pose,abs_rot)):
        occ=occ.reshape(-1)#.flatten()
        mask=(occ>=1)&(occ<16)  # ignore GO  
        cc=colors[occ[mask]-1][:,:3]/255.0 #[...,::-1]

        # occ_x,occ_y,occ_z=np.meshgrid(np.arange(200),np.arange(200),np.arange(16))
        # occ_x=occ_x.flatten()
        # occ_y=occ_y.flatten()
        # occ_z=occ_z.flatten()
        # occ_xyz=np.concatenate([occ_x[:,None],occ_y[:,None],occ_z[:,None]],axis=1)
        # occ_xyz=(occ_xyz * resolution) + resolution / 2 # to center
        # occ_xyz+=np.array([-40,-40,-1]) # to ego
        # Compute the voxels coordinates in ego frame
        occ_xyz = get_grid_coords(
            [200,200,16], [resolution]*3
        ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
        write_pc(occ_xyz[mask],os.path.join(cmp_dir, f'vis_{key}_{i}_e.ply'),c=cc)                      

        # ego to world
        rot_m=Quaternion(rot).rotation_matrix[:3,:3]
        # rot_m=rr@rot_m
        trans_mat=np.eye(4)
        trans_mat[:3,:3]=rot_m
        trans_mat[:3,3]=pose
        rr=np.array([
            [0,1,0],
            [1,0,0],
            [0,0,1]
        ])
        occ_xyz=occ_xyz@rr.T
        occ_xyz=occ_xyz@rot_m.T +pose
        
        write_pc(occ_xyz[mask],os.path.join(cmp_dir, f'vis_{key}_{i}_w.ply'),c=cc)                      

        all_occ.append(occ_xyz[mask])
        all_color.append(cc)
        pose_mesh.append(get_pose_mesh(trans_mat))


    all_occ=np.concatenate(all_occ, axis=0)
    all_color=np.concatenate(all_color, axis=0)

    write_pc(all_occ,os.path.join(cmp_dir, f'vis_{key}_all_w.ply'),c=all_color)
    o3d.io.write_triangle_mesh(os.path.join(cmp_dir, f'vis_{key}_all_w_traj.ply'),merge_mesh(pose_mesh))


def visualize_point_cloud_no_pose(
    all_pred,
    vox_origin=[-40, -40, -1],
    resolution=0.4,  #voxel size
    cmp_dir="./",
    key='000000',
    key2='gt',
    offset=0,
):
    for i,occ in enumerate(all_pred):
        occ_d=occ.copy()
        occ=occ.reshape(-1)#.flatten()
        mask=(occ>=1)&(occ<16)  # ignore GO  
        cc=colors[occ[mask]-1][:,:3]/255.0 #[...,::-1]

        occ_xyz = get_grid_coords(
            [200,200,16], [resolution]*3
        ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
        write_pc(occ_xyz[mask],os.path.join(cmp_dir, f'vis_{key}_{i+offset:02d}_e_{key2}.ply'),c=cc)                      

        np.save(os.path.join(cmp_dir, f'vis_{key}_{i+offset:02d}_e_{key2}.npy'),occ_d) 


if __name__=='__main__':
    # np.savez('/home/users/songen.gu/adwm/OccWorld/visualizations/aaaa.npz',input_occs0=input_occs0,input_occs=input_occs,metas0=metas0,metas=metas)
    # load
    data=np.load('/home/users/songen.gu/adwm/OccWorld/visualizations/aaaa.npz')
    input_occs0=data['input_occs0']
    input_occs=data['input_occs']
    dst_dir='/home/users/songen.gu/adwm/OccWorld/visualizations/abccc'
    os.makedirs(dst_dir,exist_ok=True)
    dst_wm=draw(input_occs0[10], 
        None, # predict_pts,
        [-40, -40, -1], 
        [0.4] * 3, 
        None, #  grid.squeeze(0).cpu().numpy(), 
        None,#  pt_label.squeeze(-1),
        dst_dir,#recon_dir,
        None, # img_metas[0]['cam_positions'],
        None, # img_metas[0]['focal_positions'],
        timestamp=10,
        mode=0,
        sem=False)
    dst_dir='/home/users/songen.gu/adwm/OccWorld/visualizations/abcc2'
    os.makedirs(dst_dir,exist_ok=True)
    dst_wm=draw(input_occs[10], 
        None, # predict_pts,
        [-40, -40, -1], 
        [0.4] * 3, 
        None, #  grid.squeeze(0).cpu().numpy(), 
        None,#  pt_label.squeeze(-1),
        dst_dir,#recon_dir,
        None, # img_metas[0]['cam_positions'],
        None, # img_metas[0]['focal_positions'],
        timestamp=10,
        mode=0,
        sem=False)