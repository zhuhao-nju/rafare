import numpy as np
import os 
from PIL import Image
from engineer.utils.sdf import *
from skimage import measure
import torch
import tqdm
import trimesh

# to update
def gen_mesh_sdf_wild(cfg, net, data, save_path):
    '''generate mesh by marching cube
    
    Parameters:
        cfg (CfgNode): configs. Details can be found in
            configs/PIFu_Render_People_HG.py
        net: network model e.g. PIFu
        data: input data, it contains four key variables, 'img','calib', 'b_min', 'b_max'
        save_path: where you save your mesh and single-view image
    
    return
        None
    '''
    try:
        #distributed model
        net = net.module
    except:
        pass
    
    image_tensor = data['img'].cuda()
    calib_tensor = data['calib'].cuda()
    origin_calib_tensor = data['origin_calib']
    
    
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor[None,...]
    
    net.extract_features(image_tensor)
    
    b_min = data['b_min']
    b_max = data['b_max']
    save_img_path = os.path.join(save_path,"obj.jpg")
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, :] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    save_img = save_img[...,:3]
    
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)
    
    # reconstruction
    
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(cfg.resolution, cfg.resolution, cfg.resolution,
                              b_min, b_max, transform=None)
    
    # read vol 256
    raw_sdf = data['raw_sdf']
    surf_posi_norm = data['surf_posi_norm']
    surf_sd = data['surf_sd']
    
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).cuda().float()
        
        net.query(samples, calib_tensor)
        
        pred = net.get_preds()[0][0]
        
        return pred.detach().cpu().numpy()
    
    pred_surf_sd = eval_func(surf_posi_norm)
    sdf = raw_sdf.copy()
    sdf[surf_posi[0], surf_posi[1], surf_posi[2]] = pred_surf_sd
    
#     a = surf_sd/pred_surf_sd
#     for aa in a:
#         print("%f " % aa, end='')
    print("@@@", surf_sd/pred_surf_sd)
    
#     sdf = eval_grid(coords, eval_func, num_samples=50000)
    
    if True:
        from scipy.ndimage import convolve
        kernel = np.ones((5,5,5))
        kernel[1:4, 1:4, 1:4] = 2
        kernel[2, 2, 2] = 10
        kernel /= np.linalg.norm(kernel)
        
        sdf = convolve(input = sdf, 
                       weights = kernel, 
                       mode = 'nearest')
    
    # Finally we do marching cubes
    verts, faces, normals, values = measure.marching_cubes(sdf, 0)
    
    # compensate for align_corners
    verts = verts + 0.5
    
    # transform verts into world coordinate system
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    
    verts,flip_face = transfer_uv_to_world(verts,origin_calib_tensor)
    save_obj_mesh(os.path.join(save_path,"mesh.obj"), verts, faces[:,[1,0,2]],flip_face)
    

def gen_mesh_sdf(cfg, net, data, save_path):
    '''generate mesh by marching cube
    
    Parameters:
        cfg (CfgNode): configs. Details can be found in
            configs/PIFu_Render_People_HG.py
        net: network model e.g. PIFu
        data: input data, it contains four key variables, 'img','calib', 'b_min', 'b_max'
        save_path: where you save your mesh and single-view image
    
    return
        None
    '''
    try:
        #distributed model
        net = net.module
    except:
        pass
    image_tensor = data['img'].cuda()
    calib_tensor = data['calib'].cuda() # eye(4)
    origin_calib_tensor = data['origin_calib'] # eye(4) with [1, 1] = -1
    
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor[None,...]
    
    net.extract_features(image_tensor)
    
    b_min = data['b_min']
    b_max = data['b_max']
    save_img_path = os.path.join(save_path,"obj.jpg")
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, :] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    save_img = save_img[...,:3]
    
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)
    
    # reconstruction
    
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(cfg.resolution, cfg.resolution, cfg.resolution,
                              b_min, b_max, transform=None)
    
    # read vol 256
    sdf = np.ones((256, 256, 256))
    
    # read surf_posi
#     surf_posi = np.load(os.path.join(os.path.dirname(data['name']), "surf_posi.npy"))
#     surf_posi[1,:] = 255 - surf_posi[1,:]
#     surf_posi_norm = (surf_posi.copy().astype(np.float32)) / 128 - 1
    
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).cuda().float()
        
        net.query(samples, calib_tensor)
        
        pred = net.get_preds()[0][0]
        
        return pred.detach().cpu().numpy()
    
    sdf = eval_grid(coords, eval_func, num_samples=50000)
    
    if False:
        from scipy.ndimage import convolve
        kernel = np.ones((5,5,5))
        kernel[1:4, 1:4, 1:4] = 2
        kernel[2, 2, 2] = 10
        kernel /= np.linalg.norm(kernel)
        
        sdf = convolve(input = sdf, 
                       weights = kernel, 
                       mode = 'nearest')
    
    try:
        # Finally we do marching cubes
        verts, faces, normals, values = measure.marching_cubes(sdf, 0)

        # compensate for align_corners
        verts = verts + 0.5

        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T

        verts,flip_face = transfer_uv_to_world(verts,origin_calib_tensor)
        save_obj_mesh(os.path.join(save_path,"mesh.obj"), verts, faces[:,[1,0,2]],flip_face)
    except:
        pass
    
    # save out sdf
    np.save(os.path.join(save_path,"sdf_%s.npy" % cfg.phase), sdf.astype(np.float32))
    
    
def gen_mesh_sdf_surf(cfg, net, data, save_path):
    '''generate mesh by marching cube
    
    Parameters:
        cfg (CfgNode): configs. Details can be found in
            configs/PIFu_Render_People_HG.py
        net: network model e.g. PIFu
        data: input data, it contains four key variables, 'img','calib', 'b_min', 'b_max'
        save_path: where you save your mesh and single-view image
    
    return
        None
    '''
    try:
        #distributed model
        net = net.module
    except:
        pass
    image_tensor = data['img'].cuda()
    calib_tensor = data['calib'].cuda()
    origin_calib_tensor = data['origin_calib']
    
    
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor[None,...]
    
    net.extract_features(image_tensor)
    
    b_min = data['b_min']
    b_max = data['b_max']
    save_img_path = os.path.join(save_path,"obj.jpg")
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, :] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    save_img = save_img[...,:3]
    
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)
    
    # reconstruction
    
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(cfg.resolution, cfg.resolution, cfg.resolution,
                              b_min, b_max, transform=None)
    
    # read vol 256
    io_vol = np.load(os.path.join(os.path.dirname(data['name']), "io_vol.npy"))
    io_vol = np.flip(io_vol, axis=1) # align volume to image
    sdf = (io_vol.copy().astype(np.float)*2 - 1)*0.1
    
    # read surf_posi
    surf_posi = np.load(os.path.join(os.path.dirname(data['name']), "surf_posi.npy"))
    surf_posi[1,:] = 255 - surf_posi[1,:]
    surf_posi_norm = (surf_posi.copy().astype(np.float32)) / 128 - 1
    
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).cuda().float()
        
        net.query(samples, calib_tensor)
        
        pred = net.get_preds()[0][0]
        
        return pred.detach().cpu().numpy()
    
    batch_size = 10000
    batch_num = int(np.ceil(surf_posi_norm.shape[1]/batch_size))
    
    for batch in tqdm.trange(batch_num):
        
        pred_surf_sd = eval_func(surf_posi_norm[:,batch*batch_size:(batch+1)*batch_size])
        sdf[surf_posi[0][batch*batch_size:(batch+1)*batch_size], 
            surf_posi[1][batch*batch_size:(batch+1)*batch_size], 
            surf_posi[2][batch*batch_size:(batch+1)*batch_size]] = pred_surf_sd
    
#     a = surf_sd/pred_surf_sd
#     for aa in a:
#         print("%f " % aa, end='')
#    print("@@@", surf_sd/pred_surf_sd)
    
#     sdf = eval_grid(coords, eval_func, num_samples=50000)
    
    if False:
        from scipy.ndimage import convolve
        kernel = np.ones((5,5,5))
        kernel[1:4, 1:4, 1:4] = 2
        kernel[2, 2, 2] = 10
        kernel /= np.linalg.norm(kernel)
        
        sdf = convolve(input = sdf, 
                       weights = kernel, 
                       mode = 'nearest')
    
    # Finally we do marching cubes
    verts, faces, normals, values = measure.marching_cubes(sdf, 0)
    
    # compensate for align_corners
    verts = verts + 0.5
    
    # transform verts into world coordinate system
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    
    verts,flip_face = transfer_uv_to_world(verts,origin_calib_tensor)
    save_obj_mesh(os.path.join(save_path,"mesh.obj"), verts, faces[:,[1,0,2]],flip_face)
    
    
def reconstruction(net, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=50000, transform=None,crop_query_points = None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :param crop_query_points: crop_query_points used for fine-pifu inference
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).cuda().float()

        if crop_query_points is None:
            #coarse-pifu
            net.query(samples, calib_tensor)
        else:
            #fine-pifu
            net.global_net.query(samples, calib_tensor)
            local_features = net.global_net.get_merge_feature()
            net.query(samples, local_features,calib_tensor = calib_tensor)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()
    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1
    
    
def gen_mesh(cfg, net, data, save_path, use_octree=True):
    '''generate mesh by marching cube

    Parameters:
        cfg (CfgNode): configs. Details can be found in
            configs/PIFu_Render_People_HG.py
        net: network model e.g. PIFu
        data: input data, it contains four key variables, 'img','calib', 'b_min', 'b_max'
        save_path: where you save your mesh and single-view image
        use_octree: default, True
    
    return 
        None
    '''
    try:
        #distributed model
        net = net.module
    except:
        pass
    image_tensor = data['img'].cuda()
    calib_tensor = data['calib'].cuda()
    origin_calib_tensor = data['origin_calib']


    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor[None,...]
    
    if not cfg.fine_pifu:
        #coarse-pifu inference
        net.extract_features(image_tensor)
        crop_query_points = None
    else:
        #fine-pifu inference
        crop_imgs=  data['crop_img']
        crop_query_points = data['crop_query_points']
        net.global_net.extract_features(image_tensor)
        net.extract_features(crop_imgs)

    b_min = data['b_min']
    b_max = data['b_max']
    save_img_path = os.path.join(save_path,"obj.jpg")
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, :] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    save_img = save_img[...,:3]

    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

    verts, faces, _, _ = reconstruction(net, calib_tensor, cfg.resolution, b_min, b_max, use_octree=use_octree,crop_query_points = crop_query_points)
    
    verts,flip_face = transfer_uv_to_world(verts,origin_calib_tensor)
    save_obj_mesh(os.path.join(save_path,"mesh.obj"), verts, faces,flip_face)


def transfer_uv_to_world(verts,origin_calib,img_size=512,z_depth=200):

    
    if origin_calib == None:
        return verts,False
    #verts[...,2] = verts[...,2]*z_depth/(img_size//2) # commented by Hao, this will lead to strange scale in z axis, still uncertain what is the intension of this line
    mat = origin_calib.detach().numpy()
    inv_mat = np.linalg.inv(mat)
    homo_verts = np.concatenate([verts,np.ones((verts.shape[0],1))],axis=1)
    ori_verts = np.matmul(inv_mat,homo_verts.T).T
    return ori_verts[...,:3],True

def save_obj_mesh(mesh_path, verts, faces, flip=False):
    '''save mesh, xxx.obj
    
    Parameters:
        mesh_path: save to where
        verts: the vertices of mesh [N, 3]
        faces: face_id [N, 3]->[Int] 
    return None
    '''
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        if flip:           
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
        else:
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    '''save mesh with color, xxx.obj
    
    Parameters:
        mesh_path: save to where
        verts: the vertices of mesh [N, 3]
        faces: face_id [N, 3]->[Int] 
        colors: face color: [N, 3] rgb

    return None
    '''
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    '''save mesh with uv map, xxx.obj
    
    Parameters:
        mesh_path: save to where
        verts: the vertices of mesh [N, 3]
        faces: face_id [N, 3]->[Int] 
        uvs: face color: [N, 3] rgb

    return None
    '''
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()

def rm_isolate_mesh(mesh):
    cc = trimesh.graph.connected_components(mesh.face_adjacency)

    if len(cc) > 1: # exist isolated meshes
        main_mesh = cc[0]
        for c in cc:
            if len(c)>len(main_mesh):
                main_mesh = c

        mask = np.zeros(len(mesh.faces), dtype=np.bool)
        mask[main_mesh] = True

        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
    return mesh