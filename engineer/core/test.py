import numpy as np, trimesh, torch, os, cv2, tqdm
from opt import opt
from skimage import measure
#from mmcv import Config
from engineer.utils.mmcv_config import Config
from engineer.models.builder import build_model
from engineer.utils.sdf import create_grid, eval_grid
from engineer.utils.common import transfer_uv_to_world, norm_carve, sdf_conv, clean_sdf_edge, clean_sdf_bg
from engineer.utils.mesh_utils import rm_isolate_mesh
from utils.logger import get_experiments_id
from utils.distributed import load_checkpoints

# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning)

# background, skin, nose, eye_g, l_eye, r_eye, l_brow, r_brow, l_ear, r_ear, 
# mouth, u_lip, l_lip, hair, hat, ear_r, neck_l, neck, cloth
src2mask_dict = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
src2sem_dict = np.array([8, 4, 0, 4, 2, 2, 3, 3, 6, 6, 5, 1, 1, 7, 7, 6, 6, 6, 8])

def recon_single(cfg_fn, epoch_name, src_img, src_sem_mask, num_samples=180000):

    mask = src2mask_dict[src_sem_mask].astype(np.uint8)*255
    
    cfg = Config.fromfile(cfg_fn)
    model = build_model(cfg.model).cuda()
    
    # resume
    checkpoints_path, _ = get_experiments_id(cfg)
    resume_path = os.path.join(checkpoints_path, epoch_name)
    epoch = load_checkpoints(model, None, resume_path, opt)

    # predict base sdf
    sdf_base, mat = hsdf_pred(model, src_img, src_sem_mask, mask, 
                              vol_size=256, num_samples=num_samples)

    return sdf_base, mat


def recon_batch(cfg_fn, epoch_name, fn_list, tgt_dir, phase, num_samples=180000):

    cfg = Config.fromfile(cfg_fn)
    model = build_model(cfg.model).cuda()
    
    # resume
    checkpoints_path, _ = get_experiments_id(cfg)
    resume_path = os.path.join(checkpoints_path, epoch_name)
    epoch = load_checkpoints(model, None, resume_path, opt)

    if phase == "base":
        desc = "[4/6] recon " + phase
    elif phase == "fine":
        desc = "[5/6] recon " + phase
    else:
        desc = "recon"

    for fn in tqdm.tqdm(fn_list, desc=desc):
        
        # read image
        src_img = cv2.imread(fn)
        if src_img.shape != (512, 512, 3):
            src_img = cv2.resize(src_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        src_img = src_img[:,:,::-1]

        # read mask
        src_name = os.path.splitext(os.path.basename(fn))[0]
        src_sem_mask = cv2.imread(tgt_dir+"/%s/%s_semask.png" % (src_name,src_name),0)
        mask = src2mask_dict[src_sem_mask].astype(np.uint8)*255
        
        # predict base sdf
        sdf_base, mat = hsdf_pred(model, src_img, src_sem_mask, mask, 
                                  vol_size=256, num_samples=num_samples)
        
        os.makedirs(tgt_dir+"/%s/"%src_name, exist_ok=True)
        np.save(tgt_dir+"/%s/%s_sdf_%s.npy"%(src_name,src_name,phase), sdf_base)
        
    return True

class recon_online():

    def __init__(self, cfg_fn, epoch_name, num_samples=80000):
        cfg = Config.fromfile(cfg_fn)
        self.model = build_model(cfg.model).cuda()
        self.num_samples = num_samples
        
        # resume
        checkpoints_path, _ = get_experiments_id(cfg)
        resume_path = os.path.join(checkpoints_path, epoch_name)
        epoch = load_checkpoints(self.model, None, resume_path, opt)

    def recon(self, src_img, src_sem_mask):

        mask = src2mask_dict[src_sem_mask].astype(np.uint8)*255

        # predict base sdf
        sdf_base, mat = hsdf_pred(self.model, src_img, src_sem_mask, mask, 
                                vol_size=256, num_samples=self.num_samples)
        return sdf_base, mat



# move to core/test
def hsdf_pred(model, src_img, src_sem_mask, mask, vol_size, num_samples):
    
    # preprocess
    sem_mask = src2sem_dict[np.expand_dims(src_sem_mask, 2)].transpose(2, 0, 1).astype(np.uint8)*20/255
    
    src_img = src_img.astype(np.float32).transpose(2, 0, 1)/255.
    src_img = (src_img - 0.5)/0.5
    src_img = np.concatenate((src_img, sem_mask), 0)
    
    image_tensor = torch.Tensor(src_img).float().cuda().unsqueeze(0)
    
    mask_tensor = torch.Tensor(mask).float().cuda()
    mask_tensor[mask_tensor!=0] = 1
    mask_tensor = torch.stack((mask_tensor,)*4, 0).unsqueeze(0)
    
    image_tensor = mask_tensor * image_tensor
    
    projection_matrix = np.identity(4)
    projection_matrix[1, 1] = -1
    calib_tensor = torch.Tensor(projection_matrix).float().cuda().unsqueeze(0)
        
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor[None,...]
    
    model.extract_features(image_tensor)
    
    # reconstruction
    coords, mat = create_grid(vol_size, vol_size, vol_size, 
                              np.array([-1., -1., -1.]), 
                              np.array([+1., +1., +1.]), transform=None)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, model.num_views, axis=0)
        samples = torch.from_numpy(points).cuda().float()
        model.query(samples, calib_tensor)
        pred = model.get_preds()[0][0]
        return pred.detach().cpu().numpy()
    
    sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    
    return sdf, mat

# move to core/test
def merge_norm(sdf_base, sdf_fine, norm_front, norm_back, src_sem_mask, mat):
    
    origin_calib_tensor = torch.Tensor(np.identity(4)).float().cuda().unsqueeze(0)
    
    sdf = sdf_conv(sdf_base, 5) + sdf_fine*0.001
    
    # normal_carve
    sdf = norm_carve(sdf, src_sem_mask, norm_front, norm_back)
    
    # clean background and cloth
    sdf = clean_sdf_bg(sdf, src_sem_mask, edge_dist = 1)
            
    # clean edge
    sdf = clean_sdf_edge(sdf)

    # Finally we do marching cubes
    verts, faces, _, _ = measure.marching_cubes(sdf, 0)
    
    # compensate for align_corners
    verts = verts + 0.5

    # transform verts into world coordinate system
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T

    verts, flip_face = transfer_uv_to_world(verts, origin_calib_tensor.cpu())

    mesh = trimesh.Trimesh(vertices = verts[:,:3,0], faces = faces, process = False, maintain_order = True)

    # remove isolated meshes
    mesh = rm_isolate_mesh(mesh)
    return mesh
