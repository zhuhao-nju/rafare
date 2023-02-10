import numpy as np, cv2
from scipy.ndimage import convolve
from skimage.morphology import erosion, disk

def transfer_uv_to_world(verts,origin_calib):
    if origin_calib == None:
        return verts,False
    mat = origin_calib.detach().numpy()
    inv_mat = np.linalg.inv(mat)
    homo_verts = np.concatenate([verts,np.ones((verts.shape[0],1))],axis=1)
    ori_verts = np.matmul(inv_mat,homo_verts.T).T
    return ori_verts[...,:3],True


def make_bump_kernel(kernel_size):
    half_size = kernel_size//2
    ori_size = kernel_size
    kernel_size = half_size*2 +1

    if kernel_size!=ori_size:
        print("warning: kenerl size was adjusted from %d to %d" % (ori_size, kernel_size))

    channel_0 = [[half_size-x for x in range(kernel_size)]]*kernel_size
    channel_1 = [[half_size-x]*kernel_size for x in range(kernel_size)]
    channel_2 = [[0]*kernel_size]*kernel_size

    kernel = np.stack((channel_2, channel_1, channel_0), axis = 2)
    
    kernel[kernel<0] = -1
    kernel[kernel>0] = 1
    
    return kernel.astype(np.float32)


def sdf_conv(sdf, kernel_size):
    conv_kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.float32)
    conv_kernel = conv_kernel / np.sum(conv_kernel)
    sdf = convolve(input = sdf, weights = conv_kernel, mode = 'nearest')
    return sdf

def clean_sdf_edge(sdf, vol_size = 256, edge_ratio=0.02, edge_dist=0.04):
    edge_size = round(vol_size*edge_ratio)
    sdf[:edge_size,:,:], sdf[-edge_size:,:,:] = edge_dist, edge_dist
    sdf[:,:edge_size,:], sdf[:,-edge_size:,:] = edge_dist, edge_dist
    sdf[:,:,:edge_size], sdf[:,:,-edge_size:] = edge_dist, edge_dist

    return sdf

# clean background and cloth
def clean_sdf_bg(sdf, src_sem_mask, vol_size = 256, erode_ratio = 0.05, edge_dist = 1, only_face = True):
    if only_face is True:
        sem2sdfmask_dict = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    else:
        sem2sdfmask_dict = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    
    clean_mask = sem2sdfmask_dict[src_sem_mask]
    clean_mask = cv2.resize(clean_mask, (vol_size, vol_size), interpolation = cv2.INTER_NEAREST)
    
    kernel = np.ones(tuple(np.round((np.array(clean_mask.shape)*erode_ratio)).astype(np.int)),np.uint8)
    clean_mask = cv2.erode(clean_mask.astype(np.uint8), kernel, iterations = 1)
    
    clean_volume = np.stack((clean_mask,)*vol_size)
    clean_volume = np.roll(np.flip(clean_volume.transpose((2, 1, 0)), axis = 1), 1, axis = 1)
    
    sdf[clean_volume>0] = edge_dist
    return sdf


def norm_carve(sdf, src_sem_mask, norm_front, norm_back, conv_radius = 7, norm_size = 512, carve_sc = 0.0005):

    sem2cleanmask_dict = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    
    norm_front = (norm_front.astype(np.float32)-128)/128
    norm_back = (norm_back.astype(np.float32)-128)/128

    kernel = make_bump_kernel(conv_radius)

    disp_front = convolve(norm_front[:,:,0], weights=kernel[:,:,0], mode='nearest') + \
                    convolve(norm_front[:,:,1], weights=kernel[:,:,1], mode='nearest') + \
                    convolve(norm_front[:,:,2], weights=kernel[:,:,2], mode='nearest')
    disp_back = convolve(norm_back[:,:,0], weights=kernel[:,:,0], mode='nearest') + \
                convolve(norm_back[:,:,1], weights=kernel[:,:,1], mode='nearest') + \
                convolve(norm_back[:,:,2], weights=kernel[:,:,2], mode='nearest')
    
    clean_mask = sem2cleanmask_dict[src_sem_mask]
    clean_mask = cv2.resize(clean_mask, (norm_size, norm_size), interpolation = cv2.INTER_NEAREST)
    
    norm_mask = erosion((-clean_mask+1).astype(np.float32), footprint = disk(conv_radius))
    norm_mask = convolve(norm_mask, weights=disk(conv_radius)/np.sum(disk(conv_radius)), mode='nearest')
    
    disp_front *= norm_mask
    disp_back *= norm_mask
    
    disp_front_reisze = (disp_front[::2,::2] + disp_front[1::2,1::2])/2
    disp_cube_front = np.stack([np.flip(disp_front_reisze.T, axis = 1)]*256, axis = 2)
    sdf[:,:,128:] = sdf[:,:,128:] - disp_cube_front[:,:,128:]*carve_sc

    disp_back_reisze = (disp_back[::2,::2] + disp_back[1::2,1::2])/2
    disp_cube_front = np.stack([np.flip(disp_back_reisze.T, axis = 1)]*256, axis = 2)
    sdf[:,:,:128] = sdf[:,:,:128] + disp_cube_front[:,:,:128]*carve_sc

    return sdf
