"""
Copyright 2020, Hao Zhu, NJU.
Parametric model fitter..
"""

import numpy as np, pyrender, trimesh, cv2

# render with gl camera
def render_glcam(model_in, # model name or trimesh
                 K = None,
                 Rt = None,
                 scale = 1.0,
                 rend_size = (512, 512),
                 light_trans = np.array([[0], [100], [0]]),
                 flat_shading = False):
    
    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()
    pr_mesh = pyrender.Mesh.from_trimesh(mesh)

    # Scene creation
    scene = pyrender.Scene()

    # Adding objects to the scene
    face_node = scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0] * scale, K[1][1] * scale
    cx, cy = K[0][2] * scale, K[1][2] * scale

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, 
                                    znear=0.1, zfar=100000)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = Rt[:3, :3].T
    cam_pose[:3, 3] = -Rt[:3, :3].T.dot(Rt[:, 3])
    scene.add(cam, pose=cam_pose)

    # Set up the light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    light_pose = cam_pose.copy()
    light_pose[0:3, :] += light_trans
    scene.add(light, pose=light_pose)

    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)
    if flat_shading is True:
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    else:
        color, depth = r.render(scene)

    # rgb to bgr for cv2
    color = color[:, :, [2, 1, 0]]

    return depth, color


# render with cv camera
def render_cvcam(model_in, # model name or trimesh
                 K = None,
                 Rt = None,
                 scale = 1.0,
                 rend_size = (512, 512),
                 light_trans = np.array([[0], [100], [0]]),
                 flat_shading = False):
    
    if np.array(K).all() == None:
        K = np.array([[2000, 0, 256],
                      [0, 2000, 256],
                      [0, 0, 1]], dtype=np.float64)
        
    if np.array(Rt).all() == None:
        Rt = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]], dtype=np.float64)
    
    # define R to transform from cvcam to glcam
    R_cv2gl = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
    Rt_cv = R_cv2gl.dot(Rt)
    
    return render_glcam(model_in, K, Rt_cv, scale, rend_size, light_trans, flat_shading)

# render with orth camera
def render_orthcam(model_in, # model name or trimesh
                   xy_mag,
                   rend_size,
                   flat_shading=False,
                   zfar = 10000,
                   znear = 0.05):
    
    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()
    pr_mesh = pyrender.Mesh.from_trimesh(mesh)
  
    # Scene creation
    scene = pyrender.Scene()
    
    # Adding objects to the scene
    face_node = scene.add(pr_mesh)
    
    # Camera Creation
    if type(xy_mag) == float:
        cam = pyrender.OrthographicCamera(xmag = xy_mag, ymag = xy_mag, 
                                          znear=znear, zfar=zfar)
    elif type(xy_mag) == tuple:
        cam = pyrender.OrthographicCamera(xmag = xy_mag[0], ymag = xy_mag[1], 
                                          znear=znear, zfar=zfar)
    else:
        print("Error: xy_mag should be float or tuple")
        return False
        
    scene.add(cam, pose=np.eye(4))
    
    # Set up the light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    scene.add(light, pose=np.eye(4))
    
    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)
    if flat_shading is True:
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    else:
        color, depth = r.render(scene)
    
    # rgb to bgr for cv2
    color = color[:, :, [2, 1, 0]]
    
    # fix pyrender BUG of depth rendering, pyrender version: 0.1.43
    depth[depth!=0] = (zfar + znear - ((2.0 * znear * zfar) / depth[depth!=0]) ) / (zfar - znear)
    depth[depth!=0] = ( ( depth[depth!=0] + (zfar + znear) / (zfar - znear) ) * (zfar - znear) ) / 2.0
    
    return depth, color


# render with orth camera
def render_orthcam(model_in, # model name or trimesh
                   xy_mag,
                   rend_size,
                   flat_shading=False,
                   zfar = 10000,
                   znear = 0.05):

    
    pr_mesh = pyrender.Mesh.from_trimesh(model_in.copy())
    
    # Scene creation
    scene = pyrender.Scene()
    
    # Adding objects to the scene
    face_node = scene.add(pr_mesh)
    
    # Camera Creation
    cam = pyrender.OrthographicCamera(xmag = xy_mag[0], ymag = xy_mag[1], 
                                      znear=znear, zfar=zfar)
    
    scene.add(cam, pose=np.eye(4))
    
    # Set up the light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    scene.add(light, pose=np.eye(4))
    
    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)
    if flat_shading is True:
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    else:
        color, depth = r.render(scene)
    
    # rgb to bgr for cv2
    color = color[:, :, [2, 1, 0]]
    
    # IMPORTANT! FIX pyrender BUG, pyrender version: 0.1.43
    depth[depth!=0] = (zfar + znear - ((2.0 * znear * zfar) / depth[depth!=0]) ) / (zfar - znear)
    depth[depth!=0] = ( ( depth[depth!=0] + (zfar + znear) / (zfar - znear) ) * (zfar - znear) ) / 2.0
    
    return depth, color
    
    
# rotate verts along y axis
def rotate_verts_y(verts, y):
    verts_mean = np.mean(verts, axis = 0)
    verts = verts - verts_mean

    angle = y*np.math.pi/180
    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    
    verts = np.tensordot(R, verts.T, axes = 1).T + verts_mean
    return verts


def align_rafare(fn, f_gt=-1):
    gt_img_size = 512
    render_bias = 2.
    R_pers2ortho = np.array([[1, 0, 0], 
                             [0, -1, 0], 
                             [0, 0, -1]], dtype = np.float64)
    
    # read mesh
    pred_world_mesh = load_ori_mesh(fn)
    pred_align_mesh = pred_world_mesh.copy()
    
    if f_gt > 0:
        # orthogonal to perspective
        pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
        pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])        
        pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))
                
        # scale to make face width = 1
        pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                     np.min(pred_align_mesh.vertices[:, 1]))/3
    else:
        # keep orthogonal camera, add bias to make pyrender work
        pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    return pred_align_mesh



def clahe_L(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # Converting to LAB channels
    l, a, b = cv2.split(lab) # Splitting the LAB channels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5)) # Applying CLAHE to L-channel
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b)) # Merge channels
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) # Converting back to RGB channels
    return final

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)
    
    return norm


def load_ori_mesh(fn):
    return trimesh.load(fn, 
                        resolver = None, 
                        split_object = False, 
                        group_material = False, 
                        skip_materials = True, 
                        maintain_order = True, 
                        process = False)

def render_rafare(pred_align_mesh, background=None, norm=False, rot=0):
    
    this_mesh = pred_align_mesh.copy()
    if rot!=0:
        this_mesh.vertices = rotate_verts_y(this_mesh.vertices, rot)
        
    if norm is False:
        # rend mesh
        depth_img, rend_img = render_orthcam(this_mesh, 
                                             xy_mag = (1,1), 
                                             rend_size = (512, 512))
        rend_img = clahe_L(rend_img)
        if background is None:
            return rend_img
        else:
            mask = np.stack((depth_img==0, )*3).transpose((1, 2, 0))
            merge_img = background.copy()
            merge_img[mask==0] = rend_img[mask==0]
            return merge_img
    else:
        norms = compute_normal(this_mesh.vertices, this_mesh.faces)
        this_mesh.visual = trimesh.visual.ColorVisuals(mesh = this_mesh, 
                                                       vertex_colors = (norms[:,[2,1,0]] + 1) / 2)
        depth_img, rend_img = render_orthcam(this_mesh, 
                                             xy_mag = (1,1), 
                                             rend_size = (512, 512),
                                             flat_shading = True)
        rend_img = rend_img[:,:,::-1]
        if background is None:
            return rend_img
        else:
            mask = np.stack((depth_img==0, )*3).transpose((1, 2, 0))
            merge_img = background.copy()
            merge_img[mask==0] = rend_img[mask==0]
            return merge_img




