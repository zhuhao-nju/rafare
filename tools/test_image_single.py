import sys, os, numpy as np, time, cv2
sys.path.append("./")
from opt import opt
from engineer.core.test import recon_single, merge_norm
from engineer.utils.renderer import render_rafare, align_rafare, rotate_verts_y

sys.path.append("./engineer/face_parse/")
from face_parse import face_parse

sys.path.append("./engineer/norm_pred/")
from norm_pred import norm_pred

if __name__ == "__main__":

    # read image
    src_dirname = opt.input_fn
    tgt_dir = opt.output_dir
    
    src_name = os.path.splitext(os.path.basename(src_dirname))[0]
    src_dir = os.path.dirname(src_dirname)
    
    src_img = cv2.imread(src_dirname)
    if src_img.shape != (512, 512, 3):
        src_img = cv2.resize(src_img, (512, 512), interpolation=cv2.INTER_LINEAR)

    start_time0 = time.time()
    
    # predict semantic mask
    src_sem_mask = face_parse(src_img[:,:,::-1])
    
    # predict normal maps
    norm_front = norm_pred(src_img, front=True)
    norm_back = norm_pred(src_img, front=False)

    # recon base
    sdf_base, mat = recon_single(os.path.join('.', "configs", "SDF_FS103450_HG_base.py"), "epoch_best.tar",
                                 src_img, src_sem_mask, num_samples=opt.num_samples)

    # recon fine
    sdf_fine, mat = recon_single(os.path.join('.', "configs", "SDF_FS103450_HG_fine.py"), "epoch_best.tar",
                                 src_img, src_sem_mask, num_samples=opt.num_samples)

    mesh = merge_norm(sdf_base, sdf_fine, norm_front, norm_back, src_sem_mask, mat)
    
    end_time = time.time()
    print("Run time: %f" % (end_time-start_time0))

    # save
    os.makedirs(tgt_dir, exist_ok=True)
    mesh.export(os.path.join(tgt_dir, src_name+'.obj'));
    
    # reload mesh
    pred_align_mesh = align_rafare(os.path.join(tgt_dir, src_name+'.obj'), -1)

    # render in front view
    rend_front = render_rafare(pred_align_mesh, src_img)
    rend_norm_front = render_rafare(pred_align_mesh, src_img, norm=True)
    
    # render in front view
    rend_img_side = render_rafare(pred_align_mesh, rot=60)
    rend_norm_side = render_rafare(pred_align_mesh, norm=True, rot=60)
    
    merge_img = np.concatenate((src_img, rend_front, rend_norm_front, 
                                rend_img_side[:,50:-50,:], rend_norm_side[:,50:-50,:]), 
                                axis = 1)
    cv2.imwrite(os.path.join(tgt_dir, src_name+'_vis.png'), merge_img)

