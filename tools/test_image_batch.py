import sys, os, numpy as np, time, cv2, glob, tqdm
sys.path.append("./")
from opt import opt
from engineer.core.test import recon_online, merge_norm
from engineer.utils.renderer import render_rafare, align_rafare, rotate_verts_y

sys.path.append("./engineer/face_parse/")
from face_parse import face_parse_batch

sys.path.append("./engineer/norm_pred/")
from norm_pred import norm_pred_batch

suffix_list = ['jpg', 'png', 'bmp', 'jpeg', 'tif', 'tiff', 'pgm', 
                'JPG', 'PNG', 'BMP', 'JPEG', 'TIF', 'TIFF', 'PGM']

if __name__ == "__main__":
    
    # read image
    src_dir = opt.input_dir
    tgt_dir = opt.output_dir
    
    src_fn_list = []
    for suffix in suffix_list:
        src_fn_list += glob.glob(os.path.join(src_dir, '*.'+suffix))

    # # predict semantic mask
    face_parse_batch(src_fn_list, tgt_dir, new_folder=True)

    # predict normal maps
    norm_pred_batch(src_fn_list, tgt_dir, front=True, new_folder=True)
    norm_pred_batch(src_fn_list, tgt_dir, front=False, new_folder=True)

    # recon hsdf
    model_base = recon_online(os.path.join('.', "configs", "SDF_FS103450_HG_base.py"), 
                              "epoch_best.tar", 
                              num_samples=opt.num_samples)

    model_fine = recon_online(os.path.join('.', "configs", "SDF_FS103450_HG_fine.py"), 
                              "epoch_best.tar", 
                              num_samples=opt.num_samples)
    
    for fn in tqdm.tqdm(src_fn_list, desc="hsdf recon"):
        src_name = os.path.splitext(os.path.basename(fn))[0]

        src_img = cv2.imread(fn)
        src_sem_mask = cv2.imread(os.path.join(tgt_dir, src_name, src_name+"_semask.png"), 0)
        
        sdf_base, mat = model_base.recon(src_img, src_sem_mask)
        sdf_fine, _ = model_fine.recon(src_img, src_sem_mask)

        norm_front = cv2.imread(os.path.join(tgt_dir, src_name, src_name+"_norm_front.png"))
        norm_back = cv2.imread(os.path.join(tgt_dir, src_name, src_name+"_norm_back.png"))
        mesh = merge_norm(sdf_base, sdf_fine, norm_front, norm_back, src_sem_mask, mat)
        
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

