import sys, os, numpy as np, time, cv2, glob, tqdm
sys.path.append("./")
from opt import opt
from engineer.core.test import recon_online, merge_norm
from engineer.utils.renderer import render_rafare, align_rafare, rotate_verts_y

sys.path.append("./engineer/face_parse/")
from face_parse import face_parse_batch

sys.path.append("./engineer/norm_pred/")
from norm_pred import norm_pred_batch

if __name__ == "__main__":

    # decode video
    src_fn = opt.input_fn
    tgt_dir = opt.output_dir

    src_name = os.path.splitext(os.path.basename(src_fn))[0]
    intermed_dir = os.path.join(tgt_dir, src_name+"_intermediate")
    os.makedirs(intermed_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(src_fn)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    src_fn_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame.shape != (512, 512):
                frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(intermed_dir + "%08d.png" % frame_idx, frame)
            src_fn_list.append(intermed_dir + "%08d.png" % frame_idx)
            frame_idx += 1
        else:
            cap.release()

    # predict semantic mask
    face_parse_batch(src_fn_list, intermed_dir, new_folder=False)

    # predict normal maps
    norm_pred_batch(src_fn_list, intermed_dir, front=True, new_folder=False)
    norm_pred_batch(src_fn_list, intermed_dir, front=False, new_folder=False)

    # recon hsdf
    model_base = recon_online(os.path.join('.', "configs", "SDF_FS103450_HG_base.py"), 
                              "epoch_best.tar", 
                              num_samples=opt.num_samples)

    model_fine = recon_online(os.path.join('.', "configs", "SDF_FS103450_HG_fine.py"), 
                              "epoch_best.tar", 
                              num_samples=opt.num_samples)
    
    mesh_fn_list = []
    for fn in tqdm.tqdm(src_fn_list, desc="hsdf recon"):
        frame_name = os.path.splitext(os.path.basename(fn))[0]

        src_img = cv2.imread(fn)
        src_sem_mask = cv2.imread(os.path.join(intermed_dir, frame_name+"_semask.png"), 0)
        
        sdf_base, mat = model_base.recon(src_img, src_sem_mask)
        sdf_fine, _ = model_fine.recon(src_img, src_sem_mask)

        norm_front = cv2.imread(os.path.join(intermed_dir, frame_name+"_norm_front.png"))
        norm_back = cv2.imread(os.path.join(intermed_dir, frame_name+"_norm_back.png"))
        mesh = merge_norm(sdf_base, sdf_fine, norm_front, norm_back, src_sem_mask, mat)
        
        mesh.export(os.path.join(os.path.join(intermed_dir, frame_name+"_mesh.obj")));
        mesh_fn_list.append(os.path.join(intermed_dir, "%s_mesh.obj" % frame_name))

    # visualize
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(tgt_dir + "/%s_vis.avi" % src_name, fourcc, fps, (2360, 512), True)

    for idx, fn in enumerate(tqdm.tqdm(mesh_fn_list, desc="render results")):
        
        src_frame = cv2.imread(src_fn_list[idx])

        # read mesh
        pred_align_mesh = align_rafare(fn, -1)

        # render in front view
        rend_front = render_rafare(pred_align_mesh, src_frame)
        rend_norm_front = render_rafare(pred_align_mesh, src_frame, norm=True)
        
        # render in front view
        rend_img_side = render_rafare(pred_align_mesh, rot=60)
        rend_norm_side = render_rafare(pred_align_mesh, norm=True, rot=60)
        
        merge_img = np.concatenate((src_frame, rend_front, rend_norm_front, 
                                    rend_img_side[:,50:-50,:], rend_norm_side[:,50:-50,:]), 
                                   axis = 1)
        
        out.write(merge_img)

    out.release()