import  numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
import cv2
import os
import torch
import copy
from engineer.utils.metrics import *
from utils.structure import AverageMeter
from utils.distributed import reduce_tensor
import time
import logging
import torch.distributed as dist
from engineer.utils.mesh_utils import gen_mesh_sdf, gen_mesh_sdf_wild
import torch.nn.functional as F
logger = logging.getLogger('logger.trainer')
import numpy as np

cv2.setNumThreads(0)

torch.nn.NLLLoss

def inference_wild(model, cfg, args, test_dir, raw_vol_dir, save_gallery_path):
    model.eval()
    
    img_name_list = os.listdir(test_dir)
    img_name_list = [img_name for img_name in img_name_list if img_name[-4:]=='.png' or img_name[-4:]=='.jpg']
    
    for img_name in img_name_list:
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float().cuda()
        name = img_name[:-4]
        
        src_img = cv2.imread(test_dir + img_name)
        
        # resize
        img = cv2.resize(src_img, (256, 256))
        
        # ImageToTensor
        img = src_img.astype(np.float32).transpose(2, 0, 1)/255.
        
        # normalize
        img = (img - 0.5)/0.5
        
        # ToTensor
        img = torch.Tensor(img).float().cuda()
        calib = torch.Tensor(projection_matrix).float().cuda()
        
        # make batch
        img = img.unsqueeze(0)
        calib = calib.unsqueeze(0)#.unsqueeze(0)
        
        save_gallery_path = os.path.join(gallery_id,name.split('/')[-2])
        os.makedirs(save_gallery_path,exist_ok=True)
        
        # get raw sdf from predicted pifu
        io_vol_128 = np.load(raw_vol_dir + "%s.npy" % name)
        io_vol_256 = np.repeat(np.repeat(np.repeat(io_vol_128, 2, axis=0), 2, axis=1), 2, axis=2)
        
        io_vol_256 = np.flip(io_vol_256, axis=1) # align volume to image
        raw_sdf = io_vol_256.copy().astype(np.float)*2 - 1
        
        # get surface position
        surf_posi = np.load(os.path.join(os.path.dirname(name), "surf_posi.npy"))
        surf_posi[1,:] = 255 - surf_posi[1,:]
        surf_posi_norm = (surf_posi.copy().astype(np.float32)) / 128 - 1
        
        # get surface signed distance
        surf_sd = np.load(os.path.join(os.path.dirname(name), "surf_sd.npy"))
        
        # get neigbors
        io_vol_256_pad = np.pad(io_vol_256, 1, mode='edge')
        neigbors = np.zeros((27, len(surf_posi.T)))
        for i in range(len(surf_posi.T)):
            this_posi = surf_posi.T[i]
            neigbors[:, i] = io_vol_256_pad[this_posi[0]:this_posi[0]+3, 
                             this_posi[1]:this_posi[1]+3, 
                             this_posi[2]:this_posi[2]+3,].flatten()
        
        data= {'name': name,
               'img': img,
               'calib': calib,
               'mask': None,
               'b_min': B_MIN,
               'b_max': B_MAX,
               'origin_calib':None,
               'crop_img':None,
               'crop_query_points':None,
               'raw_sdf':raw_sdf,
               'surf_posi_norm':surf_posi_norm,
               'surf_sd':surf_sd,
               'neigbors':neigbors,
              }
        
        os.makedirs(save_gallery_path + name + '/', exist_ok=True)        
        
        with torch.no_grad():
            gen_mesh_sdf_wild(cfg, model, data, save_gallery_path + name + '/')

def inference(model, cfg, args, test_loader, epoch,gallery_id,gallery_time):
    model.eval()
    watched = 0
    for batch in test_loader:
        watched+=1
        if watched>gallery_time:
            break
        logger.info("time {}/{}".format(watched,gallery_time))
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        calib = torch.Tensor(projection_matrix).float().cuda()
        projection_matrix[1, 1] = -1
        origin_calib = projection_matrix
        
        name = batch['name'][0]
        img = batch['img']
        
        crop_imgs = None
        crop_query_points = None

        try:
            origin_calib = batch['calib'][0]
        except:
            #there is not origina calib matrix
            origin_calib = None
            projection_matrix[1, 1] = -1
            calib = torch.Tensor(projection_matrix).float().cuda()
        
        save_gallery_path = os.path.join(gallery_id,name.split('/')[-2])
        os.makedirs(save_gallery_path,exist_ok=True)

        data={'name': name,
            'img': img,
            'calib': calib.unsqueeze(0),
            'mask': None,
            'b_min': B_MIN,
            'b_max': B_MAX,
            'origin_calib':origin_calib,
            'crop_img':crop_imgs,
            'crop_query_points':crop_query_points
        }
        with torch.no_grad():
            gen_mesh_sdf(cfg,model,data,save_gallery_path)


def test_epoch(model, cfg, args, test_loader, epoch,gallery_id):
    '''test epoch
    Parameters:
        model:
        cfg:
        args:
        test_loader:
        epoch: current epoch
        gallery_id: gallery save path
    Return:
        test_metrics
    '''
    model.eval()
    
    iou_metrics = AverageMeter()
    prec_metrics = AverageMeter()
    recall_metrics = AverageMeter()
    error_metrics =AverageMeter()
    epoch_start_time = time.time()
    
    with torch.no_grad():
        for idx,data in enumerate(test_loader):  
            image_tensor = data['img'].cuda()
            calib_tensor = data['calib'].cuda()
            sample_tensor = data['samples'].cuda()
            label_tensor = data['labels'].cuda()

            bs = image_tensor.shape[0]
            res, error = model(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)

            IOU, prec, recall = compute_acc(res,label_tensor)
            if args.dist:
                error = reduce_tensor(error)
                IOU = reduce_tensor(IOU)
                prec =  reduce_tensor(prec)
                recall =  reduce_tensor(recall) 
            error_metrics.update(error.item(),bs)
            iou_metrics.update(IOU.item(),bs)
            prec_metrics.update(prec.item(),bs)
            recall_metrics.update(recall.item(),bs)

            iter_net_time = time.time()
            eta = int(((iter_net_time - epoch_start_time) / (idx + 1)) * len(test_loader) - (
                    iter_net_time - epoch_start_time))
            
            word_handler = 'Test: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | IOU: {5:.06f}  | prec: {6:.05f} | recall: {7:.05f} | ETA: {8:02d}:{9:02d}:{10:02d}'.format( 
                cfg.name,epoch,idx,len(test_loader),error_metrics.avg,iou_metrics.avg, 
                prec_metrics.avg,recall_metrics.avg, 
                int(eta//3600),int((eta%3600)//60),eta%60)
            if (not args.dist) or dist.get_rank() == 0:
                logger.info(word_handler)
        logger.info("Test Final result | Epoch: {0:d} | Err: {1:.06f} | IOU: {2:.06f}  | prec: {3:.05f} | recall: {4:.05f} |".format(
            epoch, error_metrics.avg, iou_metrics.avg, prec_metrics.avg, recall_metrics.avg
            ))
    return dict(error=error_metrics.avg,iou=iou_metrics.avg,recall = recall_metrics.avg,pre =prec_metrics.avg)


# gen_residual_sdf
def gen_res_sdf(model, cfg, args, test_loader, epoch, gallery_id, gen_num):
    model.eval()
    watched = 0
    for batch in test_loader:
        watched+=1
        if watched>gen_num:
            break
        logger.info("time {}/{}".format(watched, gen_num))
        
        image_tensor = batch['img'].cuda()
        samples_tensor = batch['samples'].cuda()
        calib_tensor = batch['calib'].cuda()
        
        with torch.no_grad():
            model.extract_features(image_tensor)
            model.query(samples_tensor, calib_tensor)
            
            pred_sdf_values = model.get_preds()[0][0].detach().cpu().numpy().astype(np.float32)
        
        # save
        dirname = os.path.dirname(batch['name'][0])
        gt_sdf_values = batch['labels'].squeeze().numpy().astype(np.float32)
        res_sdf_values = gt_sdf_values - pred_sdf_values
        
        np.save(dirname + "/sdf_%s_res_values.npy" % cfg.phase, res_sdf_values)
        

    
