'''
@author: lingteng qiu
@data  : 2021-1-21
@emai: lingtengqiu@link.cuhk.edu.cn
RenderPeople Dataset:https://renderpeople.com/
'''
import cv2
import sys
sys.path.append("./")
from torch.utils.data import Dataset
import json
import os
import numpy as np
import random
import torch
import scipy.sparse as sp
from .pipelines import Compose
from .registry import DATASETS
import warnings
from PIL import Image,ImageOps
import torchvision.transforms as transforms
import trimesh
import numpy as np
from tqdm import tqdm
import logging
import torch.nn.functional as F
logger = logging.getLogger('logger.trainer')

# if not, cv2 multi-thread may lock with pytorch multi-thread
cv2.setNumThreads(0)
#cv2.ocl.setUseOpenCL(False)


@DATASETS.register_module
class FSSDFDataset(Dataset):
    #Note that __B_MIN and __B_max is means that bbox of valid sample zone for all images, unit cm
    __B_MIN = np.array([-1., -1., -1.])
    __B_MAX = np.array([+1., +1., +1.])
    def __init__(self, input_dir, sd_type, 
                 pipeline = None, is_train = True, projection_mode = 'orthogonal', 
                 img_size = 512, num_surf_points = 5000, num_uniform_points = 0, sample_sigma = 0.5, 
                 debug = False, span = 1, semantic_input = False, test = False, gen = False):
        '''
        FSSDFDataset FaceSwap SDF dataset
        Parameters: (TO UPDATE) 
            input_dir: file direction e.g. Garmant/render_people_gen， in this file you have some subfile direction e.g. rp_kai_posed_019_BLD
            pipeline: the method which process datasets, like crop, ColorJitter and so on.
            is_train: phase the datasets' state
            projection_mode: orthogonal or perspective
            num_surf_points: the number of sampled points near the mesh surface
            num_uniform_points: the number of uniform sampled points in the volume space
            sample_sigma: the distance we disturb points sampled from surface. unit: cm e.g you wanna get 5cm, you need input 5
            debug: debug the dataset like project the points into img_space scape
            test: whether it is test-datasets
        Return:
            None
        '''
        super(FSSDFDataset,self).__init__()
        self.is_train = is_train
        self.projection_mode = projection_mode
        self.input_dir = input_dir
        self.sd_type = sd_type
        self.__name="FSSDFDataset"
        self.img_size = img_size
        self.num_surf_points = num_surf_points
        self.num_uniform_points = num_uniform_points
        self.sigma = sample_sigma if type(sample_sigma) == list else [sample_sigma]
        self.subjects = self.get_subjects()
        self.debug = debug
        self.test = test
        self.gen = gen
        self.semantic_input = semantic_input
        self.semantic_dict = np.array([8, 4, 0, 4, 2, 2, 3, 3, 6, 6, 5, 1, 1, 7, 7, 6, 6, 6, 8])
        self.cheek_params = np.load(input_dir[:-1] + "_cheek_params.npy")
        
        if self.test:
            self.num_surf_points = self.num_surf_points*5
        
        if not pipeline == None:
            #color ColorJitter,blur,crop,resize,totensor,normalize .....
            self.transformer  = Compose(pipeline)
        else:
            self.transformer = None

        self.input_para=dict(
            input_dir=input_dir,
            is_train=is_train,
            projection_mode = projection_mode,
            pipeline = self.transformer,
            img_size = img_size,
            num_surf_points = num_surf_points,
            sample_sigma=sample_sigma,
            debug = debug,
            test = test 
        )
        
        #transform method or pipeline method
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def get_subjects(self):
        train_idx_pool = np.load(self.input_dir[:-1] + "_idx_pool_train.npy")
        test_idx_pool = np.load(self.input_dir[:-1] + "_idx_pool_test.npy")
        
        train_subjects = ["%08d" % i for i in train_idx_pool]
        test_subjects = ["%08d" % i for i in test_idx_pool]
        
        if self.is_train:
            return train_subjects
        else:
            return test_subjects

    def __get_render(self, subject, random_sample=False):
        '''
        Gaining the render data
        Parameters:
            subject: subject name
            view_id: the first view_id. If None, select a random one.
            the sequence of pid,pid and sidthe: para and so on 
        Return:
            value that should contain some key as following
            'name': img_name e.g  savepath/rp_fernanda_posed_021/34_0_00.jpg
            'img': [1, C, W, H] images
            'calib': [1, 4, 4] calibration matrix
            'extrinsic': [1, 4, 4] extrinsic matrix
            'mask': [1, 1, W, H] masks
        '''

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []
        
        extrinsic = np.identity(4)
        
        # Match camera space to image pixel space
        scale_intrinsic = np.identity(4)
        #render code, this part flip(axis =0),therefore, y need change
        scale_intrinsic[1, 1] = -1
                
        #uv space is [-1,1] we map [-256,255]->[-1,1]
        # Match image pixel space to image uv space
        uv_intrinsic = np.identity(4)
        
        # Transform under image pixel space
        trans_intrinsic = np.identity(4)
        
        mask = cv2.imread(os.path.join(self.input_dir, subject, 'face_mask.png'))
        render = cv2.imread(os.path.join(self.input_dir, subject, 'color_img.png'))
        if self.semantic_input is True:
            sem_mask = cv2.imread(os.path.join(self.input_dir, subject, 'semantic_mask.png'), 0)
            sem_mask = self.semantic_dict[np.expand_dims(sem_mask, 2)].astype(np.uint8)*20
            render = np.concatenate((render, sem_mask), 2)
        
        data = {
            'img':render,
            'mask':mask,
            'scale_intrinsic':scale_intrinsic,
            'trans_intrinsic':trans_intrinsic,
            'uv_intrinsic':uv_intrinsic,
            'extrinsic':extrinsic,
            'flip': False
        }
        
        if self.transformer is not None:
            data = self.transformer(data)
        else:
            raise NotImplementedError
        render = data['img']
        mask = data['mask']
        calib = data['calib']
        extrinsic = data['extrinsic']
        
        mask_list.append(mask)
        if len(mask.shape)!=len(render.shape):
            mask = mask.unsqueeze(-1)
        
        if self.semantic_input is True:
            mask = torch.stack((mask[0,:,:],)*4, 0)
        
        render = mask.expand_as(render) * render
        render_list.append(render)
        calib_list.append(calib)
        extrinsic_list.append(extrinsic)
        
        return {'name':os.path.join(self.input_dir, subject, 'color_img.png'),
                'img': torch.stack(render_list, dim=0),
                'calib': torch.stack(calib_list, dim=0),
                'extrinsic': torch.stack(extrinsic_list, dim=0),
                'mask': torch.stack(mask_list, dim=0),
                }
    
    #*********************property********************#

    @property
    def B_MAX(self):
        return self.__B_MAX
    @property
    def B_MIN(self):
        return self.__B_MIN
    
    def __getitem__(self, index):
        '''
        Capturing data from datasets according to input index
        Parameters:
            index: type(int)(0-len(self))
        return:
        '''
        
        subject = self.subjects[index]
        res = {
        'name': subject,
        'sid': index,
        }
        
        #get render image
        render_data = self.__get_render(subject, random_sample=False)
        res.update(render_data)
        
        if (self.num_surf_points + self.num_uniform_points) > 0:
            
            # read sample points and labels
            if self.sd_type == 'tier0':
                surf_posi = np.load(os.path.join(self.input_dir, subject, "surf_posi.npy"))
                surf_sd_value = np.load(os.path.join(self.input_dir, subject, "surf_sdf_value.npy"))
            elif self.sd_type == 'tier1':
                surf_posi = np.load(os.path.join(self.input_dir, subject, "surf_posi.npy"))
                surf_sd_value = np.load(os.path.join(self.input_dir, subject, "sdf_tier0_res_values.npy"))
            else:
                print("Error: no sd_type matched: %s" % self.sd_type)
                return False
            
            if self.gen is False:
                # ===== random select from only surface points =====
                surf_point_num = surf_posi.shape[1]
                random_select = np.random.randint(low = 0, high = surf_point_num, 
                                                  size = self.num_surf_points)
                samples = surf_posi[:, random_select].astype(np.float32)/128 - 1
                labels = np.expand_dims(surf_sd_value[random_select], 0).astype(np.float32)

                # # ===== (1) random select surface points =====
                # surf_point_num = surf_posi.shape[1]
                # random_select = np.random.randint(low = 0, high = surf_point_num, 
                #                                   size = self.num_surf_points)
                # samples_surf = surf_posi[:, random_select].astype(np.float32)/128 - 1
                # labels_surf = np.expand_dims(surf_sd_value[random_select], 0).astype(np.float32)

                # # ===== (2) random select uniform points =====
                # cheek_param = self.cheek_params[int(subject)]
                # cheek_center, cheek_normal = cheek_param[:3], cheek_param[3:]
                # cheek_center[2] = 0
                
                # # generate random points list
                # samples_uniform = np.random.random((self.num_uniform_points, 3))
                # samples_uniform = (samples_uniform * 2) - 1

                # # compute point-to-plane distance
                # diff_list = samples_uniform - cheek_center
                # labels_uniform = np.expand_dims(np.dot(diff_list, cheek_normal) / np.linalg.norm(diff_list, axis = 1), 
                #                                 axis = 0).astype(np.float32)
                # samples_uniform = samples_uniform.T
                # samples_uniform[samples_uniform>0.04] = 0.04
                # samples_uniform[samples_uniform<-0.04] = -0.04

                # # ===== merge (1) and (2) =====
                # samples = np.concatenate((samples_surf, samples_uniform), axis = 1)
                # labels = np.concatenate((labels_surf, labels_uniform), axis = 1)
                
                # # ===== randomly shuffle (1) and (2) =====
                # merge_tuple = np.concatenate((samples, labels), axis = 0)
                # np.random.shuffle(merge_tuple)
                # samples = merge_tuple[:3, :]
                # labels = merge_tuple[3:4, :]

            else: # gen not random
                samples = surf_posi[:, :].astype(np.float32)/128 - 1
                labels = np.expand_dims(surf_sd_value[:], 0).astype(np.float32)
            
            sample_data = {'samples': torch.Tensor(samples).float(), 
                           'labels': torch.Tensor(labels).float()}
            
            res.update(sample_data)
        
        
        if self.debug:
            self.__debug(res,sample_data)
        
        return res


    def __len__(self):   
        return len(self.subjects)

