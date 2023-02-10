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


@DATASETS.register_module
class SDFDataset(Dataset):
    #Note that __B_MIN and __B_max is means that bbox of valid sample zone for all images, unit cm
    __B_MIN = np.array([-1., -1., -1.])
    __B_MAX = np.array([+1., +1., +1.])
    def __init__(self,input_dir, sd_type, 
                 pipeline = None, is_train = True, projection_mode = 'orthogonal', 
                 random_multiview = False, img_size = 512, num_views = 1, num_sample_points = 5000, 
                 num_sample_color = 0, sample_sigma = 0.5, check_occ = 'trimesh', 
                 debug = False, span = 1, sample_aim = 0.5, semantic_input = False, 
                 crop_windows = 512, test = False, gen = False):
        '''
        SDFDataset (TO UPDATE)
        Parameters:
            input_dir: file direction e.g. Garmant/render_people_gen， in this file you have some subfile direction e.g. rp_kai_posed_019_BLD
            pipeline: the method which process datasets, like crop, ColorJitter and so on.
            is_train: phase the datasets' state
            projection_mode: orthogonal or perspective
            num_sample_points: the number of sample clounds from mesh 
            num_sample_color: the number of sample colors from mesh, default 0, means train shape model
            sample_sigma: the distance we disturb points sampled from surface. unit: cm e.g you wanna get 5cm, you need input 5
            check_occ: which method, you use it to check whether sample points are inside or outside of mesh. option: trimesh |
            debug: debug the dataset like project the points into img_space scape
            span: span step from 0 deg to 359 deg, e.g. if span == 2, deg: 0 2 4 6 ...,
            normal: whether, you want to use normal map, default False, if you want to train pifuhd, you need set it to 'True'
            sample_aim: set sample distance from mesh, according to PIFu it use 5 cm while PIFuhd choose 5 cm for coarse PIFu and 3 cm
                to fine PIFu
            fine_pifu: whether train fine pifu,
            crop_windows: crop window size using for pifuhd, default, 512
            test: whether it is test-datasets 
        Return:
            None
        '''
        super(SDFDataset,self).__init__()
        self.is_train = is_train
        self.projection_mode = projection_mode
        self.input_dir = input_dir
        self.sd_type = sd_type
        self.__name="SDFDataset"
        self.img_size = img_size
        self.num_views = num_views
        self.num_sample_points = num_sample_points
        self.num_sample_color = num_sample_color
        self.sigma = sample_sigma if type(sample_sigma) == list else [sample_sigma]
        self._get_infos()
        self.subjects = self.get_subjects()
        self.random_multiview = random_multiview
        self.check_occ =check_occ
        self.debug = debug
        self.span = span
        self.sample_aim = sample_aim
        self.crop_windows_size=crop_windows
        self.test = test
        self.gen = gen
        self.semantic_input = semantic_input
        self.semantic_dict = np.array([8, 4, 0, 4, 2, 2, 3, 3, 6, 6, 5, 1, 1, 7, 7, 6, 6, 6, 8])
        
        if self.test:
            self.num_sample_points = self.num_sample_points*5


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
            num_views = num_views,
            num_sample_points = num_sample_points,
            num_sample_color = num_sample_color,
            random_multiview = random_multiview,
            sample_sigma=sample_sigma,
            check_occ=check_occ,
            debug = debug,
            span = span,
            sample_aim = sample_aim,
            crop_windows_size = crop_windows,
            test = test 
        )

        #transform method or pipeline method
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def _get_infos(self):
        '''
        prepare for images-preprocessed
        '''
        input_list = os.listdir(self.input_dir)
        input_list = sorted(input_list)
        input_list = [os.path.join(self.input_dir, name) for name in input_list]
        
    def get_subjects(self):
        all_subjects = os.listdir(self.input_dir)
        if os.path.isfile(os.path.join(self.input_dir, 'val.txt')) is True:
            var_subjects = np.loadtxt(os.path.join(self.input_dir, 'val.txt'), dtype=str).tolist()
        else:
            var_subjects = []
            
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def __get_render(self, subject, sid=0, random_sample=False):
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
        
        # sey calibration data
        ortho_ratio = 1.0
        # world unit / model unit
        scale = 1.0
        # camera center world coordinate
        center = np.zeros(3, dtype=np.float64)
        # model rotation
        R = np.eye(3, dtype=np.float64)
        #translate the position of camera into world coordinate origin. 
        translate = -np.matmul(R, center).reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        
        # Match camera space to image pixel space
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = scale / ortho_ratio
        #render code, this part flip(axis =0),therefore, y need change
        scale_intrinsic[1, 1] = -scale / ortho_ratio
        scale_intrinsic[2, 2] = scale / ortho_ratio
        
        
        #uv space is [-1,1] we map [-256,255]->[-1,1]
        # Match image pixel space to image uv space
        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0# / float(self.img_size// 2)
        uv_intrinsic[1, 1] = 1.0# / float(self.img_size // 2)
        uv_intrinsic[2, 2] = 1.0# / float(self.img_size // 2)
        
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
        'mesh_path': os.path.join(self.input_dir, subject, "OBJ", "mesh.obj"),
        'sid': index,
        }
        
        #get render image
        render_data = self.__get_render(subject, sid=index,
                                        random_sample=self.random_multiview)
        res.update(render_data)
            
        if self.num_sample_points:
            
            # read sample points and labels
            if self.sd_type == 'base':
                surf_posi = np.load(os.path.join(self.input_dir, subject, "surf_posi.npy"))
                surf_sd_value = np.load(os.path.join(self.input_dir, subject, "sdf_base_values.npy"))
            elif self.sd_type == 'delta':
                surf_posi = np.load(os.path.join(self.input_dir, subject, "surf_posi.npy"))
                surf_sd_value = np.load(os.path.join(self.input_dir, subject, "sdf_delta_values.npy"))
            elif self.sd_type == 'tier0':
                surf_posi = np.load(os.path.join(self.input_dir, subject, "surf_posi_0p04.npy"))
                surf_sd_value = np.load(os.path.join(self.input_dir, subject, "sdf_values_0p04.npy"))
            elif self.sd_type == 'tier1':
                surf_posi = np.load(os.path.join(self.input_dir, subject, "surf_posi_0p04.npy"))
                surf_sd_value = np.load(os.path.join(self.input_dir, subject, 
                                                     "sdf_tier0_res_values.npy"))
            else:
                print("Error: no sd_type matched: %s" % self.sd_type)
                return False
            
            
            if self.gen is False:
                # random select
                surf_point_num = surf_posi.shape[1]
                random_select = np.random.randint(low = 0, high = surf_point_num, 
                                                  size = self.num_sample_points)
                samples = surf_posi[:, random_select].astype(np.float32)/128 - 1
                labels = np.expand_dims(surf_sd_value[random_select], 0).astype(np.float32)
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

