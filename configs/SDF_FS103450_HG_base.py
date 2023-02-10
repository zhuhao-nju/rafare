'''
training coarse PIFu with gt normal map
'''
import numpy as np

"---------------------------- hsdf  options -----------------------------"
phase = "tier0"
semantic_input = True
"---------------------------- debug  options -----------------------------"
debug = False
"---------------------------- normal options -----------------------------"
fine_pifu = False

"----------------------------- Model options -----------------------------"
model = dict(
    PIFu=dict(
        type='PIFUNet', 
        head =dict(
        type='PIFuhd_Surface_Head',filter_channels=[256+1, 1024, 512, 256, 128, 1], merge_layer = 2, res_layers=[2, 3, 4], norm= None,last_op='none'),
        backbone=dict(
        type = 'Hourglass',num_stacks= 4,num_hourglass=2,norm='group',hg_down='ave_pool',hourglass_dim= 256, use_sem = True),
        depth=dict(
        type='DepthNormalizer',input_size = 512,z_size=200.0),
        projection_mode='orthogonal',
        error_term='mse'
    )
)
"----------------------------- Datasets options -----------------------------"
dataset_type = 'FSSDFDataset'
train_pipeline = [ dict(type='img_pad'),dict(type='flip',flip_ratio=0.5),dict(type='scale'),dict(type='random_crop_trans'), 
    dict(type='color_jitter',brightness=0., contrast=0., saturation=0.0, hue=0.0,keys=['img']),dict(type='resize',size=(512,512)),
    dict(type='to_camera'),dict(type='ImageToTensor',keys=['img','mask']),dict(type='normalize',mean=[0.5,0.5,0.5,0.0],std=[0.5,0.5,0.5,1.0])
    ,dict(type='ToTensor',keys=['calib','extrinsic']),
]

test_pipeline = [
    dict(type='resize',size=(512,512)),
    dict(type='to_camera'),
    dict(type='ImageToTensor',keys=['img','mask']),
    dict(type='normalize',mean=[0.5,0.5,0.5,0.0],std=[0.5,0.5,0.5,1.0]),
    dict(type='ToTensor',keys=['calib','extrinsic']),
]
data = dict(
    train=dict(
    type = "FSSDFDataset",
    input_dir = '/media/hao/RED2/Data/FaceSwap_3D_SDF/',
    sd_type = 'tier0',
    is_train = True,
    pipeline=train_pipeline,
    img_size = 1024,
    num_surf_points = 5000,
    num_uniform_points = 0,
    sample_sigma=[0.05,0.03],
    debug=debug,
    semantic_input = True,
    ),

    test=dict(
    type = "FSSDFDataset",
    input_dir = '/media/hao/RED2/Data/FaceSwap_3D_SDF/',
    sd_type = 'tier0',
    is_train = False,
    pipeline=test_pipeline,
    img_size = 1024,
    num_surf_points = 5000,
    num_uniform_points = 0,
    sample_sigma=[0.05,0.03],
    debug=debug,
    semantic_input = True,
    )
)
train_collect_fn = 'train_loader_collate_fn'
test_collect_fn = 'test_loader_collate_fn'
"----------------------------- checkpoints options -----------------------------"
checkpoints = "./checkpoints"
logger = True
"----------------------------- optimizer options -----------------------------"
optim_para=dict(
    optimizer = dict(type='RMSprop',lr=1e-3,momentum=0, weight_decay=0.0000),
)
"----------------------------- training strategies -----------------------------"
num_gpu = 1
lr_policy="stoneLR"
lr_warm_up = 1e-4
warm_epoch= 1
LR=1e-3
num_epoch= 30
batch_size = 12 # 8
test_batch_size = 1
scheduler=dict(
    gamma = 0.1,
    stone = [10] 
)
save_fre_epoch = 1
"----------------------------- evaluation setting -------------------------------"
val_epoch = 1
start_val_epoch = 0
"----------------------------- inference setting -------------------------------"
resolution = 256 #for inference
"-------------------------------- config name --------------------------------"
name='SDF_FS103450_HG_base'

"-------------------------------- render --------------------------------"
render_cfg = dict(
    type='Noraml_Render',
    width = 1024,
    height = 1024,
    render_lib ='face3d',
    flip =True
)
