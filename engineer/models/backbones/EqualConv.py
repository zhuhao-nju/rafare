'''
@author:lingteng qiu
@name:Open-Pifu
'''
import torch.nn as nn
import torch
from engineer.models.registry import BACKBONES
from .base_backbone import _BaseBackbone
import numpy as np
from ..common import ConvBlock, EqualConvBlock
import torch.nn.functional as F

@BACKBONES.register_module
class EqualConv(_BaseBackbone):
    '''
    EqualConv network uses EqualConv as the image filter.
    It does the following:
        EqualConv as a backbone for PIFu networks

    '''
    def __init__(self, norm:str ='group', 
                 hg_down:str='no_down', 
                 use_sem = False):
        '''
        Initial function of Hourglass
        Parameters:
            --num_hourglass, how many block of hourglass stack. Default =2
            --norm: which normalization you use? group for group normalization while
            batch for batchnormalization
            --hg_down', down sample method, type=str, default='ave_pool', help='ave pool || conv64 || conv128 || not_down'
        '''
        super(EqualConv, self).__init__()
        self.name = 'EqualConv Backbone'
        self.norm = norm
        self.hg_down = hg_down
        
        self.use_sem = use_sem
        
        self.input_para={'norm':norm, 
                         'hg_down':hg_down, 
                         'use_sem': use_sem}
        
        inc = 3
        if self.use_sem:
            inc+=1
        
        # backbone of resnet
        self.conv1 = nn.Conv2d(inc, 64, kernel_size=7, stride=2, padding=3)
        if self.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)
        
        if self.hg_down == 'conv64':
            self.conv2 = EqualConvBlock(64, 64, self.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.hg_down == 'conv128':
            self.conv2 = EqualConvBlock(64, 128, self.norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.hg_down == 'ave_pool' or self.hg_down == 'no_down':
            self.conv2 = EqualConvBlock(64, 128, self.norm)
        else:
            raise NameError('Unknown Fan Filter setting!')
        self.conv3 = EqualConvBlock(128, 128, self.norm)
        self.conv4 = EqualConvBlock(128, 256, self.norm)
        

    def forward(self,x:torch.Tensor):
        '''
        Parameters:
            X: Tensor[B,3,512,512] according to PIFu
        
        Return:
            features after Hourglass backbone
        '''

        x = F.relu(self.bn1(self.conv1(x)), True)
        #B,64,256,256
        tmpx = x

        if self.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        elif self.hg_down == 'no_down':
            x = self.conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')
        
        normx = x
        #B, 128,128,128

        x = self.conv3(x)
        x = self.conv4(x)
        outputs = [x]

        '''
        until now, 
        tmpx  ->[B, 64,256,256]
        normx ->[B,128,128,128]
        x     ->[B,128,128,128]
        '''
        
        return outputs, tmpx.detach(), normx



    


