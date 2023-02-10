# -*- encoding: utf-8 -*-
from model import BiSeNet

import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import glob
from utils import *

def parsing_Color2label(img):
    # convert the [face-parsing.Pytorch]-format RGB image to [face-parsing.Pytorch]-format labels (single channel).
    color_list = [[0, 0, 0], [255, 0, 0], [150, 30, 150], [255, 65, 255],
                  [150, 80, 0], [170, 120, 65], [220, 180, 210], [255, 125, 125],
                  [200, 100, 100], [215, 175, 125], [125, 125, 125], [255, 150, 0],
                  [255, 255, 0], [0, 255, 255], [255, 225, 120], [125, 125, 255],
                  [0, 255, 0], [0, 0, 255], [0, 150, 80]
                  ]

    label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(0, len(color_list)):  # len(colors)
        color = color_list[i]
        index = np.where(np.all(img == color, axis=-1))
        label[index[0], index[1]] = i

    return label

def parsing_label2celeba(label):
    # convert the [face-parsing.Pytorch]-format label image to [CelebAMask-HQ]-format label image
    map_list = [0, 1, 6, 7, 4, 5, 3, 8, 9, 15, 2, 10, 11, 12, 17, 16, 18, 13, 14]
    res = label.copy()
    for i in range(0, len(map_list)):
        index = np.where(label == i)
        res[index[0], index[1]] = map_list[i]

    return res

def vis_parsing_maps(h, w, im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [255, 0, 0], [150, 30, 150], [255, 65, 255],
                   [150, 80, 0], [170, 120, 65], [220, 180, 210], [255, 125, 125],
                   [200, 100, 100], [215, 175, 125], [125, 125, 125], [255, 150, 0],
                   [255, 255, 0], [0, 255, 255], [255, 225, 120],  [125, 125, 255],
                   [0, 255, 0], [0, 0, 255],  [0, 150, 80]
                   ]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno[vis_parsing_anno == 18] = 0   # hat

    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    # 新增的一段，用于纠正错误的左右眉毛和眼睛
    index_nose = np.where(vis_parsing_anno == 10)
    index_lefteb = np.where(vis_parsing_anno == 2)
    index_righteb = np.where(vis_parsing_anno == 3)
    index_lefteye = np.where(vis_parsing_anno == 4)
    index_righteye = np.where(vis_parsing_anno == 5)
    index_leftear = np.where(vis_parsing_anno == 7)
    index_rightear = np.where(vis_parsing_anno == 8)

    nose_x = np.mean(index_nose[1])
    if index_lefteb:
        ind_false = np.where(index_lefteb[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_lefteb[0][ind_false], index_lefteb[1][ind_false]] = 3

    if index_righteb:
        ind_false = np.where(index_righteb[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_righteb[0][ind_false], index_righteb[1][ind_false]] = 2

    if index_lefteye:
        ind_false = np.where(index_lefteye[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_lefteye[0][ind_false], index_lefteye[1][ind_false]] = 5

    if index_righteye:
        ind_false = np.where(index_righteye[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_righteye[0][ind_false], index_righteye[1][ind_false]] = 4

    if index_leftear:
        ind_false = np.where(index_leftear[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_leftear[0][ind_false], index_leftear[1][ind_false]] = 8

    if index_rightear:
        ind_false = np.where(index_rightear[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_rightear[0][ind_false], index_rightear[1][ind_false]] = 7

    for pi in range(0, num_of_class+1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    vis_im = vis_parsing_anno_color

    return vis_im


# 不带pose版本
def evaluate(respth='', dspth='', cp='79999_iter.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    image_list = sorted(glob.glob(dspth + "*.jpg")) + sorted(glob.glob(dspth + "*.png")) # 0517
    print('len_img:',len(image_list))
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for i in range(0, len(image_list)):   # len(image_list)
            image_path = image_list[i]
            # subdir = image_path.split("/")[-2]
            name = image_path.split("/")[-1]

            save_dir = respth
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            img = Image.open(image_path)
            h, w, c = np.array(img).shape
            image = img.resize((512, 512), Image.BILINEAR)
            
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            name2 = name

            if not os.path.exists(respth):   # 0313
                os.makedirs(respth)          # 0313
            vis_im = vis_parsing_maps(h, w, image, parsing, stride=1)
            print(i, image_path)

            vis_im_label = parsing_label2celeba(parsing_Color2label(vis_im))
            Image.fromarray(vis_im_label).save(osp.join(respth, name2[:-3]+"png"))        # 保存图像

if __name__ == "__main__":

#     dspth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/facescape_benchmark/" + \
#             "data/wild_img/"# 被检测的人脸彩色图片
#     respth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/facescape_benchmark/" + \ 
#              "data/wild_img_semantic_mask/" # 输出的label

#    dspth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/facescape_benchmark/" + \
#            "data/lab_img/"# 被检测的人脸彩色图片
#    respth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/facescape_benchmark/" + \
#             "data/lab_img_semantic_mask/" # 输出的label

#    dspth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/qual_eva_wild/images/"
#    respth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/qual_eva_wild/semantic_masks"

#    dspth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/micc_data/micc_img/"
#    respth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/micc_data/semantic_masks/"

#     dspth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/video_wild/clip_0/norm_frames/"
#     respth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/video_wild/clip_0/norm_semantic_masks/"

#    dspth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/video_wild/clip_obama/norm_frames/"
#    respth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/video_wild/clip_obama/norm_semantic_masks/"
#     dspth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/video_wild/clip_baijia/norm_frames/"
#     respth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/video_wild/clip_baijia/norm_semantic_masks/"    
#     evaluate(respth, dspth)
#    name_list = ['clip_persuade', 'clip_queen', 'clip_shen', 'clip_speak', 
#                 'clip_theresa', 'clip_trump', 'clip_willemijn']

#    for name in name_list:
#        dspth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/video_wild/%s/norm_frames/" % name
#        respth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/video_wild/%s/norm_semantic_masks/" % name
#        evaluate(respth, dspth)
#    dspth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/fake_sem_mask_test/ori_imgs/"
#    respth = "/media/hao/Document/Code/3dface_pred/hsdf_pred_test/fake_sem_mask_test/ori_semantic_mask/"    
#    evaluate(respth, dspth)
    dspth = "/media/hao/Document/Code/3dface_pred/data/no_swap_imgs/wild_face_train/"
    respth = "/media/hao/Document/Code/3dface_pred/data/no_swap_imgs/wild_face_train_semask/"    
    evaluate(respth, dspth)
    dspth = "/media/hao/Document/Code/3dface_pred/data/no_swap_imgs/wild_face_test/"
    respth = "/media/hao/Document/Code/3dface_pred/data/no_swap_imgs/wild_face_test_semask/"    
    evaluate(respth, dspth)


