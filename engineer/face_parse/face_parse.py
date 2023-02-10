import torch, os, cv2, numpy as np, tqdm, torchvision.transforms as transforms
from model import BiSeNet

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

    # correct eyebrows and eyes
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


def face_parse(img, cp='79999_iter.pth'):
    
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = os.path.join('./checkpoints/face_parse/', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        
        if img.shape != (512, 512, 3):
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        img_t = to_tensor(img.copy())
        img_t = torch.unsqueeze(img_t, 0)
        img_t = img_t.cuda()
        out = net(img_t)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        vis_im = vis_parsing_maps(512, 512, img, parsing, stride=1)
        vis_im_label = parsing_label2celeba(parsing_Color2label(vis_im))
    
    return vis_im_label



def face_parse_batch(fn_list, tgt_dir, cp='79999_iter.pth', new_folder=False):
    
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = os.path.join('./checkpoints/face_parse/', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        
        for fn in tqdm.tqdm(fn_list, desc="face parse"):
            img = cv2.imread(fn)
            if img.shape != (512, 512, 3):
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            img = img[:,:,::-1]
            
            img_t = to_tensor(img.copy())
            img_t = torch.unsqueeze(img_t, 0)
            img_t = img_t.cuda()
            out = net(img_t)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            vis_im = vis_parsing_maps(512, 512, img, parsing, stride=1)
            vis_im_label = parsing_label2celeba(parsing_Color2label(vis_im))

            src_name = os.path.splitext(os.path.basename(fn))[0]
            img = img[:,:,::-1]
            if new_folder is True:
                os.makedirs(tgt_dir + "/%s/" % src_name, exist_ok = True)
                cv2.imwrite(tgt_dir + "/%s/%s_semask.png" % (src_name, src_name), vis_im_label)
            else:
                cv2.imwrite(tgt_dir + "/%s_semask.png" % src_name, vis_im_label)
    return True


