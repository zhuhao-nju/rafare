import os, cv2, numpy as np, tqdm
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from models.models import create_model
import util.util as util
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


ignore_list = opt.wild_ignore.split(' ')

# make testing list
raw_fn_list = os.listdir(opt.wild_dir)
fn_all_list = [fn for fn in raw_fn_list if fn[-4:]==".png" or fn[-4:]==".jpg" or fn[-4:]==".bmp"]
fn_list = []
for fn in fn_all_list:
    ignored = False
    for ignore_suffix in ignore_list:
        if fn[-len(ignore_suffix):]==ignore_suffix:
           ignored = True
    if ignored == False:
        fn_list.append(fn)

# # test
model = create_model(opt)

empty_image = torch.from_numpy(np.zeros((1, 1, 512, 512)).astype(np.uint8)).cuda()
empty_inst = torch.from_numpy(np.zeros((1, 1, 512, 512)).astype(np.uint8)).cuda()


for fn in tqdm.tqdm(fn_list):
    src_img = cv2.imread(opt.wild_dir + fn)
    src_img = cv2.resize(src_img, (512, 512))
    src_img = src_img.astype(np.float32) / 128 - 1
    src_img = src_img[:,:,[2,1,0]]
    src_img_b = np.expand_dims(src_img.transpose((2,0,1)), axis = 0)
    input_label = torch.from_numpy(src_img_b).cuda()
    generated = model.inference(input_label, empty_inst, empty_image)
    result_norm = util.tensor2im(generated.data[0])
    if opt.name.split('_')[2] == "front":
        cv2.imwrite(opt.wild_dir + fn[:-4] + "_norm_front.png", result_norm[:,:,[2, 1, 0]])
    elif opt.name.split('_')[2] == "back":
        cv2.imwrite(opt.wild_dir + fn[:-4] + "_norm_back.png", result_norm[:,:,[2, 1, 0]])
    else:
        print("cannot identify if '%s' is front or back" % opt.name)
