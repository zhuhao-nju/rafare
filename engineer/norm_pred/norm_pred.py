import torch, numpy as np, cv2, os, tqdm
from options.test_options import TestOptions as opt_norm
from models.models import create_model as build_norm_model
from util.util import tensor2im

def norm_pred(src_img, front):

    opt_norm_pred = opt_norm().parse(save=False)
    if front is True:
        opt_norm_pred.name = "norm_pred_front"
    else:
        opt_norm_pred.name = "norm_pred_back"
    opt_norm_pred.nThreads = 1   # test code only supports nThreads = 1
    opt_norm_pred.batchSize = 1  # test code only supports batchSize = 1
    opt_norm_pred.serial_batches = True  # no shuffle
    opt_norm_pred.no_flip = True  # no flip
    opt_norm_pred.netG = "local"
    opt_norm_pred.ngf = 32
    opt_norm_pred.resize_or_crop = "none"
    opt_norm_pred.label_nc = 0
    opt_norm_pred.no_instance = True
    #opt_norm_pred.verbose = False

    # # test
    model = build_norm_model(opt_norm_pred)

    empty_image = torch.from_numpy(np.zeros((1, 1, 512, 512)).astype(np.uint8)).cuda()
    empty_inst = torch.from_numpy(np.zeros((1, 1, 512, 512)).astype(np.uint8)).cuda()

    src_img = src_img.astype(np.float32) / 128 - 1
    src_img = src_img[:,:,[2,1,0]]
    src_img_b = np.expand_dims(src_img.transpose((2,0,1)), axis = 0)
    input_label = torch.from_numpy(src_img_b).cuda()
    generated = model.inference(input_label, empty_inst, empty_image)
    result_norm = tensor2im(generated.data[0])
    return result_norm

def norm_pred_batch(fn_list, tgt_dir, front, new_folder=False):

    opt_norm_pred = opt_norm().parse(save=False)
    if front is True:
        opt_norm_pred.name = "norm_pred_front"
        desc = "norm pred front"
    else:
        opt_norm_pred.name = "norm_pred_back"
        desc = "norm pred back"

    opt_norm_pred.nThreads = 1   # test code only supports nThreads = 1
    opt_norm_pred.batchSize = 1  # test code only supports batchSize = 1
    opt_norm_pred.serial_batches = True  # no shuffle
    opt_norm_pred.no_flip = True  # no flip
    opt_norm_pred.netG = "local"
    opt_norm_pred.ngf = 32
    opt_norm_pred.resize_or_crop = "none"
    opt_norm_pred.label_nc = 0
    opt_norm_pred.no_instance = True
    #opt_norm_pred.verbose = False

    # # test
    model = build_norm_model(opt_norm_pred)

    empty_image = torch.from_numpy(np.zeros((1, 1, 512, 512)).astype(np.uint8)).cuda()
    empty_inst = torch.from_numpy(np.zeros((1, 1, 512, 512)).astype(np.uint8)).cuda()

    for fn in tqdm.tqdm(fn_list, desc=desc):
        src_img = cv2.imread(fn)
        if src_img.shape != (512, 512, 3):
            src_img = cv2.resize(src_img, (512, 512), interpolation=cv2.INTER_LINEAR)

        src_img = src_img.astype(np.float32) / 128 - 1
        src_img = src_img[:,:,[2,1,0]]
        src_img_b = np.expand_dims(src_img.transpose((2,0,1)), axis = 0)
        input_label = torch.from_numpy(src_img_b).cuda()
        generated = model.inference(input_label, empty_inst, empty_image)
        result_norm = tensor2im(generated.data[0])
        
        src_name = os.path.splitext(os.path.basename(fn))[0]
        if new_folder is True:
            if front is True:
                cv2.imwrite(tgt_dir + "/%s/%s_norm_front.png" % (src_name, src_name), result_norm)
            else:
                cv2.imwrite(tgt_dir + "/%s/%s_norm_back.png" % (src_name, src_name), result_norm)
        else:
            if front is True:
                cv2.imwrite(tgt_dir + "/%s_norm_front.png" % src_name, result_norm)
            else:
                cv2.imwrite(tgt_dir + "/%s_norm_back.png" % src_name, result_norm)
    return True
