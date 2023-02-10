
import numpy as np
import cv2
import glob
from PIL import Image


def normalize_SEAN(img):
    scale = 1.1
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    res = []

    if len(img.shape) == 2:
        res = np.zeros((512, 512), dtype=np.uint8)
        left = img.shape[0] // 2 - 256
        top = max(0, img.shape[0] // 2 - 256 - 20)
        res[:, :] = img[top:top + 512, left:left + 512]

    elif len(img.shape) == 3 and img.shape[2] == 3:
        res = np.ones((512, 512, 3), dtype=np.uint8) * 255
        left = img.shape[0] // 2 - 256
        top = max(0, img.shape[0] // 2 - 256 - 20)
        res[:, :, :] = img[top:top + 512, left:left + 512, :]

    return res

def parsing_Color2label(img):
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
    map_list = [0, 1, 6, 7, 4, 5, 3, 8, 9, 15, 2, 10, 11, 12, 17, 16, 18, 13, 14]
    res = label.copy()
    for i in range(0, len(map_list)):
        index = np.where(label == i)
        res[index[0], index[1]] = map_list[i]

    return res

def celeba_label2color(label):
    color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0],
                  [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
                  [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
                  [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
                  [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    res = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(color_list):
        res[label == idx] = color

    return res

def celeba_color2label(img):
    color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0],
                  [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
                  [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
                  [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
                  [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(0, len(color_list)):  # len(colors)
        color = color_list[i]
        index = np.where(np.all(img == color, axis=-1))
        label[index[0], index[1]] = i

    return label

def parsing_label2color(label):
    color_list = [[0, 0, 0], [255, 0, 0], [150, 30, 150], [255, 65, 255],
                  [150, 80, 0], [170, 120, 65], [220, 180, 210], [255, 125, 125],
                  [200, 100, 100], [215, 175, 125], [125, 125, 125], [255, 150, 0],
                  [255, 255, 0], [0, 255, 255], [255, 225, 120], [125, 125, 255],
                  [0, 255, 0], [0, 0, 255], [0, 150, 80]
                  ]
    res = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(color_list):
        res[label == idx] = color

    return res


def celeba_label2parsinglabel(label):
    map_list = [0, 1, 6, 7, 4, 5, 3, 8, 9, 15, 2, 10, 11, 12, 17, 16, 18, 13, 14]
    res = label.copy()
    for i in range(0, len(map_list)):
        index = np.where(label == map_list[i])
        res[index[0], index[1]] = i

    return res



def combineImg(im1, im2, p1):
    res = cv2.addWeighted(im1, p1, im2, 1-p1, 0)
    return res


# if __name__ == "__main__":
#     dir1 = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0308/CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
#     dir2 = "/home/zhang/zydDataset/faceRendererData/temp_imgs/stylegan_data_semantic/"
#     lis = sorted(glob.glob(dir1 + "*.png"))
#
#     for i in range(0, len(lis)):
#         name1 = lis[i]
#         name2 = name1.replace(dir1, dir2)
#
#         im1 = cv2.imread(name1)
#         im1 = cv2.resize(im1, (0, 0), fx=2, fy=2)
#         im2 = cv2.imread(name2)
#         res = combineImg(im1, im2, 0.5)
#
#         name3 = name2.replace("stylegan_data_semantic", "stylegan_data_combine2")
#         cv2.imwrite(name3, res)
#
#         print(i)


# if __name__ == "__main__":
#     lis = sorted(glob.glob("/home/zhang/zydDataset/faceRendererData/temp_imgs/0309/stylegan_data_part/*.png"))
#     for i in range(0, len(lis)):
#         name1 = lis[i]
#         name2 = name1.replace("stylegan_data_part", "stylegan_data_semantic")
#         im1 = cv2.imread(name1)
#         im1 = cv2.resize(im1, (0, 0), fx=0.5, fy=0.5)
#         im2 = cv2.imread(name2)
#         res = combineImg(im1, im2, 0.5)
#
#         name3 = name1.replace("stylegan_data_part", "stylegan_data_combine")
#         cv2.imwrite(name3, res)


if __name__ == "__main__":
    lis = sorted(glob.glob("/home/zhang/zydDataset/faceRendererData/temp_imgs/0309/stylegan_data_part/*.png"))
    for i in range(0, len(lis)):
        name1 = lis[i]
        name2 = name1.replace("stylegan_data_part", "stylegan_data_semantic")
        im1 = cv2.imread(name1)
        im2 = cv2.imread(name2)[:, :, ::-1]

        label = parsing_label2celeba(parsing_Color2label(im2))
        rgb = celeba_label2color(label)

        name_label = name1.replace("stylegan_data_part", "stylegan_data_label")
        name_rgb = name1.replace("stylegan_data_part", "stylegan_data_color")
        Image.fromarray(label).save(name_label)
        Image.fromarray(rgb).save(name_rgb)
        print(i)


# if __name__ == "__main__":
#     lis = sorted(glob.glob("/home/zhang/zydDataset/faceRendererData/temp_imgs/0309/stylegan_data_part/*.png"))
#     for i in range(0, len(lis)):
#         name1 = lis[i]
#         name2 = name1.replace("stylegan_data_part", "stylegan_data_jpg")
#         name2 = name2.replace(".png", ".jpg")
#         im1 = cv2.imread(name1)
#
#         cv2.imwrite(name2, im1)
#
#         print(i)


# if __name__ == "__main__":
#     dir1 = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0308/CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
#     dir2 = "/home/zhang/zydDataset/faceRendererData/temp_imgs/stylegan_data_label/"
#     dir3 = "/home/zhang/zydDataset/faceRendererData/temp_imgs/stylegan_data_part/"
#     lis = sorted(glob.glob(dir1 + "*.png"))
#
#     for i in range(0, len(lis)):
#         name1 = lis[i]
#         name2 = name1.replace(dir1, dir2)
#         name3 = name1.replace(dir1, dir3)
#
#         im1 = cv2.imread(name1)     # syn
#         im1 = cv2.resize(im1, (0, 0), fx=4, fy=4)
#         im2 = cv2.imread(name2, 0)   # label
#         im2 = cv2.resize(im2, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
#         im3 = cv2.imread(name3)      # raw
#
#         index = np.where(im2 == 4)
#         im3[index[0], index[1], :] = im1[index[0], index[1], :]
#
#         index = np.where(im2 == 5)
#         im3[index[0], index[1], :] = im1[index[0], index[1], :]
#
#         name3 = name2.replace("stylegan_data_label", "stylegan_data_eye")
#         cv2.imwrite(name3, im3)
#
#         print(i)

