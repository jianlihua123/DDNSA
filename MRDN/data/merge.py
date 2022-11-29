import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import random
import copy
import tqdm


def merge(cartoon, texture, low, high):
    merge = copy.deepcopy(cartoon)
    t = np.zeros((merge.shape[0], merge.shape[1]))
    avg = int(np.mean(np.mean(cartoon))) * (random.random()*1.2+0.2)
    for i in range(cartoon.shape[0]):
        for j in range(cartoon.shape[1]):
            u = i%texture.shape[0]
            v = j%texture.shape[1]
            if texture[u,v] == 255:
                sum = 0.
                for c in range(3):
                    # maximum = (255-cartoon[i, j, c])
                    maximum = avg
                    merge[i, j, c] = random.randint(int(low * (maximum)), int(high * (maximum)))
                    sum += merge[i, j, c]
                t[i, j] = sum/3
            else:
                t[i, j] = texture[u, v]
    return merge, t


def distortion(img, degree=0.2):
    u, v = img.shape[:2]

    def f(i, j):
        return i + degree * np.sin(2 * np.pi * j)

    def g(i, j):
        return j + degree * np.cos(3 * np.pi * i)

    r = np.zeros((u, v)).astype('uint8')
    for i in range(u):
        for j in range(v):
            i0 = i / u
            j0 = j / v
            u0 = int(f(i0, j0) * u)%u
            v0 = int(g(i0, j0) * v)%v
            r[i, j] = img[u0, v0]
    return r


def main():
    cartoon_path = '/SSD64/Smooth/train/cartoon'
    texture_path = '/SSD64/Smooth/train/texture_f'
    save_path = '/SSD64/Smooth/train/GEN'
    gt_path = save_path+'/'+'GT'
    t_path = save_path+'/'+'T'
    s_path = save_path+'/'+'S'
    input_path = save_path+'/'+'Input'

    if not os.path.isdir(gt_path):
        os.mkdir(gt_path)
    if not os.path.isdir(t_path):
        os.mkdir(t_path)
    if not os.path.isdir(s_path):
        os.mkdir(s_path)
    if not os.path.isdir(input_path):
        os.mkdir(input_path)

    low = 0.75
    high = 1.
    resize_low = 0.3

    for cartoon_name in tqdm.tqdm(os.listdir(cartoon_path)):
        for texture_name in os.listdir(texture_path):
            cartoon_image_path = cartoon_path+'/'+cartoon_name
            texture_image_path = texture_path+'/'+texture_name

            cartoon_image = Image.open(cartoon_image_path)
            cartoon_image = np.asanyarray(cartoon_image)
            texture_image = Image.open(texture_image_path).convert('L')
            texture_image = np.asanyarray(texture_image)

            _, texture_image = cv2.threshold(texture_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # resize texture image
            r_size = random.randint(int(resize_low*texture_image.shape[0]), texture_image.shape[0])
            texture_image = cv2.resize(texture_image, (r_size, r_size))

            # flip texture image
            flag = random.randint(0, 1)
            texture_image = np.flip(texture_image, flag)

            # rotate texture image
            flag = random.randint(0, 3)
            rows, cols = texture_image.shape
            M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90*flag, 1)
            texture_image = cv2.warpAffine(texture_image, M, (cols, rows))

            merged_image, t = merge(cartoon_image, texture_image, low, high)
            merged_image = Image.fromarray(merged_image)

            merge_name = cartoon_name.split('.')[0]+'_'+texture_name.split('.')[0]+'.png'
            merged_image.save(input_path+'/'+merge_name)
            t = Image.fromarray(t)
            t = t.convert('L')
            t.save(t_path+'/'+merge_name)

            cartoon_image = Image.fromarray(cartoon_image)
            cartoon_image.save(gt_path+'/'+merge_name)
            cartoon_image = np.asanyarray(cartoon_image.convert('L'))
            _, s = cv2.threshold(cartoon_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            s = Image.fromarray(s)
            s.save(s_path+'/'+merge_name)

    a = 0


if __name__ == '__main__':
    main()
