import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from skimage.measure import compare_ssim
from util.measure import PSNR, SSIM
from PIL import Image

if __name__ == '__main__':
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    model_path = '/home/server/LihuaJian/result'
    model_name = 'best_99'
    model = torch.load(model_path+'/'+model_name + '.pth')["model"]
    model.eval()

    datasets_path = '/media/server/80SSD/LihuaJian/test/10/Input'
    GTs_path = '/media/server/80SSD/LihuaJian/test/10/GT'
    datasets = ['10']

    result_path = '/media/server/80SSD/LihuaJian/result' + '/' + model_name
    results = {}

    for dataset in datasets:
        path = datasets_path + '/' + dataset
        GT_path = GTs_path+'/'+dataset

        save_path = result_path+'/'+dataset
        if not os.path.exists(save_path):  # 如果路径不存在
            os.makedirs(save_path)
        image_list = glob.glob(path + "/*.*")
        gt_list = glob.glob(GT_path + '/*.*')
        assert len(gt_list) == len(image_list)

        avg_elapsed_time = 0.0
        avg_psnr_predicted = 0.0
        avg_ssim_predicted = 0.0
        gray_image = False

        for idx, image_name in enumerate(image_list):
            print("Processing ", image_name)
            im_input = plt.imread(image_name)
            gt = plt.imread(gt_list[idx])

            if len(im_input.shape) == 2:
                gray_image = True
                im_input = np.reshape(im_input, [im_input.shape[0], im_input.shape[1], 1])
                im_input = np.concatenate((im_input, im_input, im_input), 2)

            im_input = np.transpose(im_input, [2, 0, 1])
            im_input = im_input[np.newaxis, :]

            im_input = torch.from_numpy(im_input).float()
            if torch.max(im_input) > 10:
                im_input = im_input/255.

            if cuda:
                model = model.cuda()
                im_input = im_input.cuda()
            else:
                model = model.cpu()

            start_time = time.time()
            gt_o = model(im_input)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            gt_o = gt_o.cpu()
            gt_o = gt_o.data[0].numpy().astype(np.float32)

            gt_o = gt_o * 255.
            gt_o[gt_o < 0] = 0
            gt_o[gt_o > 255.] = 255.
            gt_o = gt_o.astype('uint8')

            gt = gt * 255.
            gt[gt < 0] = 0
            gt[gt > 255.] = 255.
            gt = gt.astype('uint8')

            gt_o = np.transpose(gt_o, [1, 2, 0])
            if gray_image:
                gt_o = Image.fromarray(gt_o)
                gt_o = gt_o.convert('L')
                gt_o = np.asarray(gt_o)

            psnr_predicted = PSNR(gt, gt_o, shave_border=0)
            avg_psnr_predicted += psnr_predicted

            ssim_predicted = SSIM(gt, gt_o, shave_border=0)
            avg_ssim_predicted += ssim_predicted

            name = image_name.split('/')[-1]
            name = name.split('.')[0]+'.png'

            gt_o = Image.fromarray(gt_o)
            gt_o.save(save_path + '/' + name)

        print("Dataset=", dataset)
        print("PSNR_predicted=", avg_psnr_predicted / len(image_list))
        print("SSIM predicted=", avg_ssim_predicted / len(image_list))
        print("It takes average {}s for processing".format(avg_elapsed_time / len(image_list)))

        results[dataset] = [avg_psnr_predicted / len(image_list)]
        results[dataset].append(avg_ssim_predicted / len(image_list))
    pd_results = pd.DataFrame(results, index=['PSNR', 'SSIM'], columns=datasets)
    pd_results.to_csv(result_path + '/' + 'result.csv', sep=',')
