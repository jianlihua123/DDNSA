# -*- coding: utf-8 -*-
import os
import scipy.io as scio
from util.improcess import *
from scipy.misc import imread, imsave
from util.filters import SalWeights
from tqdm import tqdm
import torch
import numpy as np
# import matplotlib.pyplot as plt
from networks.RDN import Net
from PIL import Image


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def smooth(img, module):
    img_cuda = np.reshape(img, [img.shape[0], img.shape[1], 1])
    img_cuda = np.concatenate((img_cuda, img_cuda, img_cuda), 2)
    img_cuda = np.transpose(img_cuda, [2, 0, 1])
    img_cuda = img_cuda[np.newaxis, :]
    img_cuda = torch.from_numpy(img_cuda).float()
    img_cuda = img_cuda.cuda()
    str = module(img_cuda)
    str = str.cpu()
    str = str.data[0].numpy().astype(np.float32)
    str = str * 255.
    str[str < 0.] = 0
    str[str > 255.] = 255.
    str = np.transpose(str, [1, 2, 0])
    str = str.astype('uint8')
    str = Image.fromarray(str)
    str = str.convert('L')
    str = np.asarray(str)
    # plt.imshow(str_ir, cmap='gray')
    # plt.show()
    str = str.astype('float') / 255.
    return str


# patch parameter setting
patch_size = 9
overlap = 8


# paths of source images and results
pathIR = './dataset/sourceImg/IR/'
pathVIS = './dataset/sourceImg/VIS/'
pathOUT = './results/'
mkdir(pathOUT)

# load the first layer parameters of SSAE
weight = scio.loadmat('module/W3.mat')
W1 = weight['W3']
bias = scio.loadmat('module/b3.mat')
b1 = bias['b3']

# load module of smooth filter
module_path = './module/smooth'
module = Net()
module.load_state_dict(torch.load(module_path))
module.eval()
module = module.cuda()


for k in tqdm(range(1, 21)):
    if os.path.exists(pathOUT + str(k) + '.png') is True:
        print(pathOUT + str(k) + '.png is existed')
        continue

    # read two source images
    IR = imread(pathIR + str(k) + '.png')
    VIS = imread(pathVIS + str(k) + '.png')

    # normalized to [0, 1]
    IA = IR / 255.0
    IB = VIS / 255.0

    # Contrast Fusion, including structure and texture information

    str_ir = smooth(IA, module)
    str_vis = smooth(IB, module)

    # extract the texture information
    TexA = IA - str_ir
    TexB = IB - str_vis

    # VSM fusion rule
    P = SalWeights([IR, VIS])
    P1 = P[:, :, 0]
    P2 = P[:, :, 1]
    P1 = (P1 - P1.min()) / (P1.max() - P1.min())
    P2 = (P2 - P2.min()) / (P2.max() - P2.min())

    # extend the source images to extract the salient features using Autoencoders,
    # these feature can be served as the weights
    IR_ex = cv2.copyMakeBorder(IA, 4, 4, 4, 4, cv2.BORDER_REFLECT)  # extend the image to scan by per pixel
    VS_ex = cv2.copyMakeBorder(IB, 4, 4, 4, 4, cv2.BORDER_REFLECT)  # extend the image to scan by per pixel

    w, h = IR_ex.shape
    gridx = range(0, w - patch_size + 1, patch_size - overlap)
    gridy = range(0, h - patch_size + 1, patch_size - overlap)

    A = im_to_patch(IR_ex, patch_size, overlap)
    B = im_to_patch(VS_ex, patch_size, overlap)
    ret1 = np.zeros((1, A.shape[1]))
    ret2 = np.zeros((1, B.shape[1]))

    for i in range(A.shape[1]):
        input1 = np.reshape(A[:, i], (81, 1))
        input2 = np.reshape(B[:, i], (81, 1))
        a = sigmoid(np.dot(W1, input1) + b1)
        b = sigmoid(np.dot(W1, input2) + b1)

        ret1[:, i] = np.max(a)
        ret2[:, i] = np.max(b)

    ret1 = ret1.flatten()
    ret2 = ret2.flatten()

    S1 = ret1.reshape(len(gridx), len(gridy))
    S2 = ret2.reshape(len(gridx), len(gridy))

    S1 = (S1 - S1.min()) / (S1.max() - S1.min())
    S2 = (S2 - S2.min()) / (S2.max() - S2.min())

    PS1 = P1 + S1
    PS2 = P2 + S2

    w_PS1 = 0.5 + (PS1-PS2)/2
    w_PS2 = 0.5 + (PS2-PS1)/2

    # Fuse the structure features

    F_S = w_PS1 * str_ir + w_PS2 * str_vis

    resA = im_to_patch(TexA, 5, 4)
    resB = im_to_patch(TexB, 5, 4)
    ret = np.zeros((resA.shape[0], resA.shape[1]))

    # Fuse the texture features

    for i in range(resA.shape[1]):

        block1 = np.reshape(resA[:, i], (5, 5))
        block2 = np.reshape(resB[:, i], (5, 5))

        a = bio_edge(block1, 5)
        b = bio_edge(block2, 5)
        if a > b:
            ret[:, i] = resA[:, i].flatten()
        else:
            ret[:, i] = resB[:, i].flatten()

    F_T = patch_to_im(TexA, ret, 5, 4)

    # Reconstruct the fused image
    F = F_S + F_T
    F = np.clip(F, 0, 1)
    F = (F * 255.).astype(np.uint8)

    imsave(pathOUT + 'F'+str("%03d" % k) + '.png', F)
