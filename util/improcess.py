import numpy as np
import scipy.io
import scipy.optimize
import math
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imread, imsave, imresize
import tensorflow as tf
from scipy.fftpack import fft, ifft, fft2, ifft2, fftshift, ifftshift
import skimage.transform



def im_to_patch(img, patch_size, overlap):
    w, h = img.shape
    gridx = range(0, w - patch_size + 1, patch_size - overlap)
    gridy = range(0, h - patch_size + 1, patch_size - overlap)
    res = np.zeros((patch_size*patch_size, len(gridx)*len(gridy)))
    # print(res.shape)
    # for i in gridx:
    #     print(i)
    k = 0
    for i in gridx:
        for j in gridy:
            x0 = gridx[i]
            x1 = gridx[i] + patch_size
            y0 = gridy[j]
            y1 = gridy[j] + patch_size
            # print(x0,x1)
            res[:, k] = img[x0:x1, y0:y1].flatten()
            k = k + 1

    return res




def patch_to_im(img, patch, patch_size, overlap):
    w, h = img.shape
    gridx = range(0, w - patch_size + 1, patch_size - overlap)
    gridy = range(0, h - patch_size + 1, patch_size - overlap)
    res = np.zeros((w, h))
    cnt = np.zeros((w, h))
    tmp = np.zeros((patch_size, patch_size))

    k = 0
    for i in gridx:
        for j in gridy:
            x0 = gridx[i]
            x1 = gridx[i] + patch_size
            y0 = gridy[j]
            y1 = gridy[j] + patch_size
            # print(x0,x1)
            tmp[:, :] = patch[:, k].reshape(patch_size, patch_size)
            res[x0:x1, y0:y1] = res[x0:x1, y0:y1] + tmp
            cnt[x0:x1, y0:y1] = cnt[x0:x1, y0:y1] + 1
            k = k + 1
    res[cnt < 1] = img[cnt < 1]
    cnt[cnt < 1] = 1.0
    res = res / cnt
    return res


def image_pre(img):
    img_mean = np.mean(img)
    img_std = np.std(img, ddof=1)  # ddof = 1 is nessary
    img_pre = (img - img_mean) / img_std
    return img_pre, img_mean, img_std


def array_to_image(imnp):
    imnp[imnp < 0] = 0
    imnp[imnp > 1] = 1
    imnp = np.array(imnp * 255, dtype=np.uint8)
    # imnp = np.array(imnp, dtype=np.uint8)
    imnp = Image.fromarray(imnp)
    # plt.imshow(imnp)
    # plt.show()
    return imnp


def spfreq(mat):
    m, n = mat.shape

    rff = np.array([1, -1])
    # a = np.convolve(mat, rff, 'valid')
    a = cv2.filter2D(mat, -1, rff)
    rf = np.sum(a*a) / (m*n)

    cff = np.array([1, -1])
    # b = np.convolve(mat, cff, 'valid')
    b = cv2.filter2D(mat, -1, cff)
    cf = np.sum(b*b) / (m*n)

    sf = math.sqrt(rf + cf)

    return sf


def sample_patch(num_patches, patch_side):
    dataset = np.zeros((patch_side * patch_side, num_patches))
    # path = './dataset/'  # infrared and visible images
    path = './dataset1/' ## coco dataset
    rand = np.random.RandomState(int(time.time()))
    image_number = rand.randint(1, 201, size=num_patches)
    for i in range(num_patches):
        """ Initialize indices for patch extraction """
        index3 = image_number[i]
        images = imread(path + str(index3) + '.png')

        images = images / 255.

        # print(type(images))
        images = np.array(images)
        h, w = images.shape
        index1 = np.random.randint(0, h-patch_side-1, 1)
        index2 = np.random.randint(0, w-patch_side-1, 1)
        index1 = int(index1)
        index2 = int(index2)
        """ Extract patch and store it as a column """

        patch = images[index1:index1 + patch_side, index2:index2 + patch_side]
        patch = patch.flatten()
        dataset[:, i] = patch

    """ Normalize and return the dataset """

    # dataset = normalizeDataset(dataset)
    return dataset



###########################################################################################
    """ Normalize the dataset provided as input """

def normalizeDataset(dataset):
    """ Remove mean of dataset """

    dataset = dataset - np.mean(dataset)

    """ Truncate to +/-3 standard deviations and scale to -1 to 1 """

    std_dev = 3 * np.std(dataset)
    dataset = np.maximum(np.minimum(dataset, std_dev), -std_dev) / std_dev

    """ Rescale from [-1, 1] to [0.1, 0.9] """

    dataset = (dataset + 1) * 0.4 + 0.1
    # dataset = (dataset - np.mean(dataset)) / np.std(dataset)

    return dataset


###########################################################################################
    """ Randomly samples image patches, normalizes them and returns as dataset """


def loadDataset(num_patches, patch_side):
    """ Load images into numpy array """

    images = scipy.io.loadmat('IMAGES.mat')
    images = images['IMAGES']

    """ Initialize dataset as array of zeros """

    dataset = np.zeros((patch_side * patch_side, num_patches))

    """ Initialize random numbers for random sampling of images
        There are 10 images of size 512 X 512 """

    rand = np.random.RandomState(int(time.time()))
    image_indices = rand.randint(512 - patch_side, size=(num_patches, 2))
    image_number = rand.randint(10, size=num_patches)

    """ Sample 'num_patches' random image patches """

    for i in range(num_patches):
        """ Initialize indices for patch extraction """

        index1 = image_indices[i, 0]
        index2 = image_indices[i, 1]
        index3 = image_number[i]

        """ Extract patch and store it as a column """

        patch = images[index1:index1 + patch_side, index2:index2 + patch_side, index3]
        patch = patch.flatten()
        dataset[:, i] = patch

    """ Normalize and return the dataset """

    dataset = normalizeDataset(dataset)
    return dataset

def weight_normalize(x):
    x_size = x.shape
    new = np.zeros(x.shape)
    for i in range(x_size[0]):
        sum_cloumn = np.sqrt(np.sum(np.square(x[i,:])))
        for j in range(x_size[1]):
            new[i, j] = x[i,j] / sum_cloumn
    return new

def gradient(input):
    filter = tf.reshape(tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]), [3, 3, 1, 1])
    d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    # print(d)
    return d

# def h_weight(Img, window_size):
#     Img_size = Img.shape
#     weight = np.zeros(Img_size)
#     pad_size = round (window_size / 2)
#     Img = cv2.copyMakeBorder(Img,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REFLECT)
#     for i in range(Img_size[0]):
#         for j in range(Img_size[1]):

#             block = Img[i:i+window_size, j:j+window_size]

#             block_h = np.sum(block, 1)
#             block_mean = np.mean(block, 1)
#             block_new_h = (block_h - block_mean) * np.array([block_h - block_mean]).T

#             cov_h = block_new_h / (window_size - 1)
#             H, Dh = np.linalg.eig(cov_h)
#             lambda_h = np.sum(H)

#             block_v = np.sum(block, 0)
#             block_mean = np.mean(block, 0)
#             block_new_v = np.array([block_v - block_mean]).T * (block_v - block_mean)

#             cov_v = block_new_v / (window_size - 1)
#             V, Dv = np.linalg.eig(cov_v)
#             lambda_v = np.sum(V)
#             weight[i,j] = lambda_h + lambda_v
#     return weight


def bio_edge(Img, window_size):

    block_h = np.sum(Img, 1)
    block_mean = np.mean(Img, 1)
    block_new_h = (block_h - block_mean) * np.array([block_h - block_mean]).T

    cov_h = block_new_h / (window_size - 1)

    # H, Dh = np.linalg.eig(cov_h)
    # lambda_h = np.sum(H)

    lambda_h = np.sum(cov_h.diagonal())

    block_v = np.sum(Img, 0)
    block_mean = np.mean(Img, 0)
    block_new_v = np.array([block_v - block_mean]).T * (block_v - block_mean)

    cov_v = block_new_v / (window_size - 1)

    # V, Dv = np.linalg.eig(cov_v)
    # lambda_v = np.sum(V)

    lambda_v = np.sum(cov_v.diagonal())

    lamb = lambda_h + lambda_v
    return lamb

def Enhance(x):
    c = 3.
    b = 0.4
    a = 1 / (sigmoid(c*(1-b)) - sigmoid(-1*c*(1+b)))
    y = a*(sigmoid(c*(x-b)) - sigmoid(-1*c*(x+b)))

    return y


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))



# 2D
def circulantshift2_x(xs, h):
    return np.hstack([xs[:, h:], xs[:, :h]] if h > 0 else [xs[:, h:], xs[:, :h]])

def circulantshift2_y(xs, h):
    return np.vstack([xs[h:, :], xs[:h, :]] if h > 0 else [xs[h:, :], xs[:h, :]])

def circulant2_dx(xs, h):
    return (circulantshift2_x(xs, h) - xs)

def circulant2_dy(xs, h):
    return (circulantshift2_y(xs, h) - xs)

def l0_gradient_minimization_2d(I, lmd, beta_max, beta_rate=2.0, max_iter=30, return_history=False):
    u'''image I can be both 1ch (ndim=2) or D-ch (ndim=D)'''
    S = np.array(I)

    # prepare FFT
    F_I = fft2(S, axes=(0, 1))
    Ny, Nx = S.shape[:2]
    D = S.shape[2] if S.ndim == 3 else 1
    dx, dy = np.zeros((Ny, Nx)), np.zeros((Ny, Nx))
    dx[int(Ny/2), int(Nx/2)-1:int(Nx/2)+1] = [-1, 1]
    dy[int(Ny/2)-1:int(Ny/2)+1, int(Nx/2)] = [-1, 1]
    F_denom = np.abs(fft2(dx))**2.0 + np.abs(fft2(dy))**2.0
    if D > 1: F_denom = np.dstack([F_denom]*D)

    S_history = [S]
    beta = lmd * 2.0
    hp, vp = np.zeros_like(S), np.zeros_like(S)
    for i in range(max_iter):
        # with S, solve for hp and vp in Eq. (12)
        hp, vp = circulant2_dx(S, 1), circulant2_dy(S, 1)
        if D == 1:
            mask = hp**2.0 + vp**2.0 < lmd/beta
        else:
            mask = np.sum(hp**2.0 + vp**2.0, axis=2) < lmd/beta
        hp[mask] = 0.0
        vp[mask] = 0.0

        # with hp and vp, solve for S in Eq. (8)
        hv = circulant2_dx(hp, -1) + circulant2_dy(vp, -1)
        S = np.real(ifft2((F_I + (beta*fft2(hv, axes=(0, 1))))/(1.0 + beta*F_denom), axes=(0, 1)))

        # iteration step
        if return_history:
            S_history.append(np.array(S))
        beta *= beta_rate
        if beta > beta_max: break

    if return_history:
        return S_history

    return S, I - S


def Nonlinear(x):
    W = 5*(x**0.5)
    return W

