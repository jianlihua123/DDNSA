from PIL import Image
import os
import numpy as np
import tqdm


def main():
    file_path = r'/SSD64/Smooth/train/GEN'
    save_path = r'/SSD64/Smooth/train/TrainData'
    gt_path = file_path+r'/'+'GT'
    s_path = file_path+r'/'+'S'
    t_path = file_path+r'/'+'T'
    input_path = file_path+r'/'+'Input'

    gt_save = save_path + '/' + 'GT'
    s_save = save_path + '/' + 'S'
    t_save = save_path + '/' + 'T'
    input_save = save_path + '/' + 'Input'

    if not os.path.isdir(gt_save):
        os.makedirs(gt_save)
    if not os.path.isdir(s_save):
        os.makedirs(s_save)
    if not os.path.isdir(t_save):
        os.makedirs(t_save)
    if not os.path.isdir(input_save):
        os.makedirs(input_save)

    patch_size = 64
    stride = 64

    count = 0
    for name in tqdm.tqdm(os.listdir(gt_path)):
        gt = Image.open(gt_path + '/' + name)
        gt = np.asarray(gt)
        s = Image.open(s_path + '/' + name)
        s = np.asarray(s)
        t = Image.open(t_path + '/' + name)
        t = np.asarray(t)
        input = Image.open(input_path + '/' + name)
        input = np.asarray(input)

        row, col = gt.shape[0], gt.shape[1]

        for i in range(0, row-stride, stride):
            for j in range(0, col-stride, stride):
                patch_name = '{}.png'.format(count)
                count = count + 1
                gt_patch = Image.fromarray(gt[i:i+patch_size, j:j+patch_size, :])
                gt_patch.save(gt_save+'/'+patch_name)
                s_patch = Image.fromarray(s[i:i + patch_size, j:j + patch_size])
                s_patch.save(s_save + '/' + patch_name)
                t_patch = Image.fromarray(t[i:i + patch_size, j:j + patch_size])
                t_patch.save(t_save + '/' + patch_name)
                input_patch = Image.fromarray(input[i:i + patch_size, j:j + patch_size, :])
                input_patch.save(input_save + '/' + patch_name)
    a = 0


if __name__ == '__main__':
    main()
