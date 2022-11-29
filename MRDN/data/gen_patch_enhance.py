from PIL import Image
import os
import numpy as np
import tqdm
import cv2


def main():
    # file_path = r'/SSD64/Smooth/train/GEN'
    save_path = '/media/server/80SSD/LihuaJian/train/TrainData490'

    gt_path = '/media/server/80SSD/LihuaJian/train/490/GT'
    in_path = '/media/server/80SSD/LihuaJian/train/490/Input'

    gt_save = save_path + '/' + 'GT'
    input_save = save_path + '/' + 'In'

    if not os.path.isdir(gt_save):
        os.makedirs(gt_save)
    if not os.path.isdir(input_save):
        os.makedirs(input_save)

    patch_size = 128
    stride = 32
    resize_set = [1.0, 0.5]
    flip_set = [0, 2]
    rotate_set = [0, 1]

    count = 0
    for name in tqdm.tqdm(os.listdir(gt_path)):
        for resize in resize_set:
            for flip in flip_set:
                for rotate in rotate_set:
                    gt = Image.open(gt_path + '/' + name)
                    gt = np.asarray(gt)
                    Input = Image.open(in_path + '/' + name)
                    Input = np.asarray(Input)


                    row, col = gt.shape[0], gt.shape[1]
                    row = int(row * resize)
                    col = int(col*resize)

                    gt = cv2.resize(gt,(col, row))
                    if not flip == 2:
                        gt = np.flip(gt, flip)
                    M = cv2.getRotationMatrix2D(((col - 1) / 2.0, (row - 1) / 2.0), 90 * rotate, 1)
                    gt = cv2.warpAffine(gt, M, (col, row))
                    Input = cv2.resize(Input, (col, row))
                    if not flip == 2:
                        Input = np.flip(Input, flip)
                    Input = cv2.warpAffine(Input, M, (col, row))

                    for i in range(0, row - patch_size, stride):
                        for j in range(0, col - patch_size, stride):
                            patch_name = '{}.png'.format(count)
                            count = count + 1

                            gt_patch = Image.fromarray(gt[i:i + patch_size, j:j + patch_size, :])
                            gt_patch.save(gt_save + '/' + patch_name)
                            input_patch = Image.fromarray(Input[i:i + patch_size, j:j + patch_size, :])
                            input_patch.save(input_save + '/' + patch_name)
                    # patch_name = '{}.png'.format(count)
                    # count = count + 1
                    # gt_patch = Image.fromarray(gt)
                    # gt_patch.save(gt_save + '/' + patch_name)
                    # input_patch = Image.fromarray(Input)
                    # input_patch.save(input_save + '/' + patch_name)
    a = 0


if __name__ == '__main__':
    main()
