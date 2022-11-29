import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.RDN import Net
from data.dataset import DatasetFromFolder
import math, glob
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch.nn as nn

setting_file = "setting/opt.json"
cuda = True
best_loss = float('inf')
with open(setting_file, 'r') as load_f:
    opt = json.load(load_f)


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def main():
    global opt, cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt["seed"] = 0
    print("Random Seed: ", opt["seed"])
    torch.manual_seed(opt["seed"])
    if cuda:
        torch.cuda.manual_seed(opt["seed"])
    cudnn.benchmark = True

    print("===> Loading datasets")
    file_path = {'In': opt["train_file_path"] + '/In_npy',
                 'GT': opt["train_file_path"] + '/GT_npy'}
    train_set = DatasetFromFolder(file_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt["threads"], batch_size=opt["batchSize"],
                                      shuffle=True)

    print("===> Building model")
    model = Net()

    print('Generator parameters: ', sum(param.numel() for param in model.parameters()))
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt["lr"])
    print("Learning rate: ", opt["lr"])

    print("===> Training")
    loss_list = []
    for epoch in range(opt["start_epoch"], opt["nEpochs"] + 1):
        psnr = train(training_data_loader, optimizer, model, criterion, epoch)
        loss_list.append(psnr)

    print("===> Saving")
    whole_res = pd.DataFrame(
        data={'loss': loss_list,
              'best_loss': best_loss},
        index=range(1, opt["nEpochs"] + 1)
    )
    whole_res.to_csv('result/result.csv', index_label='Epoch')

    print("===> Game over")


def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt["lr"] * (opt["momentum"] ** (epoch // opt["step"]))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch):
    global best_loss
    lr = adjust_learning_rate(epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    model.train()

    train_bar = tqdm(training_data_loader)

    for iteration, batch in enumerate(train_bar):
        In, GT = Variable(batch[0], requires_grad=False), Variable(batch[1], requires_grad=False)
        if cuda:
            In = In.cuda()
            GT = GT.cuda()

        out = model(In)
        optimizer.zero_grad()

        loss_all = criterion(out, GT)
        loss_all.backward()
        loss = loss_all.item()
        optimizer.step()

        train_bar.set_description(desc='%.6f' % (loss))
        if best_loss > loss:
            save_checkpoint(model, 9999)
            best_loss = loss
    train_bar.close()
    print("Epoch={}, lr={}, best_loss={:.6f}".format(epoch, lr, best_loss))
    save_checkpoint(model, epoch - 1)
    return loss


def save_checkpoint(model, epoch):
    model_folder = "result/"
    model_out_path = model_folder + "best_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()
