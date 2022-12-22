import os
from multiprocessing import freeze_support

import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image

from models import UNetSemantic
from losses import DiceLoss
#from datasets import FacemaskSegDataset
from metrics import *


def adjust_learning_rate(optimizer, gamma, num_steps=1):
    for i in range(num_steps):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma


root_dir = "datasets/celeba/images"
train_anns = "datasets/celeba/annotations/train.csv"
val_anns = "datasets/celeba/annotations/val.csv"

checkpoint_path = "weights"
sample_folder = 'sample'

cuda = True    # 응 아니야 이건 True로 해야해
lr = 0.001
batch_size = 4
num_workers = 4

step_iters = [50000, 75000, 100000]
gamma = 0.1
visualize_per_iter = 1000
print_per_iter = 10
save_per_iter = 1000

iters = 0
num_epochs = 100
img_size = 112
device = torch.device("cuda")


class FacemaskSegDataset(data.Dataset):
    def __init__(self, train=True):
        self.train = train

        if self.train:
            self.df = pd.read_csv(train_anns)
        else:
            self.df = pd.read_csv(val_anns)

        self.load_images()

    def load_images(self):
        self.fns = []
        for idx, rows in self.df.iterrows():
            _, img_name, mask_name = rows
            img_path = os.path.join(root_dir, img_name)
            mask_path = os.path.join(root_dir, mask_name)
            img_path = img_path.replace('\\', '/')
            mask_path = mask_path.replace('\\', '/')
            if os.path.isfile(mask_path):
                self.fns.append([img_path, mask_path])

    def __getitem__(self, index):
        img_path, mask_path = self.fns[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (img_size, img_size))
        mask[mask > 0] = 1.0
        mask = np.expand_dims(mask, axis=0)

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        return img, mask

    def collate_fn(self, batch):
        imgs = torch.stack([i[0] for i in batch])
        masks = torch.stack([i[1] for i in batch])
        return {
            'imgs': imgs,
            'masks': masks
        }

    def __len__(self):
        return len(self.fns)



def collate(self, batch):
    imgs = torch.stack([i[0] for i in batch])
    masks = torch.stack([i[1] for i in batch])
    return {
        'imgs': imgs,
        'masks': masks
    }


trainset = FacemaskSegDataset()
valset = FacemaskSegDataset(train=False)



trainloader = data.DataLoader(
    trainset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True, # 왜 True가 아닌 False일때 GPU 메모리 이용량이 늘어나는거지??
    shuffle=True,
    collate_fn=trainset.collate_fn)

valloader = data.DataLoader(
    valset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True, # GPU 메모리
    shuffle=True,
    collate_fn=valset.collate_fn)

epoch = 0
iters = 0
num_iters = 0

model = UNetSemantic().to(device)
criterion_dice = DiceLoss()
criterion_bce = nn.BCELoss()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def validate(self, sample_folder, sample_name, img_list):
    save_img_path = os.path.join(sample_folder, sample_name + '.png')
    img_list = [i.clone().cpu() for i in img_list]
    imgs = torch.stack(img_list, dim=1)

    # imgs shape: Bx5xCxWxH

    imgs = imgs.view(-1, *list(imgs.size())[2:])
    save_image(imgs, save_img_path, nrow=3)
    print(f"Save image to {save_img_path}")

def train_epoch(iters):
    model.train()
    running_loss = {
        'DICE': 0,
        'BCE': 0,
        'T': 0,
    }
    running_time = 0

    for idx, batch in enumerate(tqdm(trainloader)):
        optimizer.zero_grad()
        inputs = batch['imgs'].to(device)
        targets = batch['masks'].to(device)

        start_time = time.time()

        outputs = model(inputs)

        loss_bce = criterion_bce(outputs, targets)
        loss_dice = criterion_dice(outputs, targets)
        loss = loss_bce + loss_dice
        loss.backward()
        optimizer.step()

        end_time = time.time()

        running_loss['T'] += loss.item()
        running_loss['DICE'] += loss_dice.item()
        running_loss['BCE'] += loss_bce.item()
        running_time += end_time - start_time

        if iters % print_per_iter == 0:
            for key in running_loss.keys():
                running_loss[key] /= print_per_iter
                running_loss[key] = np.round(running_loss[key], 5)
            loss_string = '{}'.format(running_loss)[1:-1].replace("'", '').replace(",", ' ||')
            running_time = np.round(running_time, 5)
            print(
                '[{}/{}][{}/{}] || {} || Time: {}s'.format(epoch, num_epochs, iters, num_iters,
                                                           loss_string, running_time))
            running_time = 0
            running_loss = {
                'DICE': 0,
                'BCE': 0,
                'T': 0,
            }

        if iters % save_per_iter == 0:
            save_path = os.path.join(
                checkpoint_path,
                f"model_segm_{epoch}_{iters}.pth")
            save_path = save_path.replace('\\', '/')
            torch.save(model.state_dict(), save_path)
            print(f'Save model at {save_path}')
        iters += 1

def validate_epoch():
    # Validate

    model.eval()
    metrics = [DiceScore(1), PixelAccuracy(1)]
    running_loss = {
        'DICE': 0,
        'BCE': 0,
        'T': 0,
    }

    running_time = 0
    print('=============================EVALUATION===================================')
    with torch.no_grad():
        start_time = time.time()
        for idx, batch in enumerate(tqdm(valloader)):

            inputs = batch['imgs'].to(device)
            targets = batch['masks'].to(device)
            outputs = model(inputs)
            loss_bce = criterion_bce(outputs, targets)
            loss_dice = criterion_dice(outputs, targets)
            loss = loss_bce + loss_dice
            running_loss['T'] += loss.item()
            running_loss['DICE'] += loss_dice.item()
            running_loss['BCE'] += loss_bce.item()
            for metric in metrics:
                metric.update(outputs.cpu(), targets.cpu())

        end_time = time.time()
        running_time += (end_time - start_time)
        running_time = np.round(running_time, 5)
        for key in running_loss.keys():
            running_loss[key] /= len(valloader)
            running_loss[key] = np.round(running_loss[key], 5)

        loss_string = '{}'.format(running_loss)[1:-1].replace("'", '').replace(",", ' ||')

        print('[{}/{}] || Validation || {} || Time: {}s'.format(epoch, num_epochs, loss_string,
                                                                running_time))
        for metric in metrics:
            print(metric)
        print('==========================================================================')

if __name__ == '__main__':
    freeze_support()
    for epoch in range(epoch, num_epochs + 1):
        epoch = epoch
        train_epoch(iters)
        validate_epoch()
