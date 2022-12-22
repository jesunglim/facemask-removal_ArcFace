from multiprocessing import freeze_support

import os
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

from models import *
from losses import *

root_dir = "./datasets/celeba/images/"
checkpoint_path = "./weights"
sample_folder = './sample'


cuda = True
lr = 0.001
batch_size = 10
num_workers = 4

step_iters = [50000, 75000, 100000]
gamma = 0.1

d_num_layers = 3

visualize_per_iter = 500
save_per_iter = 500
print_per_iter = 10
num_epochs = 100

lambda_G = 1.0
lambda_rec_1 = 100.0
lambda_rec_2 = 100.0
lambda_per = 10.0

img_size = 112



class FacemaskDataset(data.Dataset):
    def __init__(self):
        self.root_dir = root_dir

        self.mask_folder = os.path.join(self.root_dir, 'celeba512_30k_binary')
        self.img_folder = os.path.join(self.root_dir, 'celeba512_30k')
        self.load_images()

    def load_images(self):
        self.fns = []
        idx = 0
        img_paths = sorted(os.listdir(self.img_folder))
        for img_name in img_paths:
            mask_name = img_name.split('.')[0] + '_KN95.jpg'
            img_path = os.path.join(self.img_folder, img_name)
            mask_path = os.path.join(self.mask_folder, mask_name) 
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


def adjust_learning_rate(optimizer, gamma, num_steps=1):
    for i in range(num_steps):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma


def get_epoch_iters(path):
    path = os.path.basename(path)
    tokens = path[:-4].split('_')
    try:
        if tokens[-1] == 'interrupted':
            epoch_idx = int(tokens[-3])
            iter_idx = int(tokens[-2])
        else:
            epoch_idx = int(tokens[-2])
            iter_idx = int(tokens[-1])
    except:
        return 0, 0

    return epoch_idx, iter_idx


def load_checkpoint(model_G, model_D, path):
    state = torch.load(path, map_location='cpu')
    model_G.load_state_dict(state['G'])
    model_D.load_state_dict(state['D'])
    print('Loaded checkpoint successfully')


epoch = 0
iters = 0

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.exists(sample_folder):
    os.makedirs(sample_folder)



start_iter = iters
iters = 5754000   # 모델 불러오기
device = 'cuda:1'

trainset = FacemaskDataset()

trainloader = data.DataLoader(
    trainset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True,
    collate_fn=trainset.collate_fn)

epoch = int(start_iter / len(trainloader))
epoch = 74 # 모델 불러오기
print(len(trainloader))
#iters = start_iter
num_iters = (num_epochs + 1) * len(trainloader)


model_G = GatedGenerator().to(device)
model_D = NLayerDiscriminator(d_num_layers, use_sigmoid=False).to(device)
model_P = PerceptualNet(name="vgg16", resize=False).to(device)


load_checkpoint(model_G, model_D, '/home/ielab/project/facemask_remove/weights/prev_model2.pth')

criterion_adv = GANLoss(target_real_label=0.9, target_fake_label=0.1)
criterion_rec = nn.SmoothL1Loss()
criterion_ssim = SSIM(window_size=11)
criterion_per = nn.SmoothL1Loss()

optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr)
optimizer_G = torch.optim.Adam(model_G.parameters(), lr=lr)


def validate(self, sample_folder, sample_name, img_list):
    save_img_path = os.path.join(sample_folder, sample_name + '.png')
    img_list = [i.clone().cpu() for i in img_list]
    imgs = torch.stack(img_list, dim=1)

    # imgs shape: Bx5xCxWxH

    imgs = imgs.view(-1, *list(imgs.size())[2:])
    save_image(imgs, save_img_path, nrow=5)
    print(f"Save image to {save_img_path}")

if __name__ == '__main__':
    freeze_support()
    model_G.train()
    model_D.train()

    running_loss = {
        'D': 0,
        'G': 0,
        'P': 0,
        'R_1': 0,
        'R_2': 0,
        'T': 0,
    }

    running_time = 0
    step = 0

    for e in range(epoch, num_epochs):
        epoch = e
        for i, batch in enumerate(tqdm(trainloader)):
            start_time = time.time()
            imgs = batch['imgs'].to(device)
            masks = batch['masks'].to(device)

            # Train discriminator
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            first_out, second_out = model_G(imgs, masks)

            first_out_wholeimg = imgs * (1 - masks) + first_out * masks
            second_out_wholeimg = imgs * (1 - masks) + second_out * masks

            masks = masks.cpu()

            fake_D = model_D(second_out_wholeimg.detach())
            real_D = model_D(imgs)

            loss_fake_D = criterion_adv(fake_D, target_is_real=False)
            loss_real_D = criterion_adv(real_D, target_is_real=True)

            loss_D = (loss_fake_D + loss_real_D) * 0.5

            loss_D.backward()
            optimizer_D.step()

            real_D = None

            # Train Generator
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            fake_D = model_D(second_out_wholeimg)
            loss_G = criterion_adv(fake_D, target_is_real=True)

            fake_D = None

            # Reconstruction loss
            loss_l1_1 = criterion_rec(first_out_wholeimg, imgs)
            loss_l1_2 = criterion_rec(second_out_wholeimg, imgs)
            loss_ssim_1 = criterion_ssim(first_out_wholeimg, imgs)
            loss_ssim_2 = criterion_ssim(second_out_wholeimg, imgs)

            loss_rec_1 = 0.5 * loss_l1_1 + 0.5 * (1 - loss_ssim_1)
            loss_rec_2 = 0.5 * loss_l1_2 + 0.5 * (1 - loss_ssim_2)

            # Perceptual loss
            loss_P = model_P(second_out_wholeimg, imgs)

            loss = lambda_G * loss_G + lambda_rec_1 * loss_rec_1 + lambda_rec_2 * loss_rec_2 + lambda_per * loss_P
            loss.backward()
            optimizer_G.step()

            end_time = time.time()

            imgs = imgs.cpu()
            # Visualize number
            running_time += (end_time - start_time)
            running_loss['D'] += loss_D.item()
            running_loss['G'] += (lambda_G * loss_G.item())
            running_loss['P'] += (lambda_per * loss_P.item())
            running_loss['R_1'] += (lambda_rec_1 * loss_rec_1.item())
            running_loss['R_2'] += (lambda_rec_2 * loss_rec_2.item())
            running_loss['T'] += loss.item()

            if iters % print_per_iter == 0:
                for key in running_loss.keys():
                    running_loss[key] /= print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'", '').replace(",", ' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(e, num_epochs, iters,
                                                                        num_iters, loss_string,
                                                                        running_time))

                running_loss = {
                    'D': 0,
                    'G': 0,
                    'P': 0,
                    'R_1': 0,
                    'R_2': 0,
                    'T': 0,
                }
                running_time = 0

            if iters % save_per_iter == 0:
                torch.save({
                    'D': model_D.state_dict(),
                    'G': model_G.state_dict(),
                }, os.path.join(checkpoint_path, f"fial_model.pth"))
                #os.path.join(checkpoint_path, f"model_{epoch}_{iters}.pth"))
                with open(checkpoint_path+'/checkpoint.txt', 'w') as file:
                    file.write(f"epoch:{epoch}_iters{iters}\n")

            # Step learning rate
            if iters == 10000:
                adjust_learning_rate(optimizer_D, gamma)
                adjust_learning_rate(optimizer_G, gamma)
                step += 1

            # Visualize sample
            """
            if iters % visualize_per_iter == 0:
                masked_imgs = imgs * (1 - masks) + masks

                img_list = [imgs, masked_imgs, first_out, second_out, second_out_wholeimg]
                # name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
                filename = f"{epoch}_{str(iters)}"
                validate(sample_folder, filename, img_list)
            """
            iters += 1


