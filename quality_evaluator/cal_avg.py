import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from biDataset import biDataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms, utils
from torchvision.utils import save_image
import argparse
from tqdm import tqdm






def cal_mean(filepath):

    train_dt = biDataset(filepath=filepath)
    train_loader = DataLoader(train_dt, batch_size=20, shuffle=True, drop_last=True,
                              pin_memory=torch.cuda.is_available())
    c,h,w = train_dt[0]['image'].shape
    avg_images = torch.zeros([c,h,w]).cuda()
    for data in tqdm(train_loader):
        images = data['image'].cuda()
        images = torch.sum(images,dim=0)
        avg_images += images
    avg_images = avg_images/train_dt.__len__()

    return avg_images



def cal_avg(configs):

    avg_image = cal_mean(filepath=configs.input_dir)
    save_image(avg_image, os.path.join(configs.output_dir,'avg_images.png'),
               normalize=True, range=(-1, 1))
