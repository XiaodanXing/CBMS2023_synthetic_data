import copy
import os
from copy import deepcopy
import pandas as pd
import numpy as np
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import h5py
from PIL import Image,ImageOps
import torch.nn as nn
import random

random.seed(0)


class biDataset(Dataset):

    def __init__(self,filepath = '/media/NAS02/xiaodan/osic/montages_osic2/'):  # crop_size,


        self.imlist = os.listdir(filepath)

        self.filepath = filepath


        self.transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    def __getitem__(self, idx):
        if idx < len(self.imlist):
            img_name = self.imlist[idx]

            img = Image.open(img_name)
            img = self.transforms(img)

            return {'image':img,'filename':img_name,}


    def __len__(self):
        return len(self.imlist)