import numpy as np
from prdc import compute_prdc
import pandas as pd
import timm
from biDataset import biDataset
from torch.utils.data import DataLoader
import torch
from tqdm import  tqdm
import os


from fid_score import calculate_frechet_distance
import argparse


parser = argparse.ArgumentParser()


parser.add_argument("--syn_method", default='train_A2_stylegan0.00,train_A2_stylegan0.20,train_A2_stylegan0.40,'
                                            'train_A2_stylegan0.60,train_A2_stylegan0.80,train_A2_stylegan1.00,'
                                            'train_A2_diffusion')
# parser.add_argument("--syn_method", default='train_A1_part_stylegan0.00,train_A1_part_stylegan0.20,train_A1_part_stylegan0.40,'
#                                             'train_A1_part_stylegan0.60,train_A1_part_stylegan0.80,train_A1_part_stylegan1.00,'
#                                             'train_A1_part_diffusion,train_A1_part')
parser.add_argument("--filepath", default='/media/NAS04/chestXpert/CheXpert-v1.0/synthesis')
# parser.add_argument("--filepath", default='/rds/general/user/xxing/home/data/chexpert')
parser.add_argument("--class_names", default='pe,nope')
parser.add_argument("--out_dir", default='./avg_images')

args = parser.parse_args()

def compute_feature(img_path,save_path):

    train_dt = biDataset(filepath = img_path)
    train_loader = DataLoader(train_dt, batch_size=4, shuffle=True, drop_last=True,
                              pin_memory=torch.cuda.is_available())

    encoder = timm.create_model('inception_v3',
                                 pretrained=True
                                 )
    encoder = torch.nn.parallel.DataParallel(encoder)
    encoder = encoder.cuda()
    encoder.eval()

    # compute the latent space for each image and store in (latent, image)
    features = []
    filenames = []

    batch = tqdm(train_loader, total=len(train_dt)/15)
    for data in batch:
        images = data['image'].cuda()
        filename = data['filename']
        # if num_channels !=3:
        #     images = images.repeat(1,3,1,1)
        encoded = encoder.module.global_pool(encoder.module.forward_features(images))


        e = np.array(encoded.detach().cpu())
        for i in range(images.shape[0]):
            features.append(e[i])
        filenames.append(filename)


    filenames = np.concatenate(filenames, axis=0)
    features_arr = np.asarray(features)
    filenames = np.asarray(filenames)
    features = pd.DataFrame(features_arr)
    features['filename'] = filenames

    features.to_csv(save_path)


def compute_precision_and_recall(real_features,fake_features):

    nearest_k = 5

    metrics = compute_prdc(real_features=real_features,
                           fake_features=fake_features,
                           nearest_k=nearest_k)

    print(metrics)


def fidelity_metrics(save_name_real,save_name_fake):

    nearest_k = 5
    # real_features = computeFeature(real_dir,class_names,resolution,num_channels,save_name_real)
    # fake_features = computeFeature(fake_dir, class_names, resolution, num_channels, save_name_fake)
    real_features = pd.read_csv(save_name_real,index_col=0)
    real_features = np.array(real_features.drop('filename',axis=1))
    fake_features = pd.read_csv(save_name_fake ,index_col=0)
    fake_features = np.array(fake_features.drop('filename',axis=1))
    metrics = compute_prdc(real_features=real_features,
                           fake_features=fake_features,
                           nearest_k=nearest_k)

    mu = np.mean(real_features, axis=0)
    sigma = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    fid = calculate_frechet_distance(mu_fake, sigma_fake,mu, sigma, )
    metrics['fid'] = fid
    print(metrics)

def cal_fidelity(configs):

    print('Start to computing the imageNet features for real images')

    compute_feature(configs.ref_dir,
                    os.path.join(configs.output_dir,'ref_features.csv'))

    print('Start to computing the imageNet features for synthetic images')

    compute_feature(configs.input_dir,
                    os.path.join(configs.output_dir,'syn_features.csv'))

    fidelity_metrics(os.path.join(configs.output_dir,'ref_features'),
                         os.path.join(configs.output_dir,'syn_features.csv'),
                         )


