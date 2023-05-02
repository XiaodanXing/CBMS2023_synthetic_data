# Implementation details and hyperparameters

## VQ-VAE2
For the VQ-VAE models on 256^2 and 512^2, we used a two-level latent hierarchy. The encoder first transformed the image to a 64×64 feature representation to model the bottom-level features that control the image details; then the encoder transformed the bottom-level feature representation to a 32×32 top-level features that control the global information of synthetic images. We discovered that the conditional implementation of VQ-VAE will increase the inference time and decrease the synthesis performance, thus we trained VQ-VAE models for each image category respectively. 
The learning rate was 3e-4.

## StyleGAN2
To assure the best performance on various datasets, we used the recommended config settings in StyleGAN2. For the X-ray dataset, the fmaps=1 lrate=0.0025, gamma=0.5,  ema=20; for the pathological dataset, we used fmaps=0.5, lrate=0.0025, gamma=1,ema=20

## LDM
We attached our configuration file for both X-ray and pathological dataset. 
