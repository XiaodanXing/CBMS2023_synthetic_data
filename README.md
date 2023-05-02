# CBMS2023_synthetic_data
This GitHub repository contains the source code and implementation details for CBMS 2023 submission, named *The Beauty or the Beast: Which Aspect of Synthetic Medical Images Deserves Our Focus?*  

In our submission, we report an interesting finding from our synthesis work: we discovered that the utility of synthetic data is not necessarily correlated with its fidelity. In other words,** realistic data is not always the most desirable outcome in medical image synthesis**. We present a case study on an open-source X-ray dataset, where high-fidelity synthetic data can actually harm the performance of downstream tasks.

# The repository

This repository aims to offer practical applications for 2D medical image synthesis, as well as a range of evaluation methods. The repository consists of two main parts:

1. Details of the synthesis algorithms (![VQ-VAE2](https://github.com/rosinality/vq-vae-2-pytorch), ![StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) and ![LDM](https://github.com/CompVis/latent-diffusion)) evaluated in this paper, including hyperparameters and implementation details. We will also release the source code we used for our synthesis algorithms in case there are any version changes in these repositories. We are currently in the process of cleaning up the code.


2. Evaluation algorithms, including computation of our fidelity (FID, precision) and variety (recall, average file size). These algorithms were derived and revised from ![pytorch-fid](https://github.com/mseitzer/pytorch-fid), ![generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc).



# Usage
The implementation details can be found in the ![image_synthesizer](./image_synthesizer/details.md) folder.

To assess the quality of synthetic data, you will need to prepare two folders. One folder should contain the reference real images, and the other should contain the synthetic images to be evaluated. Then, call the following function
```
python ./quality_evaluator/main.py -i /path/to/synthetic/images -r /path/to/real/images -o /path/for/outputs
```

