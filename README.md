# LC-FDNet

# [LC-FDNet : Learned Lossless Image Compression with Frequency Decomposition Network] CVPR 2022

Hochang Rhee, Yeong Il Jang, Seyun Kim, Nam Ik Cho

[[Paper](To be Added)] [[Supplementary](To be Added)] [[Arxiv](https://arxiv.org/abs/2112.06417)]

## Environments
- Ubuntu 18.04
- Pytorch 1.7.0
- CUDA 10.0.130 & cuDNN 7.6.5
- Python 3.7.7

You can type the following command to easily build the environment.
Download 'fdnet_env.yml' and type the following command.

```
conda env create -f fdnet_env.yml
```

## Abstract

Recent learning-based lossless image compression methods encode an image in the unit of subimages and achieve better or comparable performances to conventional 
non-learning algorithms, along with reasonable inference time. However, these methods do not consider the performance drop in the high-frequency region, giving equal 
consideration to the low and high-frequency regions. In this paper, we propose a new lossless image compression method that proceeds the encoding in a coarse-to-fine manner 
to deal with low and high-frequency regions differently. We first compress the low-frequency components and then use them as additional input for encoding the remaining 
high-frequency region. The low-frequency components act as strong prior in this case, which leads to improved estimation in the high-frequency area. To classify the pixels 
as low or high-frequency regions, we also design a discriminating network that finds adaptive thresholds depending on the color channel, spatial location, and image 
characteristics. The network derives the image-specific optimal ratio of low/high-frequency components as a result. Experiments show that the proposed method achieves 
state-of-the-art performance for benchmark high-resolution datasets, outperforming both conventional learning-based and non-learning approaches.

## Dataset
Train Dataset

[FLICKR2K] (https://github.com/limbee/NTIRE2017)

Test Dataset

[DIV2K] (https://data.vision.ee.ethz.ch/cvl/DIV2K/)

[CLIC] (http://clic.compression.cc/2019/challenge/)

## Brief explanation of contents

```
|── experiments
    ├──> experiment_name 
         ├──> ckpt : trained models will be saved here
         └──> log  : log will be saved here
|── dataset
    ├──> dataset_name1 
         ├──> train : training images of dataset_name1 should be saved here
         └──> test  : test images of dataset_name1 should be saved here
    ├──> dataset_name2
         ├──> train : training images of dataset_name2 should be saved here
         └──> test  : test images of dataset_name2 should be saved here         
|── utils : files for utility functions
|── config.py : configuration should be controlled only here 
|── decode.py : decode compressed files to images
|── encode.py : encode images to compressed format
|── fdnet_env.yml : virtual enviornment specification
|── model.py : architecture of FDNet
|── test.py : test the model. performance is estimated (not actual compression)
|── train.py : train the model
└── jpegxl : folder for jpegxl library. explained below.

```

## Guidelines for Codes

1. Install JPEG-XL
   1) Download JPEG-XL from https://gitlab.com/wg1/jpeg-xl and follow the installation process
   2) Change the directory name 'libjxl' to 'jpegxl'

2. Check configurations from config.py

3. Run the following command for training  the network
```
python train.py --gpu_num=0 --experiment_name='default/' --dataset='div2k/'
```

The trained model will be saved in the following directory : experiments/default/ckpt

4. Run the following command for testing the network. Note that the estimated bpp is printed. The actual bpp differs from the estimated bpp (about 0.1 bpp)
   
   **** parameter empty_cache in config.py should be set to True if memory issue occurs ****
```
python test.py --gpu_num=0 --experiment_name='default/' --dataset='div2k/' --empty_cache=True
```

5. Run the following command for actual compression (encoding)
```
python encode.py --gpu_num=0 --experiment_name='default/' --dataset='div2k/' --empty_cache=True
```
The encoded result will be saved in 'encoded_results/'

6. Run the following command for decoding
```
python decode.py --gpu_num=0 --experiment_name='default/' --dataset='div2k/' --empty_cache=True
```

This decodes the compressed files in 'encoded_results/' folder.

The decoded result will be saved in 'decoded_results/'
