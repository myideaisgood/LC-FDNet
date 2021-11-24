import cv2
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.data_transformer import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        assert (subset == 'train') or (subset == 'val') or (subset == 'test')

        self.args = args
        self.imgs = []
        self.subset = subset

        for file in os.listdir(os.path.join(args.data_dir, args.dataset, subset)):
            if file.endswith('.png'):
                self.imgs.append(file)

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __getitem__(self, idx):

        DATA_DIR = self.args.data_dir
        DATASET = self.args.dataset
        CROP_SIZE = self.args.crop_size

        # Read in Image
        imgpath = os.path.join(DATA_DIR, DATASET, self.subset, self.imgs[idx])
        img = cv2.cvtColor(cv2.imread(imgpath),cv2.COLOR_BGR2RGB)

        # Crop image for train dataset
        if self.subset == 'train':
            img = RandomCrop(img, CROP_SIZE)

        ori_img = img

        img = RGB2YUV(img)
        img = self.transform(img)

        img, padding = pad_img(img)
        img = space_to_depth_tensor(img)

        img_a, img_b, img_c, img_d = img[:,0], img[:,1], img[:,2], img[:,3]

        return img_a, img_b, img_c, img_d, ori_img, self.imgs[idx], padding
    
    def __len__(self):
        return len(self.imgs)


def show_samples(args):

    subset = 'test'
    if subset=='train':
        BATCH_SIZE = args.batch_size
    else:
        BATCH_SIZE = 1

    # Create directory to save images
    PLAYGROUND_DIR = 'playground/'

    if not os.path.exists(PLAYGROUND_DIR):
        os.mkdir(PLAYGROUND_DIR)

    dataset = Dataset(args, subset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for i, data in enumerate(dataloader, 0):

        img_a, img_b, img_c, img_d, ori_imgs, img_names, padding = data

        B, _, H, W, = img_a.shape

        img_a = torch.unsqueeze(img_a, dim=2)

        img_a = (img_a.squeeze()).permute(1,2,0)
        img_a = img_a.numpy()
        img_a = YUV2RGB(img_a)

        img_name = PLAYGROUND_DIR + 'div2k/' + img_names[0]
        cv2.imwrite(img_name, cv2.cvtColor(img_a,cv2.COLOR_RGB2BGR))

        # img_b = torch.unsqueeze(img_b, dim=2)
        # img_c = torch.unsqueeze(img_c, dim=2)
        # img_d = torch.unsqueeze(img_d, dim=2)

        # imgs = torch.cat([img_a,img_b,img_c,img_d], dim=2)

        # imgs = depth_to_space_tensor(imgs)
        
        # # Padding
        # B, C, H, W = imgs.shape
        # if subset == 'test':
        #     pad_h = padding[0]
        #     pad_w = padding[1]
        #     imgs = imgs[:,:,:H-pad_h, :W-pad_w]
        # imgs = tensor2image(imgs)
        # ori_imgs = ori_imgs.numpy()        

        # for b_idx in range(B):

        #     img = imgs[b_idx]
        #     ori_img = ori_imgs[b_idx]
            
        #     img = YUV2RGB(img)
            
        #     assert np.array_equal(img, ori_img)
        #     if subset == 'test':
        #         img_name = PLAYGROUND_DIR + img_names[0]
        #     else:
        #         img_name = PLAYGROUND_DIR + str(i).zfill(3) + '_' + str(b_idx).zfill(2) + '.png'
        #     cv2.imwrite(img_name, cv2.cvtColor(img,cv2.COLOR_RGB2BGR))