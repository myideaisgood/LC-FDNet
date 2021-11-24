import torch
from torch import nn

from utils.quantizer import diff_quantizer
from utils.threshold import diff_threshold

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class FDNet(torch.nn.Module):
    def __init__(self, color, indim, hu):
        super(FDNet, self).__init__()

        assert (color=='Y') or (color=='U') or (color=='V')

        ResBlock_L1 = ResBlock(hu, hu)
        ResBlock_L2 = ResBlock(hu, hu)
        ResBlock_L3 = ResBlock(hu, hu)

        ResBlock_H1 = ResBlock(hu, hu)
        ResBlock_H2 = ResBlock(hu, hu)
        ResBlock_H3 = ResBlock(hu, hu)

        # Y : -255 ~ 255
        # U,V : -511 ~ 511
        if color=='Y':
            self.sym_num = 511
        else:
            self.sym_num = 1021

        self.color = color

        self.enc_L = nn.Sequential(
            nn.Conv2d(indim, hu, 3, padding=1),
            nn.BatchNorm2d(hu),
            nn.ReLU(),
            ResBlock_L1,
            ResBlock_L2,
            ResBlock_L3
        )

        self.enc_H = nn.Sequential(
            nn.Conv2d(indim+1, hu, 3, padding=1),
            nn.BatchNorm2d(hu),
            nn.ReLU(),            
            ResBlock_H1,
            ResBlock_H2,
            ResBlock_H3
        )

        self.pmf_enc_L = nn.Sequential(
            nn.Conv2d(hu, 2*hu, 3, padding=1),
            nn.BatchNorm2d(2*hu),
            nn.ReLU(),
            nn.Conv2d(2*hu, self.sym_num, 3, padding=1)
        )

        self.pmf_enc_H = nn.Sequential(
            nn.Conv2d(hu, 2*hu, 3, padding=1),
            nn.BatchNorm2d(2*hu),
            nn.ReLU(),
            nn.Conv2d(2*hu, self.sym_num, 3, padding=1)
        )   

        self.res_enc_L = nn.Conv2d(hu, 1, 3, padding=1)
        self.res_enc_H = nn.Conv2d(hu, 1, 3, padding=1)  
        
        self.error_var_enc = nn.Sequential(
            nn.Conv2d(hu, 1, 3, padding=1),
            nn.ReLU()
        )

        self.var_th_enc = nn.Sequential(
            nn.Conv2d(hu, hu, 3, padding=1),
            nn.BatchNorm2d(hu),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hu, hu, 3, padding=1),
            nn.BatchNorm2d(hu),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hu, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )   

        self.softmax = nn.Softmax(dim=1)
        self.diff_quantizer = diff_quantizer.apply
        self.diff_threshold = diff_threshold.apply

    def predictor_L(self, x, ref):

        f = self.enc_L(x)
        res = self.res_enc_L(f)

        # Y : 0 ~ 255
        # U,V : -255 ~ 255
        if self.color=='Y':
            pred = torch.clamp(res+ref, min=0.0, max=255.0)  
        else:
            pred = torch.clamp(res+ref, min=-255.0, max=255.0)

        pmf_logit = self.pmf_enc_L(f)

        error_var_map = self.error_var_enc(f)
        error_var_th = 10 * self.var_th_enc(f)

        return pred, error_var_map, error_var_th, pmf_logit

    def predictor_H(self, x, ref):

        f = self.enc_H(x)
        res = self.res_enc_H(f)

        # Y : 0 ~ 255
        # U,V : -255 ~ 255
        if self.color=='Y':
            pred = torch.clamp(res+ref, min=0.0, max=255.0)  
        else:
            pred = torch.clamp(res+ref, min=-255.0, max=255.0)        

        pmf_logit = self.pmf_enc_H(f)

        return pred, pmf_logit

    def forward(self, input, gt, ref, frequency='low', mode='train'):

        if frequency == 'low':
            
            pred, error_var_map, error_var_th, pmf_logit = self.predictor_L(input, ref)

            res = pred - gt
            q_res = self.diff_quantizer(res)

            # Y : 0 ~ 510
            # U,V : 0 ~ 1020
            if self.color=='Y':
                q_res = q_res.long() + 255
            else:
                q_res = q_res.long() + 510

            mask_low = self.diff_threshold(error_var_map, error_var_th, 'down')

            if mode=='train':
                return pred, q_res, error_var_map, error_var_th, mask_low, pmf_logit
            else:
                pmf_softmax = self.softmax(pmf_logit)
                del pmf_logit
                return pred, q_res, error_var_map, error_var_th, mask_low, pmf_softmax
        else:
            pred, pmf_logit = self.predictor_H(input, ref)

            res = pred - gt
            q_res = self.diff_quantizer(res)

            # Y : 0 ~ 510
            # U,V : 0 ~ 1020
            if self.color=='Y':
                q_res = q_res.long() + 255
            else:
                q_res = q_res.long() + 510            

            if mode=='train':
                return pred, q_res, pmf_logit
            else:
                pmf_softmax = self.softmax(pmf_logit)
                del pmf_logit
                return pred, q_res, pmf_softmax