#!/usr/bin/env python3

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Conv2d, BatchNorm2d, MaxPool2d, Conv3d, BatchNorm3d, MaxPool3d, Module, MSELoss, ConvTranspose2d, ConvTranspose3d
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sn
from spectral import *
from metrics import *


#### MODELOS
class Hourglass(Module):  
    def __init__(self, in_channels, out_channels, ub_outCh, mb_outCh, lb_outCh):
        super(Hourglass, self).__init__()
        
        self.encoder_upperBlock = self.ConvBatchNormReLUx2(in_channels, ub_outCh, stride2 = 1)
        self.encoder_middleBlock = self.ConvBatchNormReLUx2(ub_outCh, mb_outCh, stride2 =1, kernel_size = 5, padding = 2)
        self.encoder_lowerBlock = self.ConvBatchNormReLUx1(mb_outCh, lb_outCh)
        
        self.skip2 = nn.Sequential(Conv2d(ub_outCh, mb_outCh, kernel_size = 1)) # 'Bigger' skip connection
        self.skip1 = nn.Sequential(Conv2d(mb_outCh, lb_outCh, kernel_size = 1)) # 'Smaller' skip connection
        
        self.decoder_lowerBlock = self.ConvBatchNormReLUx1(lb_outCh, lb_outCh)
        self.decoder_middleBlock = self.ConvBatchNormReLUx2(lb_outCh, mb_outCh, stride2 = 1, upsample = False, kernel_size = 5, padding = 2)
        self.decoder_upperBlock = self.ConvBatchNormReLUx2(mb_outCh, out_channels, stride2 = 1, upsample = False)
        

    def ConvBatchNormReLUx2(self, in_ch, out_ch, kernel_size = 3, stride1 = 1, stride2 = 2, padding = 1, upsample = False, upsample_ratio = 2):
        first_layer =  nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride1, padding),
            nn.LeakyReLU(inplace = True))
            #nn.BatchNorm2d(out_ch))
        
        second_layer = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size, stride2, padding),
            nn.LeakyReLU(inplace = True))
            #nn.BatchNorm2d(out_ch))
        
        if upsample is True:
            return nn.Sequential(first_layer, nn.Upsample(scale_factor = upsample_ratio), second_layer)
        else:
            return nn.Sequential(first_layer, second_layer)
            #return nn.Sequential(first_layer)
        
    def ConvBatchNormReLUx1(self, in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.LeakyReLU(inplace = True)
            #nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        # Encoder
        enc_out_upperBlock = self.encoder_upperBlock(x)
        enc_out_middleBlock = self.encoder_middleBlock(enc_out_upperBlock)
        enc_out_lowerBlock = self.encoder_lowerBlock(enc_out_middleBlock)
        
        # Decoder
        dec_out_lowerBlock = self.decoder_lowerBlock(enc_out_lowerBlock)
        
        x = dec_out_lowerBlock + self.skip1(enc_out_middleBlock)
        
        dec_out_middleBlock = self.decoder_middleBlock(x)
        
        x = dec_out_middleBlock + self.skip2(enc_out_upperBlock)
        
        dec_out_upperBlock = self.decoder_upperBlock(x)

        x = dec_out_upperBlock.view(dec_out_upperBlock.size(0), -1)
        
        return x

    
class Koundinya2D_CNN(Module):
    def __init__(self):
        super(Koundinya2D_CNN, self).__init__()
        
        self.cnn_layers = Sequential(
            Conv2d(in_channels = 7, out_channels = 7, kernel_size = 3, stride = 1, padding = 1),
            ReLU(inplace = True),
            
            Conv2d(in_channels = 7, out_channels = 7, kernel_size = 3, stride = 1, padding = 1),
            ReLU(inplace = True),
            
            Conv2d(in_channels = 7, out_channels = 60, kernel_size = 3, stride = 1, padding = 1),
            ReLU(inplace = True),

            Conv2d(in_channels = 60, out_channels = 100, kernel_size = 3, stride = 1, padding = 1),
            ReLU(inplace = True),

            Conv2d(in_channels = 100, out_channels = 175, kernel_size = 3, stride = 1, padding = 1),
            ReLU(inplace = True)

        )
        
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        return x

class Koundinya2D_CNN_batchnorm(Module):
    
    def __init__(self):
        super(Koundinya2D_CNN_batchnorm, self).__init__()
        
        self.cnn_layers = Sequential(
            BatchNorm2d(num_features = 7),
            Conv2d(in_channels = 7, out_channels = 7, kernel_size = 3, stride = 1, padding = 1),
            ReLU(inplace = True),
            
            BatchNorm2d(num_features = 7),
            Conv2d(in_channels = 7, out_channels = 7, kernel_size = 3, stride = 1, padding = 1),
            ReLU(inplace = True),
            
            BatchNorm2d(num_features = 7),
            Conv2d(in_channels = 7, out_channels = 60, kernel_size = 3, stride = 1, padding = 1),
            ReLU(inplace = True),
            
            BatchNorm2d(num_features = 60),
            Conv2d(in_channels = 60, out_channels = 100, kernel_size = 3, stride = 1, padding = 1),
            ReLU(inplace = True),

            BatchNorm2d(num_features = 100),
            Conv2d(in_channels = 100, out_channels = 175, kernel_size = 3, stride = 1, padding = 1),
            ReLU(inplace = True)
        )
        
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        return x