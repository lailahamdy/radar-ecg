import torch
import torch.nn as nn
import numpy as np
import math

from torch.utils.data import Dataset
from transformers import  ViTConfig,ViTModel
import cv2
from random import Random
import matplotlib.pyplot as plt
from custom_layers import CustomConv1D, CustomMaxPooling1D, CustomConv1DTranspose, CustomConv1DTranspose, CustomBatchNorm1d

rcg = np.expand_dims(np.load("./ECG Module Data/RCG_test_568_50_500.npy"), axis = 2).astype("float32")
# rcg = np.expand_dims(np.array([cv2.resize(rcg[i,:,:,], (50, 50)) for i in range(568)]), axis = 2)
labels = np.load("./ECG Module Data/Raw_ECG_Labels/ECG_test_568_500.npy").astype("float32")

class CustomDataset(Dataset):    
    def __init__(self, data, labels):
        self.data = data        
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):        
        x = self.data[index]
        y = self.labels[index]        
        return x, y
# Create an instance of CustomDataset
dataset = CustomDataset(rcg, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

class TemporalEncoder(nn.Module):
    def __init__(self, num_signals,channels,T):
        super(TemporalEncoder, self).__init__()
        self.num_signals = num_signals
        self.T = T
        self.channels = channels

        self.batchnorm1 = CustomBatchNorm1d(channels)
        self.batchnorm2 = CustomBatchNorm1d(channels)
        self.batchnorm3 = CustomBatchNorm1d(channels)
        self.batchnorm4 = CustomBatchNorm1d(channels)


        self.maxpool = CustomMaxPooling1D(pool_size=2, strides=2)

        self.conv1d_1_0 = CustomConv1D(in_channels=1, out_channels=4, kernel_size=7, padding='same')
        self.conv1d_1_1 = CustomConv1D(in_channels=4, out_channels=4, kernel_size=7, padding='same')

        self.conv1d_2_0 = CustomConv1D(in_channels=4, out_channels=8, kernel_size=7, padding='same')
        self.conv1d_2_1 = CustomConv1D(in_channels=8, out_channels=8, kernel_size=7, padding='same')

        
        self.conv1d_3_0 = CustomConv1D(in_channels=8, out_channels=16, kernel_size=7, padding='same')
        self.conv1d_3_1 = CustomConv1D(in_channels=16, out_channels=16, kernel_size=7, padding='same')

        self.conv1d_4_0 = CustomConv1D(in_channels=16, out_channels=32, kernel_size=7, padding='same')
        self.conv1d_4_1 = CustomConv1D(in_channels=32, out_channels=32, kernel_size=7, padding='same')
        
        self.conv1d_5 = CustomConv1D(in_channels=32, out_channels=32, kernel_size=7, padding='same')
        self.conv1d_6 = CustomConv1D(in_channels=16, out_channels=16, kernel_size=7, padding='same')
        self.conv1d_7 = CustomConv1D(in_channels=8, out_channels=8, kernel_size=7, padding='same')
        self.conv1d_8 = CustomConv1D(in_channels=4, out_channels=4, kernel_size=7, padding='same')
        #output_size = ((input_size - 1) * stride) - (2 * padding) + kernel_size

        self.transpose_conv1d_1 = CustomConv1DTranspose(padding = 14, input_channels = 32, filters=32, kernel_size=7, strides=2)
        self.transpose_conv1d_2 = CustomConv1DTranspose(padding = 41, input_channels = 32, filters=16, kernel_size=7, strides=2)
        self.transpose_conv1d_3 = CustomConv1DTranspose(padding = 2, input_channels = 16, filters=8, kernel_size=7, strides=2)
        self.transpose_conv1d_4 = CustomConv1DTranspose(padding = 3, input_channels = 8, filters=4, kernel_size=7, strides=2)
        
        self.relu = nn.ReLU().float
        self.flatten = nn.Flatten().float
        self.dense = nn.Linear(50*32*62, 50*50).float


    def CNN_Block_1(self, input_tensor):
        
        # 1st layer
#         #print(f"input_tensor: {input_tensor.size()}")
        
        x = self.conv1d_1_0(input_tensor) 
        x = self.relu()(x) 
        x = self.conv1d_1_1(x)
        x = self.relu()(x)
        x = self.batchnorm1(x)
        x = self.maxpool(x)
        
#         #print(f"1st layer: {x.size()}")

        # 2nd layer
        x = self.conv1d_2_0(x)
        x = self.relu()(x)
        x = self.conv1d_2_1(x)
        x = self.relu()(x)
        x = self.batchnorm2(x)
        x = self.maxpool(x)
        #print(f"2nd layer: {x.size()}")

        # 3rd layer
        x = self.conv1d_3_0(x)
        x = self.relu()(x)
        x = self.conv1d_3_1(x)
        x = self.relu()(x)
        x = self.batchnorm3(x)
        x = self.maxpool(x)
        
        #print(f"3rd layer: {x.size()}")

        # 4th layer
        x = self.conv1d_4_0(x)
        x = self.relu()(x)
        x = self.conv1d_4_1(x)
        x = self.batchnorm4(x)
        
        y = self.flatten()(x)
        
        #print(f"flatten y{y.size()}")
        y = self.dense()(y)
        
        #print(f"final x: {x.size()}")
        #print(f"final y: {y.size()}")
        
        return x,y


def CNN_Block_2(self, x):
        # 1st layer
        
        x = self.transpose_conv1d_1(x)
        x = self.conv1d_5(x)
        x = self.relu()(x)
        x = self.conv1d_5(x)
        x = self.relu()(x)

        # 2nd layer
        
        x = self.transpose_conv1d_2(x)
        x = self.conv1d_6(x)
        x = self.relu()(x)
        x = self.conv1d_6(x)
        x = self.relu()(x)

        # 3rd layer
        
        x = self.transpose_conv1d_3(x)
        x = self.conv1d_7(x)
        x = self.relu()(x)
        x = self.conv1d_7(x)
        x = self.relu()(x)

        # 4th layer
        
        x = self.transpose_conv1d_4(x)
        x = self.conv1d_8(x)
        x = self.relu()(x)
        x = self.conv1d_8(x)
        x = self.relu()(x)

        return x

    def forward(self, input_tensor):
        x,y = self.CNN_Block_1(input_tensor)
        batch_size, num_signals, channels, T = x.size()
        #print(f"batch_size:{batch_size}")
        #print(f"batch_size:{num_signals}")
        #print(f"batch_size:{channels}")
        #print(f"batch_size:{T}")
        y = y.view(batch_size, 1, num_signals, num_signals)

        x = self.CNN_Block_2(x)

        return x , y
