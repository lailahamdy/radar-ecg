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
from temporal_encoder import TemporalEncoder
class ECG_Encoder(nn.Module):    
    def __init__(self ,N , C , T ):
        super(ECG_Encoder, self).__init__()   
#         self.batch_size = batch_size
        self.num_signals = N
        self.channels = C
        self.T = T
        
        self.TemporalEncoder = TemporalEncoder(self.num_signals, self.channels, self.T)
        self.transformer = ExtendedViTModel()
        self.lstm_decoder = LSTM_TCN()
        
        
#         self.final_fc_layer =  nn.Linear(4*501, 500)
        
    #output_array = np.einsum('...ci->...co', input_array)
    def fusion(self,temporal_features ,spatial_features ):
        # Perform the dot product along the temporal dimension    
        #print(f"temporal_features: {temporal_features.size()}")
        #print(f"spatial_features: {spatial_features.size()}")
        fused_cardiac_feature = np.einsum('...ijk, ...ijk -> ...jk', temporal_features.detach().numpy(),spatial_features.detach().numpy())
        fused_cardiac_feature = torch.tensor(fused_cardiac_feature)
        #print("fused_cardiac_feature" ,fused_cardiac_feature.size())
        return fused_cardiac_feature    
    
    def forward(self ,inputs):  
        
        temporal_features , input_spatial = self.TemporalEncoder(inputs)
        
        spatial_features = self.transformer(input_spatial)
        
        fused_cardiac_feature = self.fusion(temporal_features ,spatial_features )
#         print(f"fused_cardiac_feature: {fused_cardiac_feature.size()}")
        output = self.lstm_decoder(fused_cardiac_feature)
        print("LSTM output", output.size())
        
#         flattened = torch.flatten(fused_cardiac_feature, start_dim=1) 
        
#         final_fc_layer =  nn.Linear(flattened.size(1), 500) 
        
#         encoder_output = final_fc_layer(flattened)  

        
        return output
