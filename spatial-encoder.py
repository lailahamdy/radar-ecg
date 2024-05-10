import torch
import torch.nn as nn
import numpy as np
import math

from torch.utils.data import Dataset
from transformers import  ViTConfig,ViTModel
import cv2
from random import Random
import matplotlib.pyplot as plt

class ExtendedViTModel(nn.Module):
    def __init__(self ):
        super(ExtendedViTModel, self).__init__()        # Load the ViT model
        configuration = ViTConfig(image_size=50,num_channels=1)
        self.vit_model = ViTModel(configuration)
        
#         self.batch_size = batch_size
          
    def forward(self, inputs):        
        outputs = self.vit_model(inputs)
        batch_size , _ ,_,_ = inputs.size()
        flattened = torch.flatten(outputs.last_hidden_state, start_dim=1)  
        #print("Last #print " , flattened.size())
        
        fc_layer =  nn.Linear(flattened.size(1), 100200)  
        
        output = fc_layer(flattened)
        
        return output.view(batch_size,50 , 4, 501).float()  # Convert the output to Float
# extended_vit = ExtendedViTModel(batch_size)
