import torch
import torch.nn as nn
import numpy as np
import math

class CustomConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(CustomConv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        

    def forward(self, inputs):
        # Get the input shape
        batch_size, num_signals, channels, T = inputs.size()

        # Reshape the inputs to (batch_size * num_signals, channels, T)
        inputs_reshaped = inputs.view(-1, channels, T)

        # Apply the convolution operation
        convolved = self.conv1d(inputs_reshaped)

        # Reshape the convolved tensor back to (batch_size, num_signals, channels', T')
        T_convolved = convolved.size(-1)
        convolved_reshaped = convolved.view(batch_size, num_signals, -1, T_convolved)

        return convolved_reshaped

  class CustomMaxPooling1D(nn.Module):
    def __init__(self, pool_size=2, strides=2):
        super(CustomMaxPooling1D, self).__init__()
        self.pool_size = pool_size
        self.strides = strides

    def forward(self, inputs):
        # Get the input shape
        batch_size, num_signals, channels, T = inputs.size()

        # Reshape the inputs to (batch_size * num_signals, channels, T)
        inputs_reshaped = inputs.view(-1, channels, T)

        # Apply the max pooling operation
        pooled = nn.functional.max_pool1d(inputs_reshaped, kernel_size=self.pool_size, stride=self.strides)

        # Reshape the pooled tensor back to (batch_size, num_signals, channels, T')
        T_pooled = pooled.size(-1)
        pooled_reshaped = pooled.view(batch_size, num_signals, -1, T_pooled)
        return pooled_reshaped
      
class CustomConv1DTranspose(nn.Module):
    def __init__(self, padding, input_channels ,filters, kernel_size, strides=1, activation='relu'):
        super(CustomConv1DTranspose, self).__init__()
        self.input_channels = input_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=input_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding
        )

    def forward(self, inputs):
        
        inputs = inputs.contiguous()

        # Get the input shape
        batch_size, num_signals, channels, T = inputs.size()

        # Reshape the inputs to (batch_size * num_signals, channels, T)
        inputs_reshaped = inputs.view(-1, channels, T)

        # Apply the transpose convolution operation
        conv_transposed = self.conv_transpose(inputs_reshaped)

        # Reshape the transposed tensor back to (batch_size, num_signals, filters, T')
        T_transposed = conv_transposed.size(-1)
        transposed_reshaped = conv_transposed.view(batch_size, num_signals, self.filters, T_transposed)

        return transposed_reshaped    

class CustomBatchNorm1d(nn.Module):
    def __init__(self, channels):
        super(CustomBatchNorm1d, self).__init__()
        self.channels = channels
        
        self.batchnorm = nn.BatchNorm1d(channels)

    def forward(self, x):
        batch_size = x.size(0)
        num_signals = x.size(1)
        T = x.size(3)
        
        # Reshape x to (batch_size * num_signals, channels, T)
        x_reshaped = x.view(-1, self.channels, T)

        # Apply BatchNorm1d on the reshaped tensor
        output = self.batchnorm(x_reshaped)

        # Reshape the output back to the original shape
        output = output.view(batch_size, num_signals, -1, T)

        return output
