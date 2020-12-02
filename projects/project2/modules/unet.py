import os, sys, math, random, itertools

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers import Conv2D, MaxPool2D, ReLU, UpSampling2D, Concatenate
from tensorflow.keras import Model
from tensorflow_addons.layers import GroupNormalization

NUM_ROWS, NUM_COLS = 256, 256

class UNet_block_down(Model):
    
    def __init__(
        self, out_channel, kernel_size=3, padding="same",
        data_format="channels_last", down_sample=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.data_format = data_format
        axis = -1 if data_format=="channels_last" else 1

        self.down_sample = down_sample
        self.conv1 = Conv2D(
            filters=out_channel, kernel_size=kernel_size,
            padding=padding, data_format=data_format
        )
        self.bn1 = GroupNormalization(groups=8, axis=axis)
        self.conv2 = Conv2D(filters=out_channel, kernel_size=kernel_size, padding=padding, data_format=data_format)
        self.bn2 = GroupNormalization(groups=8, axis=axis)
        self.max_pool2d = MaxPool2D(pool_size=2, strides=2, data_format=data_format)
        self.relu = ReLU()
    
    def __call__(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.down_sample:
            x = self.max_pool2d(x)
        return x
    

class UNet_block_up(Model):
    
    def __init__(
        self, out_channel, kernel_size=3, padding="same", 
        data_format="channels_last", up_sample=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.data_format = data_format
        axis = -1 if data_format=="channels_last" else 1
        
        self.upsample = UpSampling2D(size=2, data_format=data_format, interpolation="bilinear")
        self.concatenate = Concatenate(axis=axis)
        self.conv1 = Conv2D(
            filters=out_channel, kernel_size=kernel_size,
            padding=padding, data_format=data_format)
        self.bn1 = GroupNormalization(groups=8, axis=axis)
        self.conv2 = Conv2D(filters=out_channel, kernel_size=kernel_size, padding=padding, data_format=data_format)
        self.bn2 = GroupNormalization(groups=8, axis=axis)
        self.max_pool2d = MaxPool2D(pool_size=2, strides=2, data_format=data_format)
        self.relu = ReLU()
        self.up_sample = up_sample
    
    def __call__(self, saved_features, x):
        if self.up_sample:
            x = self.upsample(x)
        x = self.concatenate([x, saved_features])
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
        
    
class UNet(Model):
    
    def __init__(self, downsample=6, out_channel=3, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        self.downsample, self.out_channel = downsample, out_channel
        axis = -1 if data_format=="channels_last" else 1
        
        self.down1 = UNet_block_down(out_channel=16)
        self.down_blocks = [UNet_block_down(out_channel=2**(5+i)) for i in range(0, downsample)]
        
        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = Conv2D(bottleneck, kernel_size=3, padding="same")
        self.bn1 = GroupNormalization(8, axis=axis)
        self.mid_conv2 = Conv2D(bottleneck, kernel_size=3, padding="same")
        self.bn2 = GroupNormalization(8, axis=axis)
        self.mid_conv3 = Conv2D(bottleneck, kernel_size=3, padding="same")
        self.bn3 = GroupNormalization(8, axis=axis)
        
        self.up_blocks = [UNet_block_up(2**(4+i)) for i in range(0, downsample)]
        
        self.last_conv1 = Conv2D(16, kernel_size=3, padding="same")
        self.last_bn = GroupNormalization(8, axis=axis)
        self.last_conv2 = Conv2D(out_channel, kernel_size=1, padding="valid")
        self.relu = ReLU()
        
    def __call__(self, x):
        x = self.down1(x)
        saved_features = [x]
        for i in range(self.downsample):
            x = self.down_blocks[i](x)
            saved_features.append(x)
        
        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](saved_features[i], x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.relu(self.last_conv2(x))
        return x