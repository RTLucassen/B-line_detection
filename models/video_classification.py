"""
Loads implementation of classification models from torchvision:
(https://pytorch.org/vision/stable/models.html#torchvision.models.classification)

Modifications that were made include:
    - Number of input channels can be selected instead of the default 3 channel RGB channels for Pytorch implementations.
      (which means that the pretrained weights cannot be used for that layer)
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import torch
import torchvision
import torch.nn as nn
from torchinfo import summary

from utils.model_utils import change_input, initialize
from models.architectures_by_others import UNet3D_Born_etal


def get_video_classification_model(
    model_name: str, 
    input_channels: int, 
    N_classes: int, 
    pretrained: bool,
    init_method: str = 'kaiming_normal',
    aux_classification: bool = False
) -> nn.Module:
    """
    Args:
        model_name:  name of model used by Pytorch.
        input_channels:  number of input channels.
        N_classes:  number of output channels to predict.
        pretrained:  indicates whether pretrained parameters should be used to initialize the model.
                     (non-pretrained networks are initialized with the according to the kaiming normal method by default)
        init_method:  initialization method for weights if pretrained is False.
        aux_classification:  Not used for video classification models.

    Returns:
        model:  implementation of selected model.
    """
    # check if the input channels equals 3 (for RGB) if pretrained is True:
    if not isinstance(pretrained, bool):
        raise ValueError('Non-boolean argument for pretrained.')
    elif pretrained and input_channels != 3:
        raise ValueError('Input channels must be 3 if pretrained weights are used.') 

    if model_name == 'resnet18_3d':
        model = torchvision.models.video.r3d_18(pretrained=pretrained, progress=True)
        # replace the output layer to include only two classes
        model.fc = nn.Linear(in_features=512, out_features=N_classes)
        # change the first layer to have the desired number of input channels if pretrained is False
        if pretrained == False:
            model.stem[0] = change_input(model.stem[0], input_channels) 
    
    elif model_name == 'resnet18_2plus1d':
        model = torchvision.models.video.r2plus1d_18(pretrained=pretrained, progress=True)
        # replace the output layer to include only two classes
        model.fc = nn.Linear(in_features=512, out_features=N_classes)
        # change the first layer to have the desired number of input channels if pretrained is False
        if pretrained == False:
            model.stem[0] = change_input(model.stem[0], input_channels)       
        
    elif model_name == 'unet_3d_born_etal':
        model = UNet3D_Born_etal(pretrained=pretrained, input_channels=input_channels, N_classes=N_classes)
    else:
        raise ValueError('Model name not recognized.')

    # initialize the parameters if not pretrained
    if pretrained == False:
        print(f'Randomly initializing the model parameters using the {init_method} method.')
        init = lambda layer: initialize(layer, init_method)
        model.apply(init)

    return model


if __name__ == '__main__':
    # define the model parameters
    input_channels = 3
    N_classes = 2
    pretrained = True
    input_shape = (1, input_channels, 16, 256, 384)  
    display_depth = 5

    model = get_video_classification_model('resnet18_3d', input_channels, N_classes, pretrained)
    if True: summary(model, input_shape, depth=display_depth, col_names=["input_size", "output_size", "num_params"]) # number of weights for efficientnet is not correctly displayed
    
    if False: print(model)

    # list the trainable parameters (weights and biases)
    if False:
        for name, param in model.named_parameters():
            print(name,':', param.shape)
    
    # print the number of weights, biases, and total number of parameters
    if True:
        weights = biases = 0
        for name, param in model.named_parameters():
            N = len(param.flatten())
            if name.split('.')[-1] == 'weight':
                weights += N
            else:
                biases += N
        print(f'Weights: {weights}, Biases: {biases}, Total: {weights + biases}')