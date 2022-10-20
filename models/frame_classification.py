"""
Loads implementation of classification models from torchvision:
(https://pytorch.org/vision/stable/models.html#torchvision.models.classification)
and Luke Melas:
(https://github.com/lukemelas/EfficientNet-PyTorch)


Modifications that were made include:
    - Number of input channels can be selected instead of the default 3 channel RGB channels for Pytorch implementations.
      (which means that the pretrained weights cannot be used for that layer)
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import timm
import torch
import torchvision
import torch.nn as nn
from torchinfo import summary
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_same_padding_conv2d

from utils.model_utils import change_input, initialize
from models.architectures_by_others import VGG16_Born_etal, STN_Roy_etal


def get_frame_classification_model(
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
    # check if input channels equals 3 (for RGB) if pretrained is True:
    if not isinstance(pretrained, bool):
        raise ValueError('Non-boolean argument for pretrained.')
    elif pretrained and input_channels != 3:
        raise ValueError('Input channels must be 3 if pretrained weights are used.') 
    
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained, progress=True)
        # replace the output layer to include only two classes
        model.fc = nn.Linear(in_features=512, out_features=N_classes)
        # change the first layer to have the desired number of input channels if pretrained is False
        if pretrained == False:
            model.conv1 = change_input(model.conv1, input_channels)       
    
    elif model_name == 'densenet121':
        model = torchvision.models.densenet121(pretrained=pretrained, progress=True)
        # replace the output layer to include only two classes
        model.classifier = nn.Linear(in_features=1024, out_features=N_classes)
        # change the first layer to have the desired number of input channels if pretrained is False
        if pretrained == False:
            model.features.conv0 = change_input(model.features.conv0, input_channels)      
    
    elif 'efficientnet' in model_name:
        if pretrained:
            model = EfficientNet.from_pretrained(model_name, num_classes=N_classes)
        else:
            model = EfficientNet.from_name(model_name, {"num_classes": N_classes})
            # change the first layer to have the desired number of input channels if pretrained is False
            Conv2d = get_same_padding_conv2d(model._global_params.image_size)
            model._conv_stem = Conv2d(input_channels, model._conv_stem.out_channels, kernel_size=(3, 3), stride=(2, 2), bias=False)
    
    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=pretrained, progress=True)
        # replace the output layer to include only two classes
        model.classifier[1] = nn.Linear(in_features=1280, out_features=N_classes)
        # change the first layer to have the desired number of input channels if pretrained is False
        if pretrained == False:
            model.features[0][0] = change_input(model.features[0][0], input_channels)  

    elif model_name == 'ViT':
        model = timm.create_model('vit_tiny_patch16_384', pretrained=pretrained, num_classes=N_classes)
        # change the first layer to have the desired number of input channels if pretrained is False
        if pretrained == False:
            model.patch_embed.proj = change_input(model.patch_embed.proj, input_channels) 

    elif model_name == 'stn_roy_etal':
        model = STN_Roy_etal(img_size=(256,384), in_channels=input_channels, nclasses=N_classes) 

    elif model_name == 'vgg16_born_etal':
        model = VGG16_Born_etal(N_classes=N_classes, pretrained=pretrained)
        # change the first layer to have the desired number of input channels if pretrained is False
        if pretrained == False:
           model.encoder[0] = change_input(model.encoder[0], input_channels)

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
    input_shape = (1, input_channels, 256, 384)
    display_depth = 5

    model = get_frame_classification_model('efficientnet-b0', input_channels, N_classes, pretrained)
    if True: summary(model, input_shape, depth=display_depth, col_names=["input_size", "output_size", "num_params"]) # number of weights for is efficientnet not correctly displayed
    
    if False: print(model)

    # print the trainable parameters (weights and biases)
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