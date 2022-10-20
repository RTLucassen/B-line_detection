"""
Loads implementation of segmentation models from segmentation-model-pytorch:
(https://github.com/qubvel/segmentation_models.pytorch)
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import torch
import torchvision
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchinfo import summary

from models.unet import UNet
from utils.model_utils import initialize


def get_segmentation_model(
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
        aux_classification:  indicates whether an auxiliary classifier should be attached to the encoder.

    Returns:
        model:  implementation of selected model.
    """
    # check if the input channels equals 3 (for RGB) if pretrained is True:
    if not isinstance(pretrained, bool):
        raise ValueError('Non-boolean argument for pretrained.')
    if pretrained and input_channels != 3:
        raise ValueError('Input channels must be 3 if pretrained weights are used.') 

    if model_name == 'unet':
        # initialize U-Net
        model = UNet(
            input_channels = input_channels,
            N_classes = N_classes,
            filters = 16,
            batch_norm = True,
        )
    else:
        # split the model name into the name of the encoder and decoder and check if the combination is valid
        encoder_name, decoder_name = model_name.split('_')
        if encoder_name not in ['resnet18', 'densenet121', 'efficientnet-b0']:
            raise ValueError('Encoder name not recognized')
        elif decoder_name not in ['unet', 'deeplabv3plus']:
            raise ValueError('Decoder name not recognized')
        
        # specify the settings for the auxiliary classifier connected to the encoder
        aux_params = {'pooling': 'avg', 'dropout': 0.0, 'classes': N_classes} if aux_classification == True else None

        if decoder_name == 'unet':
            encoder_weights = 'imagenet' if pretrained else None
            model = smp.Unet(
                encoder_name=encoder_name, 
                encoder_depth=5, 
                encoder_weights=encoder_weights, 
                decoder_channels=(256, 128, 64, 32, 16),
                decoder_use_batchnorm=True, 
                decoder_attention_type=None,
                in_channels=input_channels, 
                classes=N_classes,
                aux_params=aux_params
            )
        elif decoder_name == 'deeplabv3plus':
            encoder_weights = 'imagenet' if pretrained else None
            model = smp.DeepLabV3Plus(
                encoder_name=encoder_name, 
                encoder_depth=5, 
                encoder_weights=encoder_weights, 
                in_channels=input_channels, 
                classes=N_classes,
                aux_params=aux_params
            )
        # initialize the parameters of the decoder and if not pretrained also the encoder
        init = lambda layer: initialize(layer, init_method)
        model.decoder.apply(init)
        if pretrained == False:
            print(f'Randomly initializing the encoder parameters using the {init_method} method.')
            model.encoder.apply(init)

    return model


if __name__ == '__main__':
    # define the model parameters
    input_channels = 3
    N_classes = 2
    pretrained = True
    aux_classification = False
    input_shape = (1, input_channels, 256, 384)
    display_depth = 5

    model = get_segmentation_model('resnet18_unet', input_channels, N_classes, pretrained, aux_classification=aux_classification)
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
        