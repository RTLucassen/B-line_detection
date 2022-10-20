"""
Implementation of network architectures used by other papers in automated interpretation of lung ultrasound.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# -------------------------------------------------------------------------------------------------------

class VGG16_Born_etal(nn.Module):
    """
    Implementation of VGG16 architecture used by Born et al. in: https://www.mdpi.com/2076-3417/11/2/672, 
    based on: https://github.com/jannisborn/covid19_ultrasound/blob/master/pocovidnet/pocovidnet/model.py 
    """
    def __init__(self, N_classes: int, pretrained: bool):
        """
        N_classes:  number of output channels to predict.
        pretrained:  indicates whether pretrained parameters should be used to initialize the model.
        """
        super().__init__()
        self.encoder = torchvision.models.vgg16(pretrained=pretrained, progress=True).features
        self.head = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(2*3*512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, N_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

# -------------------------------------------------------------------------------------------------------

class STN_Roy_etal(nn.Module):
    """
    Implementation of STN with CNN architecture used by Roy et al. in: https://ieeexplore.ieee.org/document/9093068, 
    based on: https://github.com/mhug-Trento/DL4covidUltrasound
    """
    def __init__(self, img_size, in_channels, nclasses, fixed_scale=True):
        super(STN_Roy_etal, self).__init__()

        self.img_size = img_size
        self.fixed_scale = fixed_scale
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),  # 48 corresponds to the number of input features it
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # IN remains unchanged during any pooling operation
            #nn.Dropout(p=0.3)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.AvgPool2d(kernel_size=4)  # paper: 8
        )

        self.block1_stn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),  # 48 corresponds to the number of input features it
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # IN remains unchanged during any pooling operation
            #nn.Dropout(p=0.3)
        )

        self.block2_stn = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block3_stn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block4_stn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block5_stn = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        )

        self.block6_stn = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block7 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Linear(256, nclasses)

        if fixed_scale: # scaling is kept fixed, only translation is learned
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(128 * self.img_size[0]//(2**5) * self.img_size[1]//(2**5), 32),
                nn.ReLU(True),
                nn.Linear(32, 4)  # predict just translation params
            )
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([0.3, 0.3, 0.2, 0.2], dtype=torch.float))
        else: # scaling, rotation and translation are learned
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(128 * self.img_size[0]//(2**5) * self.img_size[1]//(2**5), 32),
                nn.ReLU(True)
            )
            self.trans = nn.Linear(32, 4)  # predict translation params
            self.scaling = nn.Linear(32, 2)  # predict the scaling parameter
            self.rotation = nn.Linear(32, 4)  # predict the rotation parameters

            # Initialize the weights/bias with some priors
            self.trans.weight.data.zero_()
            self.trans.bias.data.copy_(torch.tensor([0.3, 0.3, 0.2, 0.2], dtype=torch.float))

            self.scaling.weight.data.zero_()
            self.scaling.bias.data.copy_(torch.tensor([0.5, 0.75], dtype=torch.float))

            self.rotation.weight.data.zero_()
            self.rotation.bias.data.normal_(0, 0.1)

    # Spatial transformer network forward function
    def stn(self, x):
        scaling = 0 # dummy variable for just translation
        xs = self.block1_stn(x)
        xs = self.block2_stn(xs)
        xs = self.block3_stn(xs)
        xs = self.block4_stn(xs)
        xs = self.block5_stn(xs)
        xs = self.block6_stn(xs)
        xs = xs.view(-1, 128 * self.img_size[0]//(2**5) * self.img_size[1]//(2**5))

        if self.fixed_scale:
            trans = self.fc_loc(xs)
            bs = trans.shape[0]
            trans_1, trans_2 = torch.split(trans, split_size_or_sections=trans.shape[1] // 2, dim=1)
            # prepare theta for each resolution
            theta_1 = torch.cat([(torch.eye(2, 2, device='cuda') * 0.5).view(1, 2, 2).repeat(bs, 1, 1),
                                 trans_1.view(bs, 2, 1)], dim=2)
            theta_2 = torch.cat([(torch.eye(2, 2, device='cuda') * 0.75).view(1, 2, 2).repeat(bs, 1, 1),
                                 trans_1.view(bs, 2, 1)], dim=2)
        else:
            xs = self.fc_loc(xs)
            # predict the scaling params
            scaling = F.sigmoid(self.scaling(xs))
            scaling_1, scaling_2 = torch.split(scaling, split_size_or_sections=scaling.shape[1] // 2, dim=1)
            # predict the translation params
            trans = self.trans(xs)
            bs = trans.shape[0]
            trans_1, trans_2 = torch.split(trans, split_size_or_sections=trans.shape[1] // 2, dim=1)
            # predict the rotation params
            rot = self.rotation(xs)
            rot_1, rot_2 = torch.split(rot, split_size_or_sections=rot.shape[1] // 2, dim=1)
            # prepare theta for each resolution
            rot_1 = torch.ones(2, 2, device='cuda').fill_diagonal_(0).view(1, 2, 2).repeat(bs, 1, 1) * rot_1.view(bs, 2, 1)
            rot_2 = torch.ones(2, 2, device='cuda').fill_diagonal_(0).view(1, 2, 2).repeat(bs, 1, 1) * rot_2.view(bs, 2, 1)
            # add to the scaling params
            rot_1 = rot_1 + torch.eye(2, 2, device='cuda').view(1, 2, 2) * scaling_1.view(bs, 1, 1)
            rot_2 = rot_2 + torch.eye(2, 2, device='cuda').view(1, 2, 2) * scaling_2.view(bs, 1, 1)
            # prepare the final theta
            theta_1 = torch.cat([rot_1, trans_1.view(bs, 2, 1)], dim=2)
            theta_2 = torch.cat([rot_2, trans_1.view(bs, 2, 1)], dim=2)

        # get the shapes
        bs, c, _ , _ = x.size()
        h , w = self.img_size[0] // 2, self.img_size[1] // 2
        stn_out_size = (bs, c, h, w)

        # apply transformations
        grid_1 = F.affine_grid(theta_1, stn_out_size)
        grid_2 = F.affine_grid(theta_2, stn_out_size)
        x_1 = F.grid_sample(x, grid_1)
        x_2 = F.grid_sample(x, grid_2)
        x = torch.cat([x_1, x_2], dim=0)

        return x, scaling

    def forward(self, x, domains=None):
        xs, scaling = self.stn(x)  # transform the input
        x = self.block1(xs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.avg_pool2d(x, x.shape[-2])
        x = x.view(x.shape[0], -1)  # reshape the tensor
        x = F.dropout(self.block7(x), training=self.training)
        x = self.out(x)
        return x

# -------------------------------------------------------------------------------------------------------

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out

def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)

class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            return out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
            return out

class UNetEncoder3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, input_channels, act='relu'):
        super(UNetEncoder3D, self).__init__()

        self.down_tr64 = DownTransition(input_channels,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

    def forward(self, x):
        self.out64 = self.down_tr64(x)
        self.out128 = self.down_tr128(self.out64)
        self.out256 = self.down_tr256(self.out128)
        self.out512 = self.down_tr512(self.out256)
        
        return self.out512

class UNet3D_Born_etal(nn.Module):
    """
    Implementation of 3D U-Net Encoder and classification head with pretrained weights from Models Genesis 
    by Zhou et al. in: https://github.com/MrGiovanni/ModelsGenesis/tree/master/pytorch,
    used by Born et al. in: https://www.mdpi.com/2076-3417/11/2/672, 
    """    
    def __init__(self, pretrained, input_channels, N_classes, model_params_path = r'C:\Users\rlucasse\Documenten\repositories\B-line_detection\models\Genesis_Chest_CT.pt'):
        super(UNet3D_Born_etal, self).__init__()

        # define base model and fully connected layers
        self.encoder = UNetEncoder3D(input_channels)
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, N_classes, bias=True)

        # load the pretrained parameters if specified
        if pretrained:
            # check if the parameter file exists
            if not os.path.exists(model_params_path):
                raise IOError('Models Genesis pretrained parameters file does not exists. Please check if the path is correct.')
            # load the weights
            checkpoint = torch.load(model_params_path)
            state_dict = checkpoint['state_dict']
            encoder_state_dict = {}
            for key in state_dict.keys():
                # only copy the encoder parameters
                if 'down' in key:
                    if 'down_tr64.ops.0.conv1.weight' in key:
                        encoder_state_dict[key.replace("module.", "")] = torch.tile(state_dict[key], (1, input_channels, 1, 1, 1))
                    else:
                        encoder_state_dict[key.replace("module.", "")] = state_dict[key]
            self.encoder.load_state_dict(encoder_state_dict)

    def forward(self, x):
        encoder_output = self.encoder(x)
        # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
        # where N = batch_size, C = channels, H = height, and W = Width
        out_glb_avg_pool = F.avg_pool3d(encoder_output, kernel_size=encoder_output.size()[2:]).view(encoder_output.size()[0],-1)
        linear_out = self.dense_1(out_glb_avg_pool)
        output = self.dense_2(F.relu(linear_out))

        return output