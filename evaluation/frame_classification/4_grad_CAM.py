"""
Creates a gradient-weighted class activation (Grad-CAM) map for an image using a trained model.
"""

import os
import sys
sys.path.append(os.path.join('..', '..'))
sys.path.append(os.path.join(__file__, '..', '..', '..'))

import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
from imageio import imwrite
from torchinfo import summary
from pytorch_grad_cam import GradCAM

from utils.config import models_folder, images_folder
from utils.colormap_utils import apply_turbo, text_phantom


# define forward hook 
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

if __name__ == '__main__':

    # define model evaluation details
    model_subfolder = 'frame_level'
    model_name = '0001_example_network_0'

    # define the image paths
    image_directory = os.path.join(images_folder, r'datasets\frames_1\test\pos')
    image_name = 'all'

    # show the visualization in a matplotlib figure
    visualize = False

    # define the layer for which to create the class activation map
    target_category = 1
    
    # specify if the class activation maps should be gradient-weighted (grad-CAM, else CAM is used)
    gradient_weighted = True

    # define the visualization settings
    lower_bound = 0
    upper_bound = 0.0968   # (based on validation set of an EfficientNet model, does not work well in combination with other models)
    minimum_opacity = 0.0
    maximum_opacity = 0.8
    add_pred = False

    # --------------------------------------------------------------------------------

    # determines which network is used (necessary to select the correct convolutional layer)
    use_ResNet18 = True if 'resnet' in model_name else False
    use_DenseNet121 = False if 'densenet' in model_name else False
    use_EfficientNet_b0 = False if 'efficientnet' in model_name else False

    # check if a valid network architecture was selected
    if use_ResNet18 == False and use_DenseNet121 == False and use_EfficientNet_b0 == False:
        raise ValueError('No valid network was selected')

    # load the model
    model_path = os.path.join(models_folder, model_subfolder, model_name, 'model.pth')
    model = torch.load(model_path)
    model.eval()  
    
    # create the CAM folder if it does not exist already
    cam_folder = os.path.join(models_folder, model_subfolder, model_name, 'CAM_images')
    if not os.path.exists(cam_folder):
        os.mkdir(cam_folder)

    # print a model summary and the layer names
    if False: 
        print(model)
        summary(model, (1, 3, 256, 384))

    # specify the target layer
    if use_ResNet18: # ResNet18
        activation_layer = model.layer4[-1].bn2
        weight_layer = model.fc

    if use_DenseNet121: # DenseNet121
        activation_layer = model.features[-1] 
        if gradient_weighted == False:
            raise ValueError('DenseNet121 does not use global average pooling and therefore only works with gradient weights.')

    if use_EfficientNet_b0: # EfficientNet-b0
        activation_layer = model._bn1 
        weight_layer = model._fc

    # Construct the grad-CAM object once, and then re-use it on many images
    cam_model = GradCAM(model=model, target_layer=activation_layer)

    # get the image names (in case all was selected)
    image_names = os.listdir(image_directory) if image_name == 'all' else [image_name]

    for name in tqdm(image_names):

        # load the image
        image_path = os.path.join(image_directory, name)
        image = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))/255
        image_in_batch = image.repeat(1, 3, 1, 1).type(torch.FloatTensor).to('cuda')

        # get the prediction
        prediction = float(torch.softmax(model(image_in_batch), dim=1).to('cpu')[0, 1])

        if gradient_weighted:
            # get the CAM map
            cam = torch.from_numpy(cam_model(input_tensor=image_in_batch, target_category=target_category))[0, ...]

        elif gradient_weighted == False:
            # attach a forward hook
            activation_layer.register_forward_hook(get_activation(activation_layer._get_name()))

            # let the model predict for the image
            with torch.no_grad():
                pred = model(image_in_batch)
                print(torch.softmax(pred, dim=1))
        
            # get the weights of the final fully-connected layer (for the selected class)
            weights = weight_layer.weight[target_category, ...]
            weights = weights[None, ..., None, None].repeat(1, 1, *image.shape)
            # get the activation maps
            activation_maps = activation[activation_layer._get_name()]
            activation_maps = torch.nn.functional.interpolate(activation_maps, scale_factor=32, mode='bilinear')

            # create the CAM
            cam = torch.sum(activation_maps * weights, dim=(0, 1))
            cam = cam.to('cpu').detach()

        # normalize cam
        # NOTE: the grad-CAM library by default normalizes each map based on the minimum and maximum, which in our opinion is undesired. 
        #       to correct for this, comment the corresponding section in the code of the library.
        lower = torch.min(cam) if lower_bound == 'min' else lower_bound
        upper = torch.max(cam) if upper_bound == 'max' else upper_bound

        if lower != None and upper != None:
            cam -= lower
            cam /= upper
            cam = torch.clip(cam, 0, 1)

        # prepare the visualization
        opacity_map = ((cam * (maximum_opacity-minimum_opacity)) + minimum_opacity)[..., None].repeat(1, 1, 3)   
        image_in_batch = image_in_batch.to('cpu').detach()[0, ...] * 255
        image_in_batch = image_in_batch.type(torch.IntTensor).permute(1, 2, 0)
        cam = torch.from_numpy(apply_turbo(cam, 0.1, 0.9) * 255).type(torch.IntTensor)

        # combine the image with the grad-CAM using the opacity map
        image_with_cam = (opacity_map * cam + (1-opacity_map) * image_in_batch).type(torch.uint8)

        # optionally add model the predicted probability as text to top left corner of image
        pred_text = (text_phantom(f'{prediction:0.2f}')*255).type(torch.uint8)
        if add_pred:
            image_with_cam[10:10+pred_text.shape[0], 10:10+pred_text.shape[1], :] = torch.where(pred_text > 0, pred_text, image_with_cam[10:10+pred_text.shape[0], 10:10+pred_text.shape[1], :])
            torch.clip(image_with_cam, 0, 255)

        # save the image in the CAM folder
        imwrite(os.path.join(cam_folder, os.path.split(os.path.splitext(image_path)[0])[-1]+'.png'), image_with_cam)

        torch.cuda.empty_cache()

        # create the figure for visualization
        if visualize:
            plt.imshow(image_with_cam)
            plt.title(os.path.split(image_path)[1])
            plt.axis('off')
            plt.show()

