"""
Creates images with overlayed the segmentation prediction using a trained model (or ensemble of models).
"""

import os
import sys
sys.path.append(os.path.join('..', '..'))
sys.path.append(os.path.join(__file__, '..', '..', '..'))

import torch
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from natsort import natsorted
from imageio import imwrite

from utils.config import images_folder, models_folder
from utils.training_utils import get_prediction
from utils.colormap_utils import apply_turbo


# define model details
model_subfolder = 'pixel_level'
model_name = '0001_example_network_0'

# define image path
image_path = os.path.join(images_folder, r'datasets\frames_1\test\pos\BEDLUS_013_009_000.tiff')  # example frame

# define the detection settings
prob_threshold = 0.5        

# define the visualization settings
binary = False
minimum_opacity = 0.0
maximum_opacity = 1.0

# --------------------------------------------------------------------------------------------------

# configure the number of workers and the device
num_workers = 0 if sys.platform == 'win32' else 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device\n')

# create the prediction folder if it does not exist already
prediction_folder = os.path.join(models_folder, model_subfolder, model_name, 'images_with_predictions')
if not os.path.exists(prediction_folder):
    os.mkdir(prediction_folder)

# first check how many models are available
dirs = natsorted(os.listdir(os.path.join(models_folder, model_subfolder, model_name)))
settings_names = [d for d in dirs if d.startswith('experiment_settings') and d.endswith('.pkl')]
names = [d for d in dirs if d.startswith('model') and d.endswith('.pth')]

# load the image
image = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))/255
image_in_batch = image.repeat(1, 3, 1, 1).type(torch.FloatTensor).to('cuda')

# initialize a variable to store the model predictions
y_pred_all = None

# loop over the models
for name, settings_name in zip(names, settings_names):

    print(f'Model: {name}')

    # load the experiment settings
    settings = pd.read_pickle(os.path.join(models_folder, model_subfolder, model_name, settings_name))

    # check if the model is a segmentation model
    if settings['task'] != 'segmentation':
        raise ValueError('Instance detection evaluation requires a segmentation model.')

    # load the model
    model = torch.load(os.path.join(models_folder, model_subfolder, model_name, name))
    model.eval()

    # get the prediction
    y_pred = get_prediction(model(image_in_batch))
    y_pred = y_pred[0] if settings['aux_classification'] else y_pred
    y_pred = torch.softmax(y_pred, dim=1).to('cpu')

    if y_pred_all == None:
        y_pred_all = y_pred
    else:
        y_pred_all += y_pred

# get the average of the model predictions and apply a threshold
y_pred_all /= len(names)
pred = y_pred_all.to('cpu').detach()[0, 1, ...]
thresholded_pred = torch.where(pred >= prob_threshold, pred, torch.zeros_like(pred))

# prepare visualization 
if binary == True:
    opacity_map = torch.where(thresholded_pred >= prob_threshold, 1, 0)[..., None].repeat(1, 1, 3)
else:
    opacity_map = ((thresholded_pred * (maximum_opacity-minimum_opacity)) + minimum_opacity)[..., None].repeat(1, 1, 3)

image = (image[..., None].repeat(1, 1, 3) * 255).type(torch.IntTensor)
prediction = torch.from_numpy(apply_turbo(thresholded_pred, 0.1, 0.9) * 255).type(torch.IntTensor)

# create the combined image with prediction using the opacity map
image_with_prediction = (opacity_map * prediction + (1-opacity_map) * image).type(torch.uint8)

# save the image in the prediction folder
imwrite(os.path.join(prediction_folder, os.path.split(os.path.splitext(image_path)[0])[-1]+f'_{prob_threshold}_{minimum_opacity}_{maximum_opacity}.png'), image_with_prediction)

# create the figure
plt.imshow(image_with_prediction)
plt.title(os.path.split(image_path)[1])
plt.axis('off')
plt.show()