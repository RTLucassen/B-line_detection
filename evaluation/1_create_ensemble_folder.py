"""
Creates folder for ensemble and copies all model files (.pth) to it.
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import torch
import torch.nn as nn
import pandas as pd
from shutil import copyfile
from natsort import natsorted

from utils.config import models_folder, info_folder

# define name that is used to group models
model_subfolder = ''
model_names = ['0001_example_network_0', '0002_example_network_1', '0003_example_network_2', '0004_example_network_3', '0005_example_network_4']
ensemble_name = 'ensemble_example_network'

# --------------------------------------------------------------------------------

# define a variable with the model directory
models_dir = models_folder if model_subfolder == None else os.path.join(models_folder, model_subfolder)

# raise an error if a selected model is not in the models directory
all_model_names = os.listdir(models_dir)
if model_names == 'all':
    model_names = natsorted(all_model_names)
else:
    for name in model_names:
        if name not in all_model_names:
            message = f'Model {name} is not available in the models directory'
            raise IOError(message)

print(f'The following {len(model_names)} models were selected:', model_names)

if len(model_names) > 0:

    label = None
    # load the experiment settings for each model to check if the same labels were used
    for model in model_names:
        settings = pd.read_pickle(os.path.join(models_dir, model, 'experiment_settings.pkl'))
        if label == None:
            label = settings['label_folder']
        elif label != settings['label_folder']:
            raise ValueError('Not all selected models were trained using the same labels.')

    # create the ensemble folder if it does not exist yet,
    # otherwise warn the user
    output_path = os.path.join(models_dir, ensemble_name)
    if os.path.exists(output_path):
        raise IOError('Output directory already exists.')
    else:
        os.mkdir(output_path)

    # create a dictionary to keep track of the model origin
    origin_dict = {'origin':[], 'model': [], 'settings':[]}

    # load models
    models = []

    # copy all model files to the ensemble folder
    for idx, model in enumerate(model_names):
        # copy model file and settings file
        copyfile(os.path.join(models_dir, model, 'model.pth'), os.path.join(output_path, f'model_{idx}.pth'))
        copyfile(os.path.join(models_dir, model, 'experiment_settings.pkl'), os.path.join(output_path, f'experiment_settings_{idx}.pkl'))
        # load the model and add it to the list
        models.append(torch.load(os.path.join(models_dir, model, 'model.pth')))
        # update the model and setting track record
        origin_dict['origin'].append(model)
        origin_dict['model'].append(f'model_{idx}.pth')
        origin_dict['settings'].append(f'experiment_settings_{idx}.pkl')

    # save a csv file with model origins
    pd.DataFrame.from_dict(origin_dict).to_csv(os.path.join(output_path, 'model_origins.csv'), index=False)
            