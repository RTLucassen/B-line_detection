"""
Creates text file with model information
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import io
import torch
import pandas as pd
from tqdm import tqdm
from torchinfo import summary
from contextlib import redirect_stdout

from utils.config import models_folder

# define model specifics
model_subfolder = ''
model_names = '0001_example_network_0'
output_shape = (256, 384) 

# --------------------------------------------------------------------------------------------------

# configure device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device\n')

# define a variable with the model directory
model_dir = models_folder if model_subfolder == None else os.path.join(models_folder, model_subfolder)

# if equal to 'all', get all directories
if model_names == 'all':
    model_names = [d for d in os.listdir(model_dir)if os.path.isdir(os.path.join(model_dir, d))]
# if a single model name was given, add it to a list
elif isinstance(model_names, str):
    model_names = [model_names]

for model_name in tqdm(sorted(list(set(model_names)))):     
    # first check how many models are available
    dirs = os.listdir(os.path.join(model_dir, model_name))
    names = [d for d in dirs if d.startswith('model') and d.endswith('.pth')]
    settings_names = [d for d in dirs if d.startswith('experiment_settings') and d.endswith('.pkl')]

    # loop over the models
    for name, settings_name in zip(names, settings_names):
        # load the experiment settings
        settings = pd.read_pickle(os.path.join(model_dir, model_name, settings_name))

        # load the model
        model = torch.load(os.path.join(model_dir, model_name, name))
        model.eval()  

        # capture the model summary in variable
        f = io.StringIO()
        with redirect_stdout(f):
            summary(model, (1, settings['input_channels'], *output_shape), verbose=1, device=device)
        out = f.getvalue()

        # create a text file with summary
        with io.open(os.path.join(model_dir, model_name, f'{os.path.splitext(name)[0]}_summary.txt'), 'w', encoding="utf-8") as writer:
            writer.write(out)