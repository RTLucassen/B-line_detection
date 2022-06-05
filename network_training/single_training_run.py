"""
Train a neural network using specified settings and log the performance. 
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import pandas as pd

from utils.config import seed, models_folder
from network_training.training_run import training_run

# Specify the settings for the network training run
settings = {
    # general
    'ID'                 : '0001',      # experiment ID
    'name'               : 'test',      # experiment name
    'task'               : 'segmentation', # experiment task (options: 'segmentation', 'frame_classification', 'video_classification')
    'aux_classification' : False,       # add auxiliary loss (only applies when 'task' equals segmentation)
    # data   
    'label_folder'       : 'frames_1_disk_4mm',  # label folder name ('frames_1_disk_4mm' for segmentation, 'frames_1' for frame classification, and 'frames_16' for video classification)
    'aug_settings'       : {'scale'                 : (1.00, {'max_scale'      : 1.1        }), # multiplication factor
                            'rotation'              : (1.00, {'max_rotation'   : 15         }), # degrees
                            'shear'                 : (1.00, {'max_shear'      : 5          }), # degrees
                            'temporal_reflection'   : (0.00, {                              }),
                            'vertical_reflection'   : (0.50, {                              }),
                            'horizontal_translation': (1.00, {'max_translation': 0.10       }), # fraction of image width
                            'vertical_translation'  : (1.00, {'max_translation': 0.10       }), # fraction fo image height
                            'occlusion'             : (1.00, {'max_height'     : 0.25,  'max_width': 0.25}), # fraction of image height and width
                            'add_intensity'         : (1.00, {'intensity_range': (-0.2, 0.2)}), # intensity value added
                            'multiply_intensity'    : (1.00, {'max_factor'     : 1.2        }), # multiplication factor
                            'gamma_shift'           : (1.00, {'gamma_range'    : (0.6, 3.0) }), # gamma value
                            'gaussian_noise'        : (0.25, {'max_sigma'      : 0.05       }), # standard deviation of Gaussian
                            'gaussian_blur'         : (0.25, {'sigma_range'    : (0.0, 2.0) })}, # standard deviation of Gaussian
    # training
    'val_fold'           : '0',         # validation fold, the remaining folds are for training
    'epochs'             : 100,         # maximum number of epochs in training
    'batch_size'         : 32,          # number of items in batch
    'negatives'          : 16,          # number of negative items in batch
    'positives_fraction' : 1,           # fraction of positive items in epoch (allows for training on less data, but was not used)
    'use_early_stopping' : True,        # adds early stopping policy
    'es_patience'        : 10,          # early stopping threshold in epochs
    # network architectire
    'model'              : 'efficientnet-b0_unet', # network name
    'pretrained'         : False,       # determines whether pretrained or randomly initialized weights are used
    'input_channels'     : 1,           # number of input channels (generally 3 for pretrained because of rgb and 1 for randomly initialized is assumed)
    'N_classes'          : 2,           # number of output classes (generally 2)
    'init_method'        : 'kaiming_normal', # initalization method name (generally 'kaiming_normal')
    # loss function settings
    'loss_function'      : 'BCE_loss',  # name of loss function used (possibilities: 'dice_loss', 'focal_loss', 'BCE_loss', 'dice_focal_loss', 'roy_etal_loss')
    'aux_loss_function'  : 'BCE_loss',  # only applies if 'aux_classification' equals True
    'frequency_weighting': True,        # determines whether frequency weighting is used (generally True)
    'focal_gamma'        : 2,           # focal gamma weighting (only applies to focal loss or combinations including focal loss)
    'focal_class_weights': [1, 50],     # class weighting for foreground and background (only applies to focal loss or combinations including focal loss)
    'loss_weights'       : [2, 1],      # loss weighting (only applies to combinations of two losses)
    'aux_weight'         : 0,           # auxiliary loss weighting (only applies if 'aux_classification' equals True)
    # optimization settings
    'optimizer'          : 'adam',      # optimizer name
    'learning_rate'      : 0.001,       # learning rate value
    'lr_decay_factor'    : 0.5,         # learning rate decay factor
    'lr_decay_patience'  : 5,           # learning rate decay threshold in epochs
}

# update the seed such that it is specific for this experiment
settings['seed'] = int(settings['ID'])

# train a network using the specified settings above and log the model and training configuration
start_run, end_run, prev_best_loss = training_run(settings)

# add the training information to settings dictionary 
settings['start_run'] = start_run.strftime('%m/%d/%Y %H:%M:%S')
settings['end_run'] = end_run.strftime('%m/%d/%Y %H:%M:%S')
settings['best_val_loss'] = prev_best_loss

# create an experiment log file if it does not exist yet
overview_path = os.path.join(models_folder, 'experiment_log.csv')
if os.path.exists(overview_path):
    exp_dict = pd.read_csv(overview_path).to_dict('list')
else:
    exp_dict = {key:[] for key in list(settings.keys())}

# add the training settings to the overview
for key in settings.keys():
    exp_dict[key].append(settings[key])

pd.DataFrame.from_dict(exp_dict).to_csv(overview_path, index=False)
