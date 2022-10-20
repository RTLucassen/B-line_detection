"""
Train a neural network using specified settings and log the performance. 
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import numpy as np
import pandas as pd

from utils.config import models_folder
from network_training.training_run import training_run


# load the experiment schedule
schedule_path = os.path.join(models_folder, 'example_training_runs.xlsx')
if os.path.exists(schedule_path):
    schedule_df = pd.read_excel(schedule_path)
    schedule_dict = schedule_df.to_dict('list')
else:
    raise IOError('Incorrect path to schedule file was specified.')

# loop over all rows in the schedule
for idx in np.arange(len(schedule_df)):
    # retrieve the settings  
    settings = schedule_df.iloc[idx].to_dict()

    # check if this experiment has not been performed, if so, start the experiment
    if settings['finished'] == False:

        del settings['finished']

        # convert the extracted information to correct the datatype
        settings['ID'] = str(int(settings['ID'])).zfill(4)
        settings['name'] = str(settings['name'])
        settings['task'] = str(settings['task'])
        settings['aux_classification'] = bool(settings['aux_classification'])
        settings['label_folder'] = str(settings['label_folder'])
        settings['aug_settings'] = eval(settings['aug_settings'])
        settings['seed'] = int(settings['seed'])
        settings['val_fold'] = str(int(settings['val_fold']))
        settings['epochs'] = int(settings['epochs'])
        settings['batch_size'] = int(settings['batch_size'])
        settings['negatives'] = int(settings['negatives'])
        settings['positives_fraction'] = float(settings['positives_fraction'])
        settings['use_early_stopping'] = bool(settings['use_early_stopping'])
        settings['es_patience'] = int(settings['es_patience'])
        settings['model'] = str(settings['model'])
        settings['pretrained'] = bool(settings['pretrained'])
        settings['input_channels'] = int(settings['input_channels'])
        settings['N_classes'] = int(settings['N_classes'])
        settings['init_method'] = str(settings['init_method'])
        settings['loss_function'] = str(settings['loss_function'])
        settings['aux_loss_function'] = str(settings['aux_loss_function'])
        settings['focal_gamma'] = float(settings['focal_gamma'])
        settings['focal_class_weights'] = eval(settings['focal_class_weights'])
        settings['loss_weights'] = eval(settings['loss_weights'])
        settings['aux_weight'] = float(settings['aux_weight'])
        settings['optimizer'] = str(settings['optimizer'])
        settings['learning_rate'] = float(settings['learning_rate'])
        settings['lr_decay_patience'] = int(settings['lr_decay_patience'])
        settings['lr_decay_factor'] = float(settings['lr_decay_factor'])
        
        # train a network using the specified settings above and log the model and training configuration
        start_run, end_run, prev_best_loss = training_run(settings)

        # add the training information to the settings dictionary 
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

        # also update the schedule
        schedule_dict['finished'][idx] = True
        schedule_dict['start_run'][idx] = start_run.strftime('%m/%d/%Y %H:%M:%S')
        schedule_dict['end_run'][idx] = end_run.strftime('%m/%d/%Y %H:%M:%S')
        schedule_dict['best_val_loss'][idx] = prev_best_loss

        pd.DataFrame.from_dict(schedule_dict).to_excel(schedule_path, index=False)