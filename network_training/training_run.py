"""
Implementation of entire network training run and logging. 
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import time
import datetime
import random
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SequentialSampler

from models.segmentation import get_segmentation_model
from models.frame_classification import get_frame_classification_model
from models.video_classification import get_video_classification_model
from utils.config import seed, images_folder, models_folder
from utils.augmentation_utils import ImageDataset
from utils.training_utils import stop_early, get_lr, seed_worker, get_prediction, get_loss_function
from utils.visualization_utils import image_viewer


def training_run(settings: dict) -> None:
    """
    Args:
        settings:  dictionary with the entire training configuration.
    """

    print('\nStart preparation for network training\n')

    # start timing
    start_run = datetime.datetime.now()

    # define the path to the dataset information
    dataset_info_path = os.path.join(images_folder, 'labelsets', settings['label_folder'], 'image_label_combinations.csv')

    # configure the number of workers and the device
    num_workers = 0 if sys.platform == 'win32' else 4 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device\n')

    # seed the randomness
    exp_seed = settings['seed']
    torch.backends.cudnn.benchmark = True # False causes cuDNN to deterministically select an algorithm.
    torch.manual_seed(exp_seed)
    np.random.seed(exp_seed)
    random.seed(exp_seed)
    print(f'Using seed: {exp_seed}\n')

    # create a directory to store training settings, logs, and the best network parameters 
    output_folder = os.path.join(models_folder, '_'.join([settings[key] for key in ['ID', 'name', 'val_fold']]))
    if os.path.exists(output_folder):
        raise IOError('Specified output directory already exists.')
    else:
        os.mkdir(output_folder) 

    # load the dataset information
    df = pd.read_csv(dataset_info_path)
    df['split'] = df['split'].apply(str)
    # get the cross-validation fold names and remove the test set information
    splits = sorted(list(set(df['split'].to_list())))
    val_fold = settings['val_fold']
    train_folds = [fold for fold in splits if fold not in ['test', val_fold]]

    print(f'Training folds: ', *train_folds)
    print(f'Validation fold: {val_fold}\n')

    # group the information for all training folds in the same dataframe
    # also create a dataframe for the validation fold information
    train_df = df[df.split.isin(train_folds)]
    val_df = df[df.split == val_fold]
    train_negatives = val_negatives = settings['negatives']

    # create a single batch with multiples of the same example for augmentation debugging purposes
    if False: train_df = pd.concat([train_df.iloc[0:1]]*settings['batch_size'], ignore_index=True)

    # select the correct task-specific settings 
    # (i.e. the dataset class, the number of negatives per batch, and the model loader function)
    if settings['task'] == 'segmentation':
        get_model = get_segmentation_model
        
    elif settings['task'] == 'frame_classification':
        get_model = get_frame_classification_model
        # check if the batch is half-filled with negative examples and if auxiliary classification is off
        if settings['negatives'] != settings['batch_size']//2:
            raise ValueError('For classification, the number of negatives must be half of the images in the batch.')        
        if settings['aux_classification'] == True:
            raise ValueError('Auxiliary classification only applies for segmentation models.')
    
    elif settings['task'] == 'video_classification':
        get_model = get_video_classification_model
        # check if the batch is half-filled with negative examples and if auxiliary classification is off
        if settings['negatives'] != settings['batch_size']//2:
            raise ValueError('For classification, the number of negatives must be half of the images in the batch.')
        if settings['aux_classification'] == True:
            raise ValueError('Auxiliary classification only applies for segmentation models.')

    else:
        raise ValueError('Specified task not recognized')

    # correct for the difference in image size expected by ViT using padding
    padding = (0, 0, 64, 64) if settings['model'] == 'ViT' else (0, 0, 0, 0)

    # create the dataset instances
    train_dataset = ImageDataset(
        task=settings['task'],
        df=train_df, 
        negatives=train_negatives, 
        batch_size=settings['batch_size'], 
        seed=exp_seed, 
        folder=images_folder, 
        aug_settings_dict=settings['aug_settings'],
        positives_faction=settings['positives_fraction'],
        padding=padding,
        pretrained=settings['pretrained'],
    )
    val_dataset = ImageDataset(
        task=settings['task'],
        df=val_df, 
        negatives=val_negatives, 
        batch_size=settings['batch_size'], 
        seed=seed, 
        folder=images_folder,
        positives_faction=settings['positives_fraction'],
        padding=padding,
        pretrained=settings['pretrained']
    ) 

    # create the dataset loaders (shuffling is instead performed by the custom dataset class)                                                                                 
    train_dataloader = DataLoader(
        train_dataset, 
        num_workers=num_workers,
        pin_memory=True,
        batch_size=settings['batch_size'], 
        sampler=SequentialSampler(train_dataset), 
        worker_init_fn=seed_worker, 
        shuffle=False  
    )
    val_dataloader = DataLoader(
        val_dataset, 
        num_workers=num_workers,
        pin_memory=True, 
        batch_size=settings['batch_size'], 
        sampler=SequentialSampler(val_dataset),   
        worker_init_fn=seed_worker, 
        shuffle=False 
    )  

    # create an instance of the network and move it to the selected device
    net = get_model(
        settings['model'], 
        settings['input_channels'], 
        settings['N_classes'], 
        settings['pretrained'], 
        settings['init_method'],
        settings['aux_classification']
    )
    # check if multiple GPUs can be used
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(net)
        print(f'Using {torch.cuda.device_count()} GPUs')
    else:
        model = net
    
    # bring the model to selected device and print a summary of the architecture
    model = model.to(device)
    summary(model, depth=4, col_names=['num_params'])

    # check what class weights should be used
    class_weights = train_dataset.get_class_weights() if settings['focal_class_weights'] == ['balanced'] else settings['focal_class_weights']
    class_weights = torch.tensor(class_weights).type(torch.FloatTensor).to(device)
    # define the loss function parameters
    loss_param = {
        'frequency_weighting': settings['frequency_weighting'],
        'gamma': settings['focal_gamma'],
        'class_weights': class_weights,
        'dice_focal_weights': settings['loss_weights']
    }
    # define the loss function(s)
    loss_function = get_loss_function(settings['loss_function'], loss_param)
    aux_loss_function = get_loss_function(settings['aux_loss_function'], loss_param) if settings['aux_classification'] else None

    # define the optimization algorithm
    if settings['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=settings['learning_rate'])
    else:
        raise ValueError('Optimizer setting was not recognized.')

    # define the learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        factor=settings['lr_decay_factor'], 
        patience=max(0, settings['lr_decay_patience']-1),
        verbose=True
    )

    # --------------------------------------------------------------------------------

    print('Start network training\n')
    
    # define variables to log the training statistics and a variable to keep track 
    # of the epoch with the best validation loss
    log = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': [], 'time': []}
    epoch_best_val_loss = 0

    for epoch in np.arange(settings['epochs']):
                
        # start timing the epoch
        start_epoch = time.time()

        # ---------------- TRAINING -------------------
        train_loss = 0

        # X = image data, y = labeled data, w = frequency weighting, p = path to image
        for X, y, w, p in tqdm(train_dataloader):           

            # add the paths to record to keep track of all sampled images
            train_dataset.add_to_record(p)

            # for debugging purposes
            if False: print(X.shape); print(y); print(w); print(p)
            if False: image_viewer(X) 
            if False and settings['task'] == 'segmentation': image_viewer(y)
            
            # bring the data to the correct device
            X, y, w = X.to(device), y.to(device), w.to(device)
            # check if an auxiliary loss is used
            if settings['aux_classification']: 
                # get the classification labels from the segmentation labels
                y_aux = torch.amax(y, dim=list(np.arange(2, len(y.shape))))[:, 1:2]
                y_aux = torch.cat([1-y_aux, y_aux], dim=1)
                # get the prediction, compute the separate losses, and aggregate them
                pred, aux_pred = get_prediction(model(X))
                main_loss = loss_function(pred, y, w)
                aux_loss = aux_loss_function(aux_pred, y_aux, w)
                loss = main_loss + settings['aux_weight']*aux_loss
            else:
                # get the prediction and the loss
                pred = get_prediction(model(X))
                loss = loss_function(pred, y, w)

            # update the trainable parameters using backpropagation
            optimizer.zero_grad() # set the gradient to 0 again
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()*settings['batch_size']

        # add the average training loss value to the log
        train_reference = sum(train_dataset.frame_weights) if settings['frequency_weighting'] else train_dataset.__len__()
        train_loss /= train_reference

        # set the model in evaluation mode (changes behaviour of batch normalization or dropout layers)
        model.eval()

        # shuffle training dataset for next iteration
        train_dataset.shuffle()

        # --------------- VALIDATION ------------------
        
        # deactivate autograd engine (backpropagation not required here)
        val_loss = 0
        with torch.no_grad(): 
            # X = image data, y = labeled data, w = frequency weighting, p = path to image
            for X, y, w, p in val_dataloader:

                # add the paths to record to keep track of all sampled images
                val_dataset.add_to_record(p)

                # bring the data to the correct device
                X, y, w = X.to(device), y.to(device), w.to(device)

                # check if an auxiliary loss is used
                if settings['aux_classification']: 
                    # get the classification labels from the segmentation labels
                    y_aux = torch.amax(y, dim=list(np.arange(2, len(y.shape))))[:, 1:2]
                    y_aux = torch.cat([1-y_aux, y_aux], dim=1)
                    # get the prediction, compute the separate losses, and aggregate them
                    pred, aux_pred = get_prediction(model(X))
                    main_loss = loss_function(pred, y, w)
                    aux_loss = aux_loss_function(aux_pred, y_aux, w)
                    loss = main_loss + settings['aux_weight']*aux_loss
                else:
                    # get the prediction and the loss
                    pred = get_prediction(model(X))
                    loss = loss_function(pred, y, w)
                
                val_loss += loss.item()*settings['batch_size']

        # add the average validation loss value to the log
        val_reference = sum(val_dataset.frame_weights) if settings['frequency_weighting'] else val_dataset.__len__()
        val_loss = None if val_reference == 0 else val_loss/val_reference

        # end timing
        time_diff = time.time() - start_epoch

        # notify the user about the training and validation loss
        print(f'\nEpoch {str(epoch).zfill(4)} (time: {time_diff:.1f}s)')
        print(f'Training   | loss: {train_loss:.4f}')
        if len(val_df) != 0: print(f'Validation | loss: {val_loss:.4f}\n')
    
        # log the epoch information
        log['epoch'].append(epoch)
        log['train_loss'].append(train_loss)
        log['val_loss'].append(val_loss)
        log['lr'].append(get_lr(optimizer))
        log['time'].append(time_diff)

        # set the model to training mode (changes behaviour of batch normalization or dropout layers)
        model.train()

        if len(val_df) != 0:
            # after the first epoch, start to check whether 
            # the model performance has improved on the validation set. 
            if len(log['val_loss']) > 1:

                prev_best_loss = log['val_loss'][epoch_best_val_loss]
                last_loss = log['val_loss'][epoch]

                if last_loss < prev_best_loss:
                    print(f'The validation loss improved from {prev_best_loss:.4f} to {last_loss:.4f}. Model parameters are saved.\n')
                    epoch_best_val_loss = epoch
                    prev_best_loss = log['val_loss'][epoch_best_val_loss]

                    # save the model with improved weights
                    torch.save(net, os.path.join(output_folder, 'model.pth'))
            
                else:
                    print(f'The validation loss did not improve from {prev_best_loss:.4f}.\n')
            else:
                prev_best_loss = log['val_loss'][epoch_best_val_loss]
                # save the model weights for the first run
                torch.save(net, os.path.join(output_folder, 'model.pth'))
            
            # check if the learning rate should decay based on past validation results
            scheduler.step(val_loss)   

            # early stopping rule
            if settings['use_early_stopping']:
                if stop_early(log['val_loss'], settings['es_patience']):
                    es_patience = settings['es_patience']
                    print(f'Training was stopped early because validation loss did not improve in the last {es_patience} epochs.')
                    break
        else:
            prev_best_loss = None

    # save the final model after the last epoch or just before stopping early
    torch.save(net, os.path.join(output_folder, 'final_model.pth'))
    
    # save a record of all training and validation examples used (to check that there is no overlap)
    train_dataset.get_record(output_folder, 'training_record.csv')
    if len(val_df) != 0: val_dataset.get_record(output_folder, 'validation_record.csv')

    # --------------------------------------------------------------------------------

    print('Logging training information')

    # end time run
    end_run = datetime.datetime.now()

    # save the logged epoch information
    pd.DataFrame.from_dict(log).to_csv(os.path.join(output_folder, 'log.csv'), index=False)  
    
    # create a plot with the training and validation loss
    plt.figure()
    plt.plot(np.arange(len(log['train_loss'])), log['train_loss'], color='blue', label='train')
    plt.plot(np.arange(len(log['val_loss'])), log['val_loss'], color='black', label='val')
    plt.xlim([0, len(log['train_loss'])-1])
    plt.ylim([0, 1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'loss_plot.png'))
    plt.close()  

    # save the model and training configurations as a dictionary variable in the corresponding model folder
    file = open(os.path.join(output_folder, 'experiment_settings.pkl'), 'wb')
    pickle.dump(settings, file)
    file.close()
    
    print('Training finished')

    return start_run, end_run, prev_best_loss