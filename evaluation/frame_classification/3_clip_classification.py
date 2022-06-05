"""
Evaluates frame-level classification performance and optionally plots visualizations.
"""

import os
import sys
sys.path.append(os.path.join('..', '..'))
sys.path.append(os.path.join(__file__, '..', '..', '..'))

import time
import torch
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from sklearn.metrics import auc
from torch.utils.data import DataLoader, SequentialSampler

from utils.config import images_folder, models_folder, annotations_folder, info_folder
from utils.evaluation_utils import classify, get_clip_results, aggregate_predictions, isnumber
from utils.augmentation_utils import ClipDataset
from utils.training_utils import get_prediction


# mute warnings
warnings.filterwarnings("ignore")

# define model evaluation details
model_subfolder = 'frame_level'
model_names = ['0001_example_network_0'] 
dataset_split = 'val'
extension = ''

# define paths
clip_label_path = os.path.join(annotations_folder, 'B-line_expert_classification.csv')
dataset_split_path = os.path.join(info_folder, 'dataset_split_dictionary.pkl')

# define detection settings
classification_thresholds = 0.0   # for B-line presence in videos, was determined based on validation set result
aggregation_method = 'max'

# define saving settings
store_spreadsheet = False
store_visualizations = False
store_curve = True

# other parameters
batch_size = 32
auc_step_size = 0.0005

# --------------------------------------------------------------------------------------------------

# configure the number of workers and the device
num_workers = 0 if sys.platform == 'win32' else 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device\n')

# load the dataset split information
split_dict = pd.read_pickle(dataset_split_path)

# load the clip-level classification 
clip_labels = pd.read_csv(clip_label_path)

# define the directory to the model
model_dir = models_folder if model_subfolder == None else os.path.join(models_folder, model_subfolder)

# if equal to 'all', get all directories
if model_names == 'all':
    model_names = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
# if a single model name was given, add it to a list
elif isinstance(model_names, str):
    model_names = [model_names]

# convert threshold to list if it is not one yet
if not isinstance(classification_thresholds, list):
    classification_thresholds = [classification_thresholds]
    # if a single threshold was given, use it for all models
    if len(classification_thresholds) == 1:
        classification_thresholds *= len(model_names)
    # raise an error when the number of models and thresholds do not match
    elif len(model_names) != len(classification_thresholds):
        raise ValueError('Number of selected models does not match number of corresponding thresholds.')   

# loop over all models
for i, model_name in enumerate(model_names):  
    # first check how many models are available and get the corresponding information
    dirs = natsorted(os.listdir(os.path.join(model_dir, model_name)))
    settings_names = [d for d in dirs if d.startswith('experiment_settings') and d.endswith('.pkl')]
    names = [d for d in dirs if d.startswith('model') and d.endswith('.pth')]
    classification_threshold = classification_thresholds[i]

    # create a subfolder to store all results
    result_dir = os.path.join(model_dir, model_name, f'results_{classification_threshold:0.4f}_{aggregation_method}_{dataset_split}{extension}')
    if os.path.exists(os.path.join(result_dir)):
        raise IOError('Results directory already exists.')
    else:    
        os.mkdir(result_dir)

    # create a dictionary to store the results
    results_dict = {'case': [], 'clip': [], 'N_frames': [], 'predictions': [], 'agg_prediction': [], 
        'clip_prediction': [], 'label': [], 'result': [], 'total_time': [], 'avg_time': []}

    # check if the validation set should be used
    if dataset_split == 'val':
        split = model_name.split('_')[-1]
        if isnumber(split):
            cases = natsorted(split_dict[split])
            print(f'Evaluating {model_name} on the data split {split}')
        else:
            raise ValueError('Split for evaluation not recognised')
    else:
        cases = natsorted(split_dict[dataset_split]) 
        print(f'Evaluating {model_name} on the data split {dataset_split}')

    # loop over all cases and clips in the selected split 
    for case in cases:
        
        print(case)

        # get all clips for one case
        clips = natsorted(list(clip_labels[clip_labels['case'] == case]['clip']))
        
        for clip in tqdm(clips):
            # get the label and convert the clip number to a string
            label = int(clip_labels[(clip_labels['case'] == case) & (clip_labels['clip'] == clip)]['label'])
            clip = str(clip).zfill(3)

            # initialize a variable to store the model predictions
            y_pred_all = None

            # start timing
            start = time.perf_counter()

            # loop over the models
            for name, settings_name in zip(names, settings_names):

                # load the experiment settings and store some settings in variables for later use
                settings = pd.read_pickle(os.path.join(model_dir, model_name, settings_name))
                frames = int(settings['label_folder'].split('_')[1])
                overlap_fraction = 0
                apply_padding = False
                video_model = False

                # get all paths
                directory = os.path.join(images_folder, f'processed_frames', case)
                paths = os.listdir(directory)
        
                # get all paths to images that belong to the current clip
                image_paths = [path for path in paths if path.split('_')[2] == clip]

                # correct for the difference in image size expected by ViT using padding
                padding = (0, 0, 64, 64) if settings['model'] == 'ViT' else (0, 0, 0, 0)

                # create the dataset and dataloader object
                dataset = ClipDataset(image_paths, directory, frames, overlap_fraction, apply_padding, settings['pretrained'], video_model, padding)
                dataloader = DataLoader(dataset, batch_size, sampler=SequentialSampler(dataset), shuffle=False, pin_memory=True)

                # load the model
                model = torch.load(os.path.join(model_dir, model_name, name))
                model.eval()

                first = 0
                # loop over batches
                with torch.no_grad():
                    for X in dataloader:
                        # bring the data to the correct device
                        X = X.to(device)
                        
                        # get the model prediction
                        y_pred = get_prediction(model(X))
                        if y_pred.shape[0] == 2*X.shape[0]:
                            y_pred = torch.split(y_pred, split_size_or_sections=y_pred.shape[0] // 2)[0]
                        y_pred = torch.softmax(y_pred, dim=1).to('cpu')

                        # replace None by an empty tensor
                        if y_pred_all == None:
                            y_pred_all = torch.zeros((dataset.__len__(), *y_pred.shape[1:]))

                        # add the predictions to the storage variable
                        last = first + X.shape[0]
                        y_pred_all[first:last, ...] += y_pred
                        first = last

            # obtain the average by dividing the summed model predictions by the number of models
            y_pred_all /= len(names)

            # -------------------------  EVALUATION  -------------------------

            # select the foreground predictions, take the average, and compare it to the classification threshold
            N_frames = y_pred_all.shape[0]
            predictions = y_pred_all[:, 1]
            agg_prediction = aggregate_predictions(predictions, aggregation_method=aggregation_method)
            clip_prediction = 1 if agg_prediction >= classification_threshold else 0
            # get the evaluation time
            total_time = time.perf_counter()-start
            avg_time = total_time/N_frames

            # add the clip results to the results dictionary                
            results_dict['clip'].append(clip)
            results_dict['case'].append(case)
            results_dict['N_frames'].append(N_frames) 
            results_dict['predictions'].append(predictions.tolist())
            results_dict['agg_prediction'].append(agg_prediction) 
            results_dict['clip_prediction'].append(clip_prediction)
            results_dict['label'].append(label)
            results_dict['result'].append(classify(clip_prediction, label)) 
            results_dict['total_time'].append(total_time)
            results_dict['avg_time'].append(avg_time) 

    # get the dataframes with the detection results
    summary_df, clip_results_df = get_clip_results(results_dict)

    # create an Excel file with the detection results
    if store_spreadsheet:
        with pd.ExcelWriter(os.path.join(result_dir, 'individual_clip_results.xlsx')) as writer:
            summary_df.to_excel(writer, sheet_name='summary_results', index=False)
            clip_results_df.to_excel(writer, sheet_name='clip_results', index=False)
    
    if store_visualizations:
        # create an output directory if it does not exist yet
        output_dir = os.path.join(result_dir, f'individual_clip_probabilities')
        if not os.path.exists(os.path.join(output_dir)):
            os.mkdir(output_dir)

        # loop over all clips in the dataframe
        for index, row in clip_results_df.iterrows():
            # get the variables for figure title
            case, clip, label = row['case'], row['clip'], row['label']
            # plot the predicted probability over the frames
            plt.plot([1, int(row['N_frames'])], [float(row['agg_prediction'])]*2, color='grey', linestyle='dashed')
            plt.plot(np.arange(1, row['N_frames']+1), row['predictions'], color='black')
            plt.ylim([0, 1])
            plt.title(f'{case}, Clip: {clip}, Label: {label}')
            plt.xlabel('Frame')
            plt.ylabel('predicted probability')
            
            # save the figure and close it
            plt.savefig(os.path.join(output_dir, f'{case}_clip_{clip}_probability.png'))
            plt.close()
            
    if store_curve:
        # create an output directory if it does not exist yet
        output_dir = os.path.join(result_dir, f'clip_classification_curves')
        if not os.path.exists(os.path.join(output_dir)):
            os.mkdir(output_dir)

        # define a variable to keep track of results for different settings of the classification threshold
        track_summary_results = None

        # copy the results dictionary
        results_roc_dict = results_dict.copy()

        threshold_values = list(np.arange(0, 1+auc_step_size, auc_step_size))
        for threshold in tqdm(threshold_values):
            # get the predictions and results based on different classification threshold values
            results_roc_dict['prediction'] = [1 if avg_detections >= threshold else 0 for avg_detections in results_roc_dict['agg_prediction']]
            results_roc_dict['result'] = [classify(pred, lab) for pred, lab in zip(results_roc_dict['prediction'], results_roc_dict['label'])]
            summary_df, _ = get_clip_results(results_roc_dict)
            summary_df['threshold'] = threshold
            summary_df['abs_diff_recall_specificity'] = abs(summary_df['recall']-summary_df['specificity'])

            # append the results
            if track_summary_results == None:
                track_summary_results = summary_df.to_dict('list')
            else:
                summary_dict = summary_df.to_dict('list')
                for key in track_summary_results.keys():
                    track_summary_results[key].append(summary_dict[key][0])

        # calculate the AUC of the ROC curve
        roc_auc_value = auc([1-value for value in track_summary_results['specificity']], track_summary_results['recall'])

        # store summary results as a pickle file and as an Excel file
        file = open(os.path.join(output_dir, f'classification_threshold_results.pkl'), 'wb')
        pickle.dump(track_summary_results, file)
        file.close()

        summary_results_df = pd.DataFrame.from_dict(track_summary_results)
        summary_results_df.to_excel(os.path.join(output_dir, f'classification_threshold_results.xlsx'))

        # find the threshold value for which the sensitivity and specificity are (almost) equal,
        # which is equal to the threshold for which the precision and recall match
        equal_row = None
        for index, row in summary_results_df.iterrows():
            if index == 0:
                equal_row = row
            elif row ['abs_diff_recall_specificity'] < equal_row['abs_diff_recall_specificity']:
                equal_row = row
        
        # write a summary file with the results
        equal_threshold = equal_row['threshold']
        equal_recall = equal_row['recall']
        equal_specificity = equal_row['specificity']
        with open(os.path.join(output_dir, f'summary.txt'), 'w') as f:
            f.write(f'{equal_threshold}\t{equal_recall}\t{equal_specificity}\t{roc_auc_value}')

        # create a figure with the detection performance against one of the detection settings
        fig, ax = plt.subplots(2, 2, figsize=(10, 9))
        ax[0,0].plot(threshold_values, track_summary_results['total TP'], label='total TP', color='limegreen')
        ax[0,0].plot(threshold_values, track_summary_results['total TN'], label='total TN', color='lightgreen')
        ax[0,0].plot(threshold_values, track_summary_results['total FP'], label='total FP', color='red')
        ax[0,0].plot(threshold_values, track_summary_results['total FN'], label='total FN', color='salmon')
        ax[0,0].set_xlabel('classification threshold')
        ax[0,0].set_ylabel('count')
        ax[0,0].set_ylim(bottom=0)
        ax[0,0].set_xticks(np.arange(0, max(threshold_values)+0.01, 0.5))
        ax[0,0].legend()

        ax[0,1].plot(threshold_values, track_summary_results['precision'], label='precision', color='lightskyblue')
        ax[0,1].plot(threshold_values, track_summary_results['recall'], label='recall', color='royalblue')
        ax[0,1].plot(threshold_values, track_summary_results['f1-score'], label='f1-score', color='midnightblue')
        ax[0,1].plot(threshold_values, track_summary_results['specificity'], label='specificity', color='salmon')
        ax[0,1].plot(threshold_values, track_summary_results['accuracy'], label='accuracy', color='limegreen')
        ax[0,1].set_xlabel('classification threshold')
        ax[0,1].set_ylabel('score')
        ax[0,1].set_xticks(np.arange(0, max(threshold_values)+0.01, 0.5))
        ax[0,1].set_yticks(np.arange(0, 1.01, 0.2))
        ax[0,1].legend()

        ax[1,0].plot(track_summary_results['recall'], track_summary_results['precision'], color='royalblue')
        ax[1,0].set_xlabel('Recall')
        ax[1,0].set_ylabel('Precision')
        ax[1,0].set_xticks(np.arange(0, 1.01, 0.2))
        ax[1,0].set_yticks(np.arange(0, 1.01, 0.2))

        ax[1,1].plot([1-value for value in track_summary_results['specificity']], track_summary_results['recall'], color='royalblue')
        ax[1,1].set_xlabel('False positive rate')
        ax[1,1].set_ylabel('True positive rate')
        ax[1,1].set_xticks(np.arange(0, 1.01, 0.2))
        ax[1,1].set_yticks(np.arange(0, 1.01, 0.2))
        roc_auc_value = auc([1-value for value in track_summary_results['specificity']], track_summary_results['recall'])
        ax[1,1].set_title(f'AUC: {roc_auc_value:0.4f}')

        fig.suptitle(f'Classification threshold: {min(threshold_values):0.2f}-{max(threshold_values):0.2f}')
        plt.savefig(os.path.join(output_dir, f'classification_threshold_results.png'))
        plt.close()


