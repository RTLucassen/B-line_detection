"""
Evaluates instance (B-line origin segmentation) detection performance 
and optionally plots visualizations for (ensembles of) segmentation models.
"""

import os
import sys
sys.path.append(os.path.join('..', '..'))
sys.path.append(os.path.join(__file__, '..', '..', '..'))

import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from skimage.morphology import disk
from torch.utils.data import DataLoader, SequentialSampler

from utils.augmentation_utils import ClipDataset
from utils.config import images_folder, models_folder, annotations_folder, info_folder
from utils.training_utils import get_prediction
from utils.evaluation_utils import separate_structures, detect_instances, get_detection_results, isnumber
from utils.colormap_utils import truncated_cmap


# define model evaluation details
model_subfolder = 'pixel_level'
model_names = ['0001_example_network_0']
dataset_split = 'val'
extension = ''

# define paths
top_point_annotations_path = os.path.join(annotations_folder, 'top_point_annotations.csv')
clip_label_path = os.path.join(annotations_folder, 'B-line_expert_classification.csv')
dataset_split_path = os.path.join(info_folder, 'dataset_split_dictionary.pkl')
scale_path = os.path.join(info_folder, 'physical_scale_dictionary.pkl')
processing_path = os.path.join(info_folder, 'processing_dictionary.pkl')

# define the detection settings
prob_threshold = 0.50
min_relative_pixel_area = 0.00  
prob_weighted_centroids = True  
max_physical_distance = 0.50      # cm

# define saving settings
store_visualizations = False
store_spreadsheet = True

# other parameters
batch_size = 16
min_annotations = 1

# --------------------------------------------------------------------------------------------------

# configure the number of workers and the device
num_workers = 0 if sys.platform == 'win32' else 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device\n')

# load the scale and processing information
split_dict = pd.read_pickle(dataset_split_path)
scale_dict = pd.read_pickle(scale_path)
processing_dict = pd.read_pickle(processing_path)

# load the top point annotations dataframe
top_point_df = pd.read_csv(top_point_annotations_path)

# define the directory to the model
model_dir = models_folder if model_subfolder == None else os.path.join(models_folder, model_subfolder)  

# if equal to 'all', get all directories
if model_names == 'all':
    model_names = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
# if a single model name was given, add it to a list
elif isinstance(model_names, str):
    model_names = [model_names]

# loop over all models
for model_name in natsorted(list(set(model_names))):  
    # first check how many models are available
    dirs = natsorted(os.listdir(os.path.join(model_dir, model_name)))
    settings_names = [d for d in dirs if d.startswith('experiment_settings') and d.endswith('.pkl')]
    if 'ensemble.pth' in dirs:
        names = ['ensemble.pth']
        settings_names = [settings_names[0]]
    else:
        names = [d for d in dirs if d.startswith('model') and d.endswith('.pth')]

    # if the validation set is used, select the correct validation fold
    if dataset_split == 'val':
        split = model_name.split('_')[-1]
        if isnumber(split):
            print(f'Evaluating {model_name} on the data split {dataset_split}')
        else:
            print('Warning: split for evaluation not recognised')
            continue
    # else use the specified fold
    else:
        split = dataset_split
        print(f'Evaluating {model_name} on the data split {dataset_split}')

    # create a subfolder to store detection results
    result_dir = os.path.join(model_dir, model_name, f'individual_predictions_{prob_threshold:0.2f}_{min_relative_pixel_area:0.2f}_{max_physical_distance:0.2f}_{min_annotations}_{dataset_split}{extension}')
    if os.path.exists(os.path.join(result_dir)):
        raise IOError('Results directory already exists.')
    else:    
        os.mkdir(result_dir)

    # initialize a variable to store the model predictions
    y_pred_all = None

    # loop over the models
    for name, settings_name in zip(names, settings_names):

        print(f'Model: {name}')

        # load the experiment settings
        settings = pd.read_pickle(os.path.join(model_dir, model_name, settings_name))

        # check if the model is a segmentation model
        if settings['task'] != 'segmentation':
            raise ValueError('Instance detection evaluation requires a segmentation model.')

        # load the dataset information
        clip_labels = pd.read_csv(clip_label_path)
        selected_clips = clip_labels[clip_labels['case'].isin(split_dict[split])]
        positive_clips = selected_clips[selected_clips['label'] == 1]
        
        image_paths = []
        for index, row in positive_clips.iterrows():
            case = row['case']
            clip = row['clip']
            filenames = os.listdir(os.path.join(images_folder, 'processed_frames', case))

            # select all names from a clip that were reviewed by the expert (i.e., only every 4th)
            selected_names = [name for name in filenames if (int(name.split('_')[2]) == clip) and (int(os.path.splitext(name)[0].split('_')[3])%4 == 0)]
            selected_paths = [os.path.join('processed_frames', case, name) for name in selected_names]
            # count the number of annotations
            count = 0
            for i, r in top_point_df[(top_point_df['case'] == case) & (top_point_df['clip'] == clip)].iterrows():
                count += len(eval(r['top_points']))

            if count >= min_annotations:
                image_paths += selected_paths
        
        # create the dataset and dataloader object
        dataset = ClipDataset(image_paths, images_folder, 1, 0, False, settings['pretrained'], False)
        dataloader = DataLoader(dataset, batch_size, sampler=SequentialSampler(dataset), shuffle=False, pin_memory=True)

        # load the model
        model = torch.load(os.path.join(model_dir, model_name, name))
        model.eval()

        first = 0
        # loop over batches
        with torch.no_grad():
            for X in tqdm(dataloader):
                # bring the data to correct device
                X = X.to(device)
                # get the prediction
                y_pred = get_prediction(model(X))
                y_pred = y_pred[0] if settings['aux_classification'] else y_pred
                y_pred = torch.softmax(y_pred, dim=1).to('cpu') if name != 'ensemble.pth' else y_pred.to('cpu')

                # replace None by empty tensor
                if y_pred_all == None:
                    y_pred_all = torch.zeros((len(dataset.images), *y_pred.shape[1:]))

                # add the predictions to the storage variable
                last = first + X.shape[0]
                y_pred_all[first:last, ...] += y_pred
                first = last

    # obtain the average by dividing the summed model predictions by the number of models
    y_pred_all /= len(names)

    # -------------------------  EVALUATION  -------------------------

    # create a dictionary to store detection performance for each 
    results_dict = {'image': [], 'case': [], 'clip': [], 'frame': [], 'dict': [], 'TP': [], 'FP': [], 'FN': []}

    # loop over all frame predictions
    for idx, path in tqdm(enumerate(dataset.images)):
        # get the case, clip, and frame information
        path_splitted = path.replace('\\', '/').split('/')
        name = os.path.splitext(path_splitted[-1])[0]
        _, case, clip, frame = name.split('_')
        case_name = 'Case-'+case

        # get the maximum pixel detection distance based on the physical scale of the clip
        scale = scale_dict[case_name][clip][0]
        processing_info = processing_dict[case_name][clip]
        max_pixel_distance = max_physical_distance * scale * processing_info[2]
       
        # create a structuring element
        radius = round(float(settings['label_folder'].split('_')[3][:-2]) / 10 / 2 * scale * processing_info[2])
        min_pixel_area = np.sum(disk(radius)) * min_relative_pixel_area

        # select the frame foreground prediction, threshold the prediction, and separate the structures
        prediction = y_pred_all[idx, 1, ...]
        binary_prediction = torch.where(prediction >= prob_threshold, 1, 0)
        structures = separate_structures(binary_prediction, min_pixel_area)
        # optionally multiply the structures with the predicted probabilities
        if prob_weighted_centroids:
            structures = structures * np.tile(prediction[..., None], (1, 1, structures.shape[-1]))

        # get the top points of the annotations for the specific clip
        df = top_point_df
        selection = df[(df['case'] == case_name) & (df['clip'] == int(clip)) & (df['frame'] == int(frame))]['top_points']
        if len(selection) != 0:
            # get the top points
            top_points = eval(selection.values[0])
        else:
            top_points = []

        # get the detection results and calculate the detection statistics
        detection_dict = detect_instances(structures, top_points, max_pixel_distance)
        TP, FP, FN = len(set([p[1] for p in detection_dict['TP']])), len(detection_dict['FP']), len(detection_dict['FN'])
    
        # add the frame level detection results to the results dictionary 
        results_dict['image'].append(path)
        results_dict['case'].append(case)
        results_dict['clip'].append(clip)
        results_dict['frame'].append(frame)
        results_dict['dict'].append(detection_dict)
        results_dict['TP'].append(TP)
        results_dict['FP'].append(FP)
        results_dict['FN'].append(FN)

        # create visualizations and store them if enabled
        if store_visualizations:
            # create the output directory if it does not exist yet
            output_dir = os.path.join(result_dir, f'detection_results_{prob_threshold:0.2f}_{min_relative_pixel_area:0.2f}_{max_physical_distance:0.2f}_{min_annotations}')
            if not os.path.exists(os.path.join(output_dir)):
                os.mkdir(output_dir)

            # create the truncated turbo colormap
            truncated_turbo = truncated_cmap('turbo', 0.1, 0.9, 100)

            # load the image
            image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(images_folder, path.replace('\\', '/'))))

            # create the figure
            fig, ax = plt.subplots(1,2, figsize = (8, 4))
            for x in ax: x.set_xticks([]), x.set_yticks([])            
            ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)
            ax[0].imshow(np.ma.masked_where(prediction < 0.01, prediction), cmap=truncated_turbo, vmin=0, vmax=1)
            ax[0].set_xlabel('Frame with prediction')
            
            TP_circles = [plt.Circle(point[0], max_pixel_distance, color='green', fill=False) for point in detection_dict['TP']]
            FP_circles = [plt.Circle(point, max_pixel_distance, color='red', fill=False) for point in detection_dict['FP']]

            ax[1].imshow(np.ones_like(image)*100 + image * 0.40, cmap='gray', vmin=0, vmax=255)
            ax[1].scatter([p[1][0] for p in detection_dict['TP']], [p[1][1] for p in detection_dict['TP']], color='lightgreen', label='TP (GT)', marker='.')
            ax[1].scatter([p[0][0] for p in detection_dict['TP']], [p[0][1] for p in detection_dict['TP']], color='green', label='TP (pred)', marker='.')    
            ax[1].scatter([p[0] for p in detection_dict['FP']], [p[1] for p in detection_dict['FP']], color='red', label='FP', marker='.')
            ax[1].scatter([p[0] for p in detection_dict['FN']], [p[1] for p in detection_dict['FN']], color='white', label='FN', marker='.')
            for patch in TP_circles+FP_circles:
                ax[1].add_patch(patch)
            ax[1].legend(fontsize = 'x-small')
            ax[1].set_title(f'TP: {TP}, FP: {FP}, FN: {FN}')
            ax[1].set_xlabel('Detection result')
            
            plt.tight_layout()
            plt.suptitle(name+f'\nProb. threshold: {prob_threshold}, Minimum relative area: {min_relative_pixel_area*100}%, Detection dist. {max_physical_distance} cm')
            plt.savefig(os.path.join(output_dir, f'{name}.png'))
            plt.close()

    # create an Excel file with detection results
    if store_spreadsheet:
        # get the dataframes with detection results
        summary_df, clip_results_df, frame_results_df = get_detection_results(results_dict)

        with pd.ExcelWriter(os.path.join(result_dir, f'detection_results_{prob_threshold:0.2f}_{min_relative_pixel_area:0.2f}_{max_physical_distance:0.2f}_{min_annotations}.xlsx')) as writer:
            summary_df.to_excel(writer, sheet_name='summary_results', index=False)
            clip_results_df.to_excel(writer, sheet_name='clip_results', index=False)
            frame_results_df.to_excel(writer, sheet_name='frame_results', index=False)