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
import imageio
import pygifsicle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from sklearn.metrics import auc
from skimage import img_as_ubyte
from skimage.morphology import disk
from torch.utils.data import DataLoader, SequentialSampler

from utils.config import images_folder, models_folder, annotations_folder, info_folder
from utils.evaluation_utils import separate_structures, classify, get_clip_results, aggregate_predictions, isnumber
from utils.augmentation_utils import ClipDataset
from utils.training_utils import get_prediction
from utils.colormap_utils import apply_turbo, text_phantom


def concat_frames(images: list, image_directory: str) -> torch.Tensor:
    """
    Args:
        images:  names of images
        image_directory:  directory where the images are stored.

    Returns:
        concatenated_frames:  all frames of the dataset concatenated.
    """
    concatenated_frames = None

    # create the dataset object
    dataset = ClipDataset(images, image_directory, 1, 0, False, False)

    # loop over the frame indices
    for i in np.arange(dataset.__len__()):
        
        image = dataset.__getitem__(i)
        # add the frame to the concatenated stack of frames
        if not isinstance(concatenated_frames, torch.Tensor):
            concatenated_frames = image
        else:
            concatenated_frames = torch.cat((concatenated_frames, image))

    return concatenated_frames

def count_detections(tensor: torch.Tensor, remove_single_pred: bool) -> torch.Tensor:
    """
    Args:
        tensor:  binary tensor with predictions in each slice (frame, height, width).
        remove_single_pred:  indicates wether single predictions (i.e. ones with no prediction in the frame before or after) should be removed.

    Returns:
        counts:  number of detections
        processed_tensor:  input tensor (with single predictions removed if remove_single_pred).    
    """
    # define the variables
    N_frames = tensor.shape[0]
    counts = torch.zeros(N_frames)
    processed_tensor = torch.zeros_like(tensor)

    # loop over frames
    for frame in np.arange(N_frames):
        # separate all structures in the frame
        structures = torch.from_numpy(separate_structures(tensor[frame, ..., 0]))

        if remove_single_pred:
            # find the pixels intersection of the predictions between the frame of interest and the ones before and after it
            connected_before = tensor[frame-1, ...]*tensor[frame, ...] if frame > 0 else torch.zeros_like(tensor[frame, ...])
            connected_after = tensor[frame, ...]*tensor[frame+1, ...] if frame < N_frames-1 else torch.zeros_like(tensor[frame, ...])
            # take the union of the aforementioned
            connected = torch.clip(connected_before + connected_after, 0, 1)
            # get the intersection between the separated structures in the frame and the connected regions
            intersection = structures * connected.repeat(1, 1, structures.shape[-1])
            positive = torch.clip(torch.sum(intersection, dim=(0,1)), 0, 1).tolist()
            # add the structures that are temporally coherent
            for index, value in enumerate(positive):
                if value == 1:
                    processed_tensor[frame, ..., 0] += structures[..., index]
                    counts[frame] += 1
        else:
            processed_tensor[frame, ...] = tensor[frame, ...]
            counts[frame] += structures.shape[-1]
    
    return counts, processed_tensor 

def create_gif(
    tensor: torch.Tensor, 
    directory: str, 
    name: str, 
    fps: float, 
    timeline: bool = True, 
    label: bool = None, 
    prediction: bool = None,
    raw_prediction: bool = None,
    fps_factor: float = 0.5,
    lossy_compression: bool = True
) -> None:
    """
    Args:
        tensor:  image data with the following dimensions (frames, height, width, color channels) that are converted to a gif.
        directory:  directory where the gif is saved.
        name:  name of the gif
        fps:  frames per second
        timeline:  indicates if a timeline should be embedded.
        label:  indicates if the label should be embedded, and what label if so.
        prediction:  indicates if the prediction should be embedded, and what the prediction is if so.
        fps_factor:  fps is multiplied with this factor (can be used to create slower or faster gifs).
        lossy_compression:  indicates whether the gif should be saved using lossy compression. 
    """
    # define the intensity for the embedded information
    embedding_intensity = 0.7
    
    # check if the output directory exists
    if not os.path.exists(directory):
        raise IOError('Directory does not exist')

    # get the tensor shape information
    frames, height, width, _ = tensor.shape

    # embeds a timeline bar in the gif
    if timeline:
        for frame in np.arange(frames):
            tensor[frame, -8:-5, 0:round(frame/(frames-1)*width), :] = embedding_intensity
    
    # embeds the label in the gif
    if label != None:
        if label:
            tensor[:, 10:13, -19:-10, :] = embedding_intensity
            tensor[:, 7:16, -16:-13, :] = embedding_intensity
        elif label == False:
            tensor[:, 10:13, -19:-10, :] = embedding_intensity
    
    # embeds the prediction results in the gif
    if prediction != None:
        if prediction:
            tensor[:, 25:28, -19:-10, :] = embedding_intensity
            tensor[:, 22:31, -16:-13, :] = embedding_intensity
        elif prediction == False:
            tensor[:, 25:28, -19:-10, :] = embedding_intensity
    
    # embeds the raw prediction result in the gif
    if raw_prediction != None:
        raw_pred = text_phantom(raw_prediction)
        tensor[:, 10:10+raw_pred.shape[0], 10:10+raw_pred.shape[1], :] = raw_pred*embedding_intensity
            
    # create a gif of the tensor
    path = os.path.join(directory, f'{name}.gif')
    writer = imageio.get_writer(path, fps=fps*fps_factor)
    for i in np.arange(tensor.shape[0]):
        writer.append_data(img_as_ubyte(tensor[i, ...]))
    writer.close()

    # applies the lossy compression  
    try:
        options = ["--lossy"] if lossy_compression else []
        pygifsicle.optimize(path, options=options)
    except:
        print('Gif compression using Gifsicle was unsuccessful.')


if __name__ == '__main__':

    # define model evaluation details
    model_subfolder = 'pixel_level'
    model_names = '0001_example_network_0'
    dataset_split = 'val'
    extension = ''

    # define paths
    dataset_split_path = os.path.join(info_folder, 'dataset_split_dictionary.pkl')
    scale_path = os.path.join(info_folder, 'physical_scale_dictionary.pkl')
    processing_path = os.path.join(info_folder, 'processing_dictionary.pkl')
    fps_path = os.path.join(info_folder, 'frame_rate_dictionary.pkl')
    clip_label_path = os.path.join(annotations_folder, 'B-line_expert_classification.csv')

    # define the detection settings
    prob_threshold = 0.50
    classification_thresholds = 0.0  # for B-line presence in videos, was determined based on validation set result
    aggregation_method = 'avg'

    # define gif saving settings
    store_visualizations = False
    embed_timeline = True
    embed_result = True
    lossy_compression = True  # requires Gifsicle to be installed
    
    # define saving settings
    store_spreadsheet = False
    store_curve = True

    # other parameters
    batch_size = 8
    auc_step_size = 0.01

    # --------------------------------------------------------------------------------------------------

    # optional post-processing methods that are not used
    min_relative_pixel_area = 0  
    remove_single_pred = False 

    # configure the number of workers and the device
    num_workers = 0 if sys.platform == 'win32' else 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device\n')

    # load the dataset split information
    split_dict = pd.read_pickle(dataset_split_path)
    scale_dict = pd.read_pickle(scale_path)
    processing_dict = pd.read_pickle(processing_path)
    fps_dict = pd.read_pickle(fps_path)

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

        # create a dictionary to store results
        results_dict = {'case': [], 'clip': [], 'N_detections': [], 'N_frames': [], 'agg_N_detections': [],
            'prediction': [], 'label': [], 'result': [], 'total_time': [], 'avg_time': []}

        # check if the validation set should be used
        if isinstance(dataset_split , list):
            cases = dataset_split
            dataset_split = 'manual'
            print(f'Evaluating {model_name} on the manually selected cases')
        elif dataset_split == 'val':
            split = model_name.split('_')[-1]
            if isnumber(split):
                cases = natsorted(split_dict[split])
                print(f'Evaluating {model_name} on the data split {split}')
            else:
                raise ValueError('Split for evaluation not recognised')
        else:
            cases = natsorted(split_dict[dataset_split]) 
            print(f'Evaluating {model_name} on the data split {dataset_split}')

        # create the subfolder to store all results
        result_dir = os.path.join(model_dir, model_name, f'results_{prob_threshold:0.2f}_{classification_threshold:0.4f}_{aggregation_method}_{dataset_split}{extension}')
        if os.path.exists(os.path.join(result_dir)):
            raise IOError('Results directory already exists.')
        else:    
            os.mkdir(result_dir)

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
                    coordinate_system = settings['label_folder'].split('_')[0]
                    frames = int(settings['label_folder'].split('_')[1])
                    overlap_fraction = 0
                    apply_padding = False
                    video_model = False

                    # get all paths
                    directory = os.path.join(images_folder, f'processed_frames', case)
                    paths = os.listdir(directory)

                    # get all paths to the images that belong to the current clip
                    image_paths = [path for path in paths if path.split('_')[2] == clip]

                    # create the dataset and dataloader object
                    dataset = ClipDataset(image_paths, directory, frames, overlap_fraction, apply_padding, settings['pretrained'], video_model)
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
                            # get the prediction
                            y_pred = get_prediction(model(X))
                            y_pred = y_pred[0] if settings['aux_classification'] else y_pred
                            y_pred = torch.softmax(y_pred, dim=1).to('cpu')

                            # replace None by empty tensor
                            if y_pred_all == None:
                                y_pred_all = torch.zeros((dataset.__len__(), *y_pred.shape[1:]))

                            # add the predictions to the storage variable
                            last = first + X.shape[0]
                            y_pred_all[first:last, ...] += y_pred
                            first = last

                # obtain the average by dividing the summed model predictions by the number of models
                y_pred_all /= len(names)

                # -------------------------  EVALUATION  -------------------------

                # get the maximum pixel detection distance based on the physical scale of the clip
                scale = scale_dict[case][clip][0]
                processing_info = processing_dict[case][clip]

                # create a structuring element
                radius = round(float(settings['label_folder'].split('_')[3][:-2]) / 10 / 2 * scale * processing_info[2])
                min_pixel_area = np.sum(disk(radius)) * min_relative_pixel_area

                # select the frame foreground prediction, threshold the prediction, and separate the structures
                prediction = y_pred_all[:, 1, ...]
                binary_prediction = torch.where(prediction >= prob_threshold, 1, 0)
                
                # initialize a variable to store structure mask
                mask = torch.zeros_like(prediction)[..., None]

                # loop over the frames and extract the number of predicted regions
                for i in np.arange(prediction.shape[0]):
                    structures = separate_structures(binary_prediction[i, ...], min_pixel_area)
                    mask[i, ...] = torch.sum(torch.from_numpy(structures), dim=-1)[..., None] 

                # count the number of detections (and account for optional single prediction removal)
                N_detections, mask = count_detections(mask, remove_single_pred=remove_single_pred)
                
                # get the clip prediction   
                N_frames = prediction.shape[0]
                agg_N_detections = aggregate_predictions(N_detections, aggregation_method=aggregation_method)
                clip_prediction = 1 if agg_N_detections >= classification_threshold else 0
                # get evaluation time
                total_time = time.perf_counter()-start
                avg_time = total_time/N_frames

                # add the clip level detection results to the results dictionary            
                results_dict['clip'].append(clip)
                results_dict['case'].append(case)
                results_dict['N_detections'].append(N_detections.tolist()) 
                results_dict['N_frames'].append(N_frames) 
                results_dict['agg_N_detections'].append(agg_N_detections) 
                results_dict['prediction'].append(clip_prediction)
                results_dict['label'].append(label)
                results_dict['result'].append(classify(clip_prediction, label)) 
                results_dict['total_time'].append(total_time)
                results_dict['avg_time'].append(avg_time) 

                if store_visualizations:
                    # create the output directory if it does not exist yet
                    output_dir = os.path.join(result_dir, f'clip_prediction_gifs')
                    if not os.path.exists(os.path.join(output_dir)):
                        os.mkdir(output_dir)

                    # get the clip tensor        
                    X_all = concat_frames(image_paths, directory)[..., None]

                    # create the clip with overlayed prediction
                    to_rgb = lambda x: torch.tile(x, (1, 1, 1, 3))
                    combined = to_rgb(X_all)*(1-to_rgb(mask)) + torch.from_numpy(apply_turbo(prediction))*to_rgb(mask)

                    # create the gif of the clip with the overlayed prediction
                    label_symbol = label if embed_result else None
                    prediction_symbol = clip_prediction if embed_result else None
                    raw_prediction = f'{agg_N_detections:0.2f}' if embed_result else None
                    create_gif(
                        tensor = combined, 
                        directory = output_dir, 
                        name = f'{case}_{clip}', 
                        fps = float(fps_dict[case][clip]), 
                        timeline = embed_timeline, 
                        label = label_symbol, 
                        prediction = prediction_symbol, 
                        raw_prediction = raw_prediction, 
                        lossy_compression=lossy_compression
                    )

        # get the dataframes with the detection results
        summary_df, clip_results_df = get_clip_results(results_dict)

        # create an Excel file with the detection results
        if store_spreadsheet:
            with pd.ExcelWriter(os.path.join(result_dir, 'individual_clip_results.xlsx')) as writer:
                summary_df.to_excel(writer, sheet_name='summary_results', index=False)
                clip_results_df.to_excel(writer, sheet_name='clip_results', index=False)
        
        if store_curve:
            # create the output directory if it does not exist yet
            output_dir = os.path.join(result_dir, f'clip_classification_curves')
            if not os.path.exists(os.path.join(output_dir)):
                os.mkdir(output_dir)

            # define a variable to keep track of results for different settings of the classification threshold
            track_summary_results = None

            # copy the results dictionary
            results_roc_dict = results_dict.copy()

            threshold_values = list(np.arange(0, max(results_roc_dict['agg_N_detections'])+auc_step_size, auc_step_size))
            for threshold in tqdm(threshold_values):
                # get the predictions and results based on different classification threshold values
                results_roc_dict['prediction'] = [1 if avg_detections >= threshold else 0 for avg_detections in results_roc_dict['agg_N_detections']]
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

            # create a figure with detection performance against one of the detection settings
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
            ax[1,1].set_title(f'AUC: {roc_auc_value:0.4f}')

            get_value = lambda l: str(l[0]) if len(l) == 1 else f'{min(l):0.2f}-{max(l):0.2f}'  
            replace_space = lambda l: ''.join([i if i != ' ' else '_' for i in list(l)])   

            sections = [f'Probability threshold: {prob_threshold:0.2f}',
                        f'Minimum relative pixel area: {min_relative_pixel_area:0.2f}\n',
                        f'Classification threshold: {min(threshold_values):0.2f}-{max(threshold_values):0.2f}',
                        f'Single prediction removal: {remove_single_pred}']
            fig.suptitle('   '.join(sections))
            plt.savefig(os.path.join(output_dir, f'classification_threshold_results.png'))
            plt.close()


