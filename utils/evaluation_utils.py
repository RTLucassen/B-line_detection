"""
Utility functions for evaluation of model performance.
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import torch
import numpy as np
import pandas as pd
from natsort import natsorted
from skimage.measure import label
from scipy.ndimage import center_of_mass

from utils.conversion_utils import euclidean_dist


# ----------------------  GENERAL  --------------------------

def isnumber(number: str) -> bool:
    """
    Args:
        number:  string with possibly a number.

    Returns:
        result:  boolean result if string is a number.
    """
    try:
        int(number)
    except:
        return False
    else:
        return True

def aggregate_predictions(predictions: torch.Tensor, aggregation_method: str) -> float:
    """
    Args:
        predictions:  tensor with predictions for each frame/clip.
        aggregation_method:  string indicating method of aggregation.
    
    Returns:
        clip_prediction:  aggregated prediction for entire clip.
    """
    # convert to float type
    predictions = predictions.type(torch.float32)

    if aggregation_method == 'avg':
        clip_prediction = float(torch.mean(predictions))

    elif aggregation_method == 'max':
        clip_prediction = float(torch.max(predictions))

    elif 'max_avg' in aggregation_method:
        length = int(aggregation_method.split('_')[2])
        overlap_fraction = float(aggregation_method.split('_')[3])
        averages = []
        # check if the length of the clip is larger than the specified length
        if predictions.shape[0] < length:
            averages.append(float(torch.mean(predictions)))
        # calculate the average prediction of sequences of a specified length, 
        else:
            i = 0
            overlap = round(length*overlap_fraction)

            # check if the overlap fraction is valid
            if overlap_fraction < 0 or overlap_fraction > 1:
                raise ValueError('Overlap fraction must be in the [0,1] range.')
            # check if the overlap is smaller than the length
            if overlap >= length:
                raise ValueError('Overlap must be smaller than length.')

            while i + length <= len(predictions):
                averages.append(float(torch.mean(predictions[i:i+length])))
                i += (length-overlap)
            
            if i != len(predictions)-overlap:
                averages.append(float(torch.mean(predictions[-length:])))

        # find the maximum of the averages
        clip_prediction = max(averages)

    else:
        raise ValueError('Aggregation method not recognized')
    
    return clip_prediction


# --------------------------  DETECTION EVALUATION  ------------------------------

def separate_structures(array: np.ndarray, min_pixel_threshold: int = 0) -> np.ndarray:
    """ 
    Args:
        array:  binary array with predictions or ground truth labels of the shape: (X, Y, ...).
        min_pixel_threshold:  minimum number if pixels for a structure to be considered.

    Returns:
        one_hot_structures:  one-hot encoded volume with one connecting structure per slice of the shape (X, Y, ..., S).
    """
    # get the map where separated object have a different value
    structures = label(array)
    # convert to one-hot encoding
    one_hot_structures = (np.arange(structures.max()) == structures[..., None]-1).astype(np.uint8)
    # find all slices in the encoding for which the structure contains more than the minimum number of pixels
    selection = [i for i in np.arange(one_hot_structures.shape[-1]) if np.sum(one_hot_structures, axis=(0,1))[i] >= min_pixel_threshold]

    return one_hot_structures[..., selection]

def get_centroids(one_hot_array: np.ndarray) -> dict:
    """ 
    Args:
        one_hot_array:  one-hot encoded volume with one connecting structure per slice of the shape (X, Y, ..., S).

    Returns:
        centroids:  contains centroid coordinate per structure
    """
    # create a dictionary with the index and corresponding centroid position
    centroids = [center_of_mass(one_hot_array[..., idx])[::-1] for idx in np.arange(one_hot_array.shape[-1])]        
    return centroids

def evaluate_detections(predicted_points: list, annotated_points: list, max_distance: float) -> dict:
    """
    Args:
        predicted_points:  coordinates (x, y) for each predicted point.
        annotated_points:  coordinates (x, y) for each annotated point.

    Returns:
        detection_dict:  contains information about true and false positive detections and false negative points.
    """
    # create a dictionary to store info about true_positives, false_positives, and false_negatives
    detection_dict = {'TP':[], 'FP':[], 'FN':[]}

    # all true negatives
    if predicted_points == [] and annotated_points == []:
        pass
    # all false positives
    elif annotated_points == []:
        detection_dict['FP'] = predicted_points
    # all false negatives
    elif predicted_points == []: 
        detection_dict['FN'] = annotated_points 
    else:
        # create an array to store distances
        results = np.zeros((len(predicted_points), len(annotated_points)))

        # loop over all binary structures in ground truth annotation and thresholded prediction
        # and calculate the distances between the structures
        for i in np.arange(len(predicted_points)):
            for j in np.arange(len(annotated_points)):
                dist = euclidean_dist(predicted_points[i], annotated_points[j])
                results[i, j] = dist

        # check for false positive structures
        for i in np.arange(len(predicted_points)):
            if np.min(results[i, :]) > max_distance:
                detection_dict['FP'].append(predicted_points[i])

        # check for false negatives structures
        for j in np.arange(len(annotated_points)):
            comparison = np.where(results[:, j] <= max_distance, 1, 0)
            if np.min(results[:, j]) > max_distance:
                detection_dict['FN'].append(annotated_points[j])
            # if it is not, it must be a true positive
            else:
                # find the specific connection to be able to draw a line between
                # the structures in the visualization
                for i in np.arange(len(predicted_points)):
                    if comparison[i] == 1:
                        detection_dict['TP'].append((predicted_points[i], annotated_points[j]))
    
    return detection_dict

def detect_instances(
    structures: np.array, 
    top_points: list, 
    max_pixel_distance: float, 
) -> dict:
    """
    Args:
        structures:  predicted structures (either binary or with probabilities after thresholding and removal of small structures)
        top_points:  annotated top points used as ground truth.
        max_pixel_distance:  maximum distance between ground truth point and prediction centroid to be counted as detected.

    Returns:
        detection_dict:  contains true positive, false positive, and false negative coordinates for each frame.
    """
    # get the (probability weighted) centroids from the structures
    centroids = get_centroids(structures)
    # get true positives, false positives and false negatives
    detection_dict = evaluate_detections(centroids, top_points, max_pixel_distance) 
    
    return detection_dict

def get_detection_statistics(TP: int, FP: float, FN: float) -> tuple:
    """
    Args:
        TP:  number of true positives
        FP:  number of false positives
        FN:  number of false negatives
    
    Returns:
        precision:  percentage of your positive predictions that is correct.
        recall:  percentage of actual positive samples that was correctly classified. 
        f1_score:  harmonic mean of precision and recall.
    """
    # handle edge cases to prevent division by zero errors
    if TP+FP+FN == 0:
        precision = 1
        recall = 1
    else:    
        precision = TP/(TP+FP) if (TP+FP) != 0 else 1 
        recall = TP/(TP+FN) if (TP+FN) != 0 else 1 

    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0  

    return precision, recall, f1_score


# ------------------------  DETECTION RESULTS DATAFRAMES  ----------------------------

def natsort_dict(dictionary: dict) -> dict:
    """
    Naturally sort elements of lists in dictionary.

    Args:
        dictionary:  contains dicionary with column names as keys and lists as values (which are columns).

    Returns:
        dictionary:  input dictionary after sorting.
    """
    sorted_lists = list(zip(*natsorted(list(zip(*[dictionary[key] for key in dictionary.keys()])))))
    dictionary = {key: list(sorted_lists[i]) for i, key in enumerate(dictionary)}
    
    return dictionary

def average_with_nan(values: list) -> float:
    """
    Args:
        values:  values to be averaged, which can contain np.nan items (not included in average).
    
    Returns:
        average:  average of values excluding np.nan items.
        N_frames:  number of frames used in the calculation.
    """
    filtered_values = [value for value in values if not np.isnan(value)]
    N_frames = len(filtered_values)
    average = np.nan if N_frames == 0 else sum(filtered_values) / N_frames

    return average, N_frames

def get_detection_results(results_dict: dict) -> tuple:
    """
    Args:
        results_dict:  contains detection results for each frame.
    
    Returns:
        summary_df:  detection results summary.  
        clip_results_df:  detection results per clip.
        frame_results_df:  detection results per frame.
    """
    # natural sort the results and create a frame results dataframe
    results_dict = natsort_dict(results_dict)
    frame_results_df = pd.DataFrame.from_dict(results_dict)

    # create a dictionary for average frame results into clip results
    clip_results_dict = {'case': [], 'clip': [], 'precision': [], 'recall': [], 'f1-score': []}

    # loop over the cases and clips
    for case in list(set(frame_results_df.case)):

        case_df = frame_results_df[frame_results_df.case == case]

        for clip in list(set(case_df['clip'])):

            clip_results_dict['case'].append(case)
            clip_results_dict['clip'].append(clip)
            
            # calculate the sum of TP, FP, and FN for all frames in one clip
            TP = sum(list(case_df[case_df['clip'] == clip]['TP']))
            FP = sum(list(case_df[case_df['clip'] == clip]['FP']))
            FN = sum(list(case_df[case_df['clip'] == clip]['FN']))

            # calculate precision, recall, and F1-score
            precision, recall, f1_score = get_detection_statistics(TP, FP, FN)

            clip_results_dict['precision'].append(precision)
            clip_results_dict['recall'].append(recall)
            clip_results_dict['f1-score'].append(f1_score)

    # natural sort the results and create a frame results dataframe
    clip_results_dict = natsort_dict(clip_results_dict)
    clip_results_df = pd.DataFrame.from_dict(clip_results_dict)

    # calculate average precision and recall
    avg_precision = average_with_nan(clip_results_dict['precision'])[0]
    avg_recall = average_with_nan(clip_results_dict['recall'])[0]

    # create a dataframe with summary results
    summary_results_dict = {'total TP': [sum(results_dict['TP'])],  
                            'total FP': [sum(results_dict['FP'])],
                            'total FN': [sum(results_dict['FN'])],
                            'avg. precision': [avg_precision],
                            'avg. recall': [avg_recall],
                            'F1-score': [2*avg_precision*avg_recall/(avg_precision+avg_recall)]}
                   
    summary_df = pd.DataFrame.from_dict(summary_results_dict)

    return summary_df, clip_results_df, frame_results_df


# ----------------------  CLASSIFICATION RESULTS DATAFRAMES  --------------------------

def classify(prediction: int, label: int) -> str:
    """
    Args:
        prediction:  0 for negative prediction and 1 for positive prediction.
        label:  0 for negative label and 1 for positive label.

    Returns:
        result: 'TP' for true positives, 'FP' for false positives, 'FN' for false negatives, 'TN' for true negatives
    """
    if prediction == 1 and label == 1:
        return 'TP'
    elif prediction == 1 and label == 0:
        return 'FP'
    elif prediction == 0 and label == 1:
        return 'FN'
    elif prediction == 0 and label == 0:
        return 'TN'
    else:
        raise ValueError('Invalid input argument for either prediction or label')

def get_clip_results(results_dict: dict) -> tuple:
    """
    Args:
        results_dict:  contains detection results for each frame.
    
    Returns:
        summary_df:  detection results summary.  
        clip_results_df:  detection results per clip.
    """
    # natural sort the results and create a frame results dataframe
    results_dict = natsort_dict(results_dict)
    clip_results_df = pd.DataFrame.from_dict(results_dict)

    # create a dictionary for average frame results into clip results
    summary_dict = {'N_clips': [len(results_dict['clip'])]}
    
    # get the total number of TP, FP, FN, TN
    for key in ['TP', 'FP', 'FN', 'TN']:
        summary_dict[f'total {key}'] = [list(clip_results_df['result']).count(key)]
    
    # handle special cases to prevent division by zero errors
    if summary_dict['total TP'][0]+summary_dict['total FP'][0]+summary_dict['total FN'][0] == 0:
        summary_dict['precision'] = [1]
        summary_dict['recall'] = [1]
    else:  
        # calculate the precision 
        if summary_dict['total TP'][0]+summary_dict['total FP'][0] == 0: # does not really occur for reasonable thresholds
            summary_dict['precision'] = [np.nan]
        else:
            summary_dict['precision'] = [summary_dict['total TP'][0] / (summary_dict['total TP'][0] + summary_dict['total FP'][0])]
        # calculate the recall
        if summary_dict['total TP'][0]+summary_dict['total FN'][0] == 0: # does not really occur for reasonable thresholds
            summary_dict['recall'] = [np.nan]
        else:
            summary_dict['recall'] = [summary_dict['total TP'][0] / (summary_dict['total TP'][0] + summary_dict['total FN'][0])]

    if np.isnan(summary_dict['precision'][0]) or np.isnan(summary_dict['recall'][0]) or summary_dict['precision'][0]+summary_dict['recall'][0] == 0:
        summary_dict['f1-score'] = [0]
    else:
        summary_dict['f1-score'] = [(2 * summary_dict['precision'][0] * summary_dict['recall'][0]) / (summary_dict['precision'][0] + summary_dict['recall'][0])]

    # other metrics
    summary_dict['specificity'] = [summary_dict['total TN'][0] / (summary_dict['total TN'][0] + summary_dict['total FP'][0])] 
    summary_dict['accuracy'] = [(summary_dict['total TP'][0] + summary_dict['total TN'][0]) / summary_dict['N_clips'][0]]   
    
    summary_df = pd.DataFrame.from_dict(summary_dict)

    return summary_df, clip_results_df

