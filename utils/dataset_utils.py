"""
Utility functions for dataset generation.
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

def create_folder_structure(name: str, directory: str, split_dict: dict) -> None:
    """ 
    Create folder structure to store data in per split divided between positive and negative examples.

    Args:
        name: folder name
        directory: directory where to create the new folder with subfolders
        split_dict: contains information about the dataset split for creating subfolders.
    """
    # check if the directory exists
    if not os.path.exists(directory):
        raise ValueError('Directory already exists')
    
    output_folder = os.path.join(directory, name)

    # create a folder with the given name if it does not exist yet
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # loop over the splits
    for split in split_dict.keys():

        # create a subfolder to store images for a specific fold
        output_subfolder = os.path.join(output_folder, split)
        if not os.path.isdir(output_subfolder):
            os.mkdir(output_subfolder)

        # create subfolders to store positive and negative images for each specific fold
        positive_subfolder = os.path.join(output_subfolder, 'pos')
        negative_subfolder = os.path.join(output_subfolder, 'neg')
        
        for subfolder in [positive_subfolder, negative_subfolder]:
            if not os.path.isdir(subfolder):
                os.mkdir(subfolder)

def create_datapoint(
    source_path: str, 
    new_directory: str,
    output_type: str, 
    adjacent_frames: tuple,
    frames_dict: dict,
    outside_clip: str,
    must_include: list = None
) -> None:
    """ 
    Args:
        source_path:  path to image.
        new_directory:  new directory to where the image is copied (with the same filename).
        output_type:  datatype for output.
        adjacent_frames:  number of adjacent frames before and after the frame of interest to include.
        frames_dict:  dictionary that contains the total number of frames for each clip, sorted per case.
        outside_clip:  indicates how to handle frame position in the segments that fall outside of the clip.
                       if equal to 'shift', shift all frames until all positions are valid.
                       if equal to 'nearest_neighbor', copy the nearest valid frame.
                       if equal to 'pass', skip this segment.
        must_include:  list with filenames of which at least one must be included in each segment of subsequent frames to be saved.
    """
    datapoint = None

    # extract the filename
    old_directory, filename = os.path.split(source_path)
    name, extension = os.path.splitext(filename)
    # extract the case, clip, and frame number 
    _, case, clip, frame = name.split('_')

    # get the total number of frames for that clip
    N_frames = frames_dict['Case-'+case][clip]

    # loop over the selected indices and create the corresponding paths
    frame_numbers = [number for number in range(int(frame)-adjacent_frames[0], int(frame)+adjacent_frames[1]+1)]
    corrected_frame_numbers = correct_indices(frame_numbers, 0, N_frames-1, outside_clip)

    if corrected_frame_numbers != None: # (None can be returned if outside_clip was set to pass)
        paths = [os.path.join(old_directory, f'BEDLUS_{case}_{clip}_{str(number).zfill(3)}{extension}') for number in corrected_frame_numbers]   
        
        # check if the segment includes at least one annotated frame
        condition = True if must_include == None else any(path in paths for path in must_include)
        if condition:
            # load frames and store the dataframe
            for path in paths:
                if not isinstance(datapoint, np.ndarray):
                    datapoint = sitk.GetArrayFromImage(sitk.ReadImage(path))[None, ...]
                else:
                    datapoint = np.concatenate((datapoint, sitk.GetArrayFromImage(sitk.ReadImage(path))[None, ...]), axis=0)

            # save the image
            sitk.WriteImage(sitk.GetImageFromArray(datapoint), os.path.join(new_directory, f'{name}{output_type}'))

def correct_indices(indices: list, minimum: int, maximum: int, outside_range: str) -> list:
    """
    Args:
        indices:  contains list with indices.
        minimum:  minimum index
        maximum:  maximum index
        outside_range:  indicates how to handle frame position in the segments that fall outside of the clip.
                        if equal to 'shift', shift all frames until all positions are valid.
                        if equal to 'nearest_neighbor', copy the nearest valid frame.
                        if equal to 'pass', return None.
    Returns:
        corrected_indices:  input paths but non-existant onces are replaced by nearest paths that does exist.
    """
    if outside_range == 'shift':
        # get the correction value based on the minimum and maximum
        # add the correction factor (i.e. shift the segment)
        if min(indices) < minimum:
            correction = minimum-min(indices)
        elif max(indices) > maximum:
            correction = maximum-max(indices)
        else:
            correction = 0
        corrected_indices = [number + correction for number in indices]
    
    elif outside_range == 'nearest_neighbor':
        # clip the frame numbers outside of the clip (i.e. find the nearest neighboring frame)
        corrected_indices = [min(max(number, minimum), maximum) for number in indices]    
    
    elif outside_range == 'pass':
        corrected_indices = None if min(indices) < minimum or max(indices) > maximum else indices

    else:
        raise ValueError('Invalid argument for outside_clip')
    
    return corrected_indices

def get_all_frames(case: str, directory: str, class_df: pd.DataFrame, select_label: int) -> list:
    """ 
    Args:
        case:  case name
        directory:  directory where to search for the frames.
        class_df:  dataframe with label information per class.
        select_label:  either 0 (negative clips) or 1 (positive clips), indicating which of the two is selected.

    Returns:
        frame_paths:  paths to all frames for the selected case.
    """
    # select all clips of the given case and select all clips from the given label
    case_df = class_df[class_df.case == case]
    selected_df = case_df[case_df.label == select_label]
    # get a list with the selected clip numbers
    selected_clips = [str(clip).zfill(3) for clip in selected_df['clip'].to_list()]

    # get the frame names of all clips for the given case
    frames_names = os.listdir(os.path.join(directory, case))
    # only select the frame names from selected clips
    frame_paths = [os.path.join(directory, case, name) for name in frames_names if name.split('_')[2] in selected_clips]
    
    return frame_paths