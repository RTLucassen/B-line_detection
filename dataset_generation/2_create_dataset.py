"""
Create a folder structure and copy frame images that were prepared during the data preparation stage to it.
The images are separated in different folders based on the dataset split that was prepared in 1_create_splits.py,
and whether they were labeled positive or negative.
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import concurrent.futures
import pandas as pd
from math import ceil
from natsort import natsorted

from utils.config import images_folder, info_folder, annotations_folder
from utils.dataset_utils import create_folder_structure, create_datapoint, get_all_frames

# specify settings
frames = 16           # number of frames (should be 1 except for video classification models, in which case 16 was used)

# define directories to use 
input_folder = os.path.join(images_folder, 'processed_frames')    
output_folder = os.path.join(images_folder, 'datasets')

# construct name
name_output_subfolder = f'frames_{frames}'

# define paths to dataset split, number of frames, and annotation information
dataset_split_path = os.path.join(info_folder, 'dataset_split_dictionary.pkl')
frames_path = os.path.join(info_folder, 'frames_dictionary.pkl')
expert_classification = os.path.join(annotations_folder, 'B-line_expert_classification.csv')
expert_annotation = os.path.join(annotations_folder, 'B-line_expert_annotation.csv')

# define list with split names to skip
skipped_splits = []

# define output type
output_type = '.tiff'

# --------------------------------------------------------------------------------

# indicates whether the annotated frame of interest is centered
centered_annotated_frame = False    # can be left untouched
# every Nth frame is selected (for negatives and in case that centered_annotated_frame equals false)
frame_selection_threshold = 1       # can be left untouched

# get the dictionary with the dataset split information and the number of frames per clip
split_dict = pd.read_pickle(dataset_split_path)
frames_dict = pd.read_pickle(frames_path)

# create output folder with given name if it does not exist yet
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# create a subfolder to store dataset images
create_folder_structure(name_output_subfolder, output_folder, split_dict)

# read the csv file with classification results
# divide the clips in positively and negatively labeled
class_df = pd.read_csv(expert_classification)
annotation_df = pd.read_csv(expert_annotation)

# loop over the dataset splits
for split in split_dict.keys():
    
    if split not in skipped_splits:

        print(f'Split: {split}')

        # get all cases in the current split
        cases = split_dict[split]

        # --------------- POSITIVE FRAMES ------------------

        # get the paths to all annotated frames and process them
        annotated_frames = []
        for index, row in annotation_df[annotation_df['case'].isin(cases)].iterrows():
            # get the case, clip, and frame number
            case = row['case']
            clip = str(row['clip']).zfill(3)
            frame = str(row['frame']).zfill(3)
            # add the path to frame to the list
            annotated_frames.append(os.path.join(input_folder, case, f'BEDLUS_{case[-3:]}_{clip}_{frame}.png'))

        # if centered_annotated_frame equals True, loop over all frames in the positively labeled videos
        # and only save those that contain at least one annotated frame
        if centered_annotated_frame == False:
            # define an empty list to store the paths to all negatively labeled frames
            selected_positive_frames = []

            # get the frame names for all frames in the positively labeled clips 
            # which are part of the selected cases in the current dataset split
            for case in cases:
                all_positive_frames = natsorted(get_all_frames(case, input_folder, class_df, 1))
                selected_positive_frames += [frame for i, frame in enumerate(all_positive_frames) if i % frame_selection_threshold == 0]
            print(f'Copying positive frames for split {split} to the dataset directory...')

            # set some inputs of the convert_clip function
            output_directory = os.path.join(output_folder, name_output_subfolder, split, 'pos')
            adjacent_frames = (0, frames-1)
            outside_clip = 'pass'
            create = lambda source: create_datapoint(source, output_directory, output_type, adjacent_frames, frames_dict, outside_clip, annotated_frames) 

            # handle clips using multithreading for speedup
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # copy the negative frames into their new directory
                executor.map(create, selected_positive_frames)

        # else assume that the annotated frame should be in the middle of the segment
        # and in case that the first or last segment frames fall outside of the clip, select the nearest neighboring frame.
        else:        
            print(f'Copying positive frames for split {split} to the dataset directory...')

            # set some inputs of the convert_clip function
            output_directory = os.path.join(output_folder, name_output_subfolder, split, 'pos')
            adjacent_frames = (ceil(frames/2)-1, int(frames/2))
            outside_clip = 'nearest_neighbor'           
            create = lambda source: create_datapoint(source, output_directory, output_type, adjacent_frames, frames_dict, outside_clip)            

            # handle clips using multithreading for speedup
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # copy the negative frames into their new directory
                executor.map(create, annotated_frames)


        # --------------- NEGATIVE FRAMES ------------------

        # define an empty list to store the paths to all negatively labeled frames
        negative_frames = []

        # get the frame names for negatively labeled clips 
        # that are part of all cases in the current dataset split
        for case in cases:
            all_frames = natsorted(get_all_frames(case, input_folder, class_df, 0))
            negative_frames += [frame for i, frame in enumerate(all_frames) if i % frame_selection_threshold == 0]

        print(f'Copying negative frames for split {split} to the dataset directory...')

        # set some inputs of the convert_clip function
        output_directory = os.path.join(output_folder, name_output_subfolder, split, 'neg')
        adjacent_frames = (0, frames-1) if centered_annotated_frame == False else (ceil(frames/2)-1, int(frames/2))
        outside_clip = 'pass' if centered_annotated_frame == False else 'nearest_neighbor'
        create = lambda source: create_datapoint(source, output_directory, output_type, adjacent_frames, frames_dict, outside_clip)

        # handle clips using multithreading for speedup
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # copy the negative frames into their new directory
            executor.map(create, negative_frames)