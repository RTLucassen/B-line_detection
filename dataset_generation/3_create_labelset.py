"""
Create a folder structure and create the label for the dataset that was prepared in 2_create_dataset.py.
For segmentation tasks, labelmaps are created. For classification tasks, only a file with the corresponding label is created.
The labelmaps are separated in different folders based on the dataset split that was prepared.
Only a single empty labelmap is created for all negative cases of a fold.
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk 
from tqdm import tqdm
from natsort import natsorted

from utils.config import images_folder, info_folder, annotations_folder
from utils.labelset_utils import create_label, add_background
from utils.dataset_utils import create_folder_structure


# define task (either segmentation or classification) 
# and the number of frames for each item in the dataset
task = 'segmentation'
folder = 'frames_1'

# define segmentation-specific parameters
if task == 'segmentation':
    # define spatial label map settings (not including classes)
    output_shape = (256, 384)
    diameter = 0.4 # cm
    output_datatype = '.tiff'

# select directory
input_folder = os.path.join(images_folder, 'datasets', folder)

# construct name
addition = f'_disk_{diameter*10:.0f}mm' if task == 'segmentation' else '' 
name_output_subfolder = folder + addition

# define directory and paths to dataset and annotation information
dataset_split_path = os.path.join(info_folder, 'dataset_split_dictionary.pkl')
scale_path = os.path.join(info_folder, 'physical_scale_dictionary.pkl')
shape_path = os.path.join(info_folder, 'shape_dictionary.pkl')
processing_path = os.path.join(info_folder, 'processing_dictionary.pkl')
corner_points_path = os.path.join(info_folder, 'corner_points_dictionary.pkl')

expert_classification = os.path.join(annotations_folder, 'B-line_expert_classification.csv')
expert_annotation = os.path.join(annotations_folder, 'B-line_expert_annotation.csv')

# define name for csv file with image-label combinations
image_label_combinations = 'image_label_combinations.csv'

# define output folder and data type for images
output_folder = os.path.join(images_folder, 'labelsets')

# --------------------------------------------------------------------------------

# get the dictionary with the dataset split information, annotated corner points, and processing information
split_dict = pd.read_pickle(dataset_split_path)
scale_dict = pd.read_pickle(scale_path)
shape_dict = pd.read_pickle(shape_path)
processing_dict = pd.read_pickle(processing_path)
points_dict = pd.read_pickle(corner_points_path)

# create output folder with given name if it does not exist yet
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# create a subfolder to store dataset labels
if task == 'segmentation':
    create_folder_structure(name_output_subfolder, output_folder, split_dict)
else:
    output_subfolder = os.path.join(output_folder, name_output_subfolder)
    if not os.path.isdir(output_subfolder):
        os.mkdir(output_subfolder)

# read the csv file with classification results
# divide the clips in positively and negatively labeled
class_df = pd.read_csv(expert_classification)
annotation_df = pd.read_csv(expert_annotation)

# create a dictionary to keep track of paths to images and corresponding labels
data_dict = {'split': [], 'b-lines_present':[], 'image_path': [], 'label_info': []}

radius_summary = {}

# loop over the dataset splits
for split in split_dict.keys():
    
    print(f'Split: {split}')

    # get all cases in the current split
    cases = split_dict[split]

    if task == 'classification':
        
        for classification, label in zip(['pos', 'neg'], [1, 0]):
            # define path to images with a specific label directory and get all filenames
            frames_path = os.path.join(input_folder, split, classification)
            frames = os.listdir(frames_path)

            # update data dictionary
            data_dict['split'] += [split]*len(frames)
            data_dict['b-lines_present'] += [classification]*len(frames)
            data_dict['image_path'] += [os.path.join(frames_path, frame)[len(images_folder)+1:] for frame in frames]
            data_dict['label_info'] += [label]*len(frames)

    elif task == 'segmentation':

        # --------------- POSITIVE FRAMES ------------------

        # define the path to positive frames directory and get all filenames
        positive_frames_path = os.path.join(input_folder, split, 'pos')
        positive_frames = os.listdir(positive_frames_path)

        for frame_name in tqdm(positive_frames):
            # extract the case and filename
            _, case, clip, frame = os.path.splitext(frame_name)[0].split('_')
            case_name = 'Case-'+case        

            # select all annotations for the frame 
            frame_annotations = annotation_df[
                (annotation_df['case'] == case_name) & 
                (annotation_df['clip'] == int(clip)) & 
                (annotation_df['frame'] == int(frame))
            ]
            
            if len(frame_annotations) != 1:
                raise ValueError('Annotations for each frame should be single dataframe entries. Please check the annotation spreadsheet.')

            # get clip-specific scale, processing, and corner point information
            scale = scale_dict[case_name][clip][0]
            height, width = shape_dict[case_name][clip]
            processing_info = processing_dict[case_name][clip]
            corner_points = points_dict[case_name][clip]

            # create the labelmap
            label, radius = create_label(
                annotations = frame_annotations, 
                corner_points = corner_points, 
                processing_info = processing_info,
                width = width,
                height = height,
                scale = scale,
                diameter = diameter,
                output_shape = output_shape,
            )

            # update radius summary
            if radius not in radius_summary.keys():
                radius_summary[radius] = 1
            else:
                radius_summary[radius] += 1

            # create the label and add a background map
            label = add_background(label, axis=0)

            # save label
            label_path = os.path.join(output_folder, name_output_subfolder, split, 'pos', f'BEDLUS_{case}_{clip}_{frame}_label{output_datatype}')
            sitk.WriteImage(sitk.GetImageFromArray(label), label_path)

            # update data dictionary
            data_dict['split'].append(split)
            data_dict['b-lines_present'].append('pos')
            data_dict['image_path'].append(os.path.join(positive_frames_path, f'BEDLUS_{case}_{clip}_{frame}{output_datatype}')[len(images_folder)+1:])
            data_dict['label_info'].append(label_path[len(images_folder)+1:])

        # --------------- NEGATIVE FRAMES ------------------
        
        if split != 'test':
            # define the path to negative frames directory and get all filenames
            negative_frames_path = os.path.join(input_folder, split, 'neg')
            negative_frames = os.listdir(negative_frames_path)

            # save a single empty label map to use for all negative images and add a background to it
            label = np.zeros(output_shape, dtype=np.uint8)
            label = add_background(label, axis=0)
            label = torch.from_numpy(label)

            # save label
            label_path = os.path.join(output_folder, name_output_subfolder, split, 'neg', f'negative_label{output_datatype}')
            sitk.WriteImage(sitk.GetImageFromArray(label), label_path)

            # update data dictionary
            data_dict['split'] += [split]*len(negative_frames)
            data_dict['b-lines_present'] += ['neg']*len(negative_frames)
            data_dict['image_path'] += [os.path.join(negative_frames_path, frame)[len(images_folder)+1:] for frame in negative_frames]
            data_dict['label_info'] += [label_path[len(images_folder)+1:]]*len(negative_frames)
    
    else:
        raise ValueError('Unrecognized task')

# save the data dictionary as csv file
data_df = pd.DataFrame(data_dict, columns=['split', 'b-lines_present', 'image_path', 'label_info'])
data_df.to_csv(os.path.join(output_folder, name_output_subfolder, image_label_combinations), index=False)

# notify the user about radii in constructed labels
if task == 'segmentation':
    print('')
    print('Overview of radii:')
    for r in natsorted(radius_summary.keys()):
        print(f'Radius: {r}, Count: {radius_summary[r]}')
