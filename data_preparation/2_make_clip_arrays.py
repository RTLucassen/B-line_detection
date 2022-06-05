"""
Create and save clips as numpy arrays using separated frame images.
(used for 3_corner_point_annotation.py and 4_retrieve_physical_scale.py)
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from imageio import imread
from natsort import natsorted
from skimage import img_as_float

from utils.config import images_folder, arrays_folder, info_folder


# define directories
input_folder = os.path.join(images_folder, 'unprocessed_frames (jpg)')
output_folder = os.path.join(arrays_folder, 'unprocessed_clip_arrays')
# get the dictionary with original clip shapes
output_shape = pd.read_pickle(os.path.join(info_folder, 'shape_dictionary.pkl'))
# define skipped cases
skipped_cases = []
    
# --------------------------------------------------------------------------------

# check if output folder exists
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# get all files in the input folder
cases = os.listdir(input_folder)

for case in tqdm(cases):
    if case not in skipped_cases:
        # define the output folder for cases
        # create it if it does not exist yet
        output_case_folder = os.path.join(output_folder, case)
        if not os.path.isdir(output_case_folder):
            os.mkdir(output_case_folder)

        # get all frame names and sort them
        frames = natsorted(os.listdir(os.path.join(input_folder, case)))
        
        # create a dictionary to store all frames per clip
        clip_dictionary = {}
        for frame in frames:
            clip = os.path.splitext(frame.split('_')[2])[0]
            if clip in clip_dictionary:
                clip_dictionary[clip].append(frame)
            else:
                clip_dictionary[clip] = [frame]

        # loop over all clips
        for clip in clip_dictionary.keys():

            # get the shape of frames in the clip
            if isinstance(output_shape, tuple):
                clip_shape = output_shape
            else:
                clip_shape = output_shape[case][clip]
            
            # select the frames for this clip
            clip_frames = clip_dictionary[clip]
            # allocate memory for the array
            clip_array = np.zeros((len(clip_frames), *clip_shape), dtype=np.float16)
            # get the paths to the frames
            paths = [os.path.join(input_folder, case, frame) for frame in clip_frames]

            # handle frames using multithreading for speedup
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # convert the clips to separate frames
                frames = executor.map(imread, paths)
                # add frame information to array
                for i, frame in enumerate(frames):
                    clip_array[i, ...] = img_as_float(frame)  
                
            np.save(os.path.join(output_case_folder, f'clip_{clip}.npy'), clip_array)