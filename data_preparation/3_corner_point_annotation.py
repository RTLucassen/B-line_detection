"""
Annotate the four corner points in every lung ultrasound clip
using "corner_annotator" from utils.annotation_utils.py for the purpose of preprocessing.
For ease of use, the annotation tool instructions are appended below.

Tool for annotating the four corner points in the ultrasound videos.
o  Use the left mouse button to click on the corner points. 
   Subsequent scripts expect the following order of clicked points:
   1. top left, 2. bottom left, 3. top right, 4. bottom right.
o  Using the scrolling wheel, the user can scroll through the frames in a clip.
o  Key commands (only work if non-rectified image is shown):
   -  'c' = copy the previous annotation
   -  'n' = skip the current clip and continue to the next one 
            (useful when a corner is not visible in one clip, but is visible in another with the same shape)
   -  'd' = delete previous point that was clicked
   -  't' = rectify the current image (useful to identify inaccuracies in the annotation)
   -  '=' = increase the intensities (useful for low contrast regions, often at the bottom of the ultrasound image)
   -  '-' = decrease the intensities
   -  ']' = add padding to the outside (useful when corners fall just outside of the image) 
   -  '[' = remove padding (only works if padding was already added)
   -  'ENTER' = accept annotation (only works if four points were selected)
   -  'SHIFT' = increases scrolling speed when pressed
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

from utils.config import arrays_folder, info_folder
from utils.annotation_utils import corner_annotator


# define input directory and filename
input_folder = os.path.join(arrays_folder, 'unprocessed_clip_arrays')
point_dict_name = 'corner_points_dictionary.pkl'

# --------------------------------------------------------------------------------

# get cases
cases = os.listdir(input_folder)

# initialize variable to store previous annotation during annotation process
prev_annotation = None

# load annotation dictionary or create empty dictionary to store all newly annotated points
if os.path.exists(os.path.join(info_folder, point_dict_name)):
    point_cases = pd.read_pickle(os.path.join(info_folder, point_dict_name))
    print(f'Annotations loaded for cases: {sorted(point_cases.keys())}')
else:    
    point_cases = {}  

for case in cases:
    # skip case if it was already annotated
    if case in point_cases.keys():
        print(f'{case} was already annotated.')
    else:
        print(case)

        # find clips
        clips_folder = os.path.join(input_folder, case)
        clips = os.listdir(clips_folder)

        # create empty dictionary to store annotated points per clip
        point_clips = {}

        # initialize index variable
        idx = 0
        # use a while loop to allow skipping cases
        while len(clips) != 0: 
            # load clip array and perform annotations
            array = np.load(os.path.join(clips_folder, clips[idx])).astype(np.float32)
            points = corner_annotator(array, prev_annotation)
            
            if points != None:
                # get clip number, add points to the dictionary and set prev_annotation to use for the next clip
                clip_number = os.path.splitext(clips[idx])[0].split('_')[-1]
                point_clips[clip_number] = points
                prev_annotation = points
                clips.remove(clips[idx])
            else:
                idx += 1 
            
            # reset index
            if idx >= len(clips):
                idx = 0                  

        # add dictionary with the points per clip to the dictionary with all cases
        point_cases[case] = point_clips

        # save the file after all clips for a single case were annotated
        file = open(os.path.join(info_folder, point_dict_name), 'wb')
        pickle.dump(point_cases, file)
        file.close()