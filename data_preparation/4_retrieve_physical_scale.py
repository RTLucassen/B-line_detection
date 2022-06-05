"""
Annotate two points on the embedded ruler and set the corresponding distance
to manually determine the pixel/cm ratio in every lung ultrasound clip
using "scale_annotator" from utils.annotation_utils.py.
For ease of use, the annotation tool instructions are appended below.

Tool for annotating two points in the embedded ruler in a ultrasound frame for calculation of the pixel/cm ratio.
o  Use the left mouse button to click on the points. The order of the points is not important. 
o  Using the scrolling wheel, the user can change the distance over which the ratio will be calculated.
o  Key commands:
    -  'c' = copy the previous annotation
    -  'r' = remove all annotated points
    -  'ENTER' = accept annotation (only works if two points were selected)
    -  'CTRL' = enable point annotation or distance toggling when pressed
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
from utils.conversion_utils import euclidean_dist
from utils.annotation_utils import scale_annotator


# define input directory and filename
input_folder = os.path.join(arrays_folder, 'unprocessed_clip_arrays')
point_dict_name = 'physical_scale_dictionary.pkl'

# --------------------------------------------------------------------------------

# get cases
cases = os.listdir(input_folder)

# initialize variable to store previous annotation and distance during annotation
prev_annotation = None
prev_distance = None

# load annotation dictionary or create empty dictionary to store all ratios
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

        # create empty dictionary to store all ratios per clip
        point_clips = {}

        # initialize index variable
        idx = 0
        # use a while loop to allow skipping cases
        while len(clips) != 0: 
            # load clip array and select the first frame
            image = np.load(os.path.join(clips_folder, clips[idx])).astype(np.float32)[0, ...]
            # annotate the two points and set the distance
            points, distance = scale_annotator(image, prev_annotation, prev_distance)
            
            if points != None:
                # get clip number
                clip_number = os.path.splitext(clips[idx])[0].split('_')[-1]
                # round point coordinates and calculate the ratio
                prev_annotation, prev_distance = points, distance
                ratio = euclidean_dist(points[0], points[1])/distance
                
                point_clips[clip_number] = (ratio, points, distance)
                print(f'Clip {clip_number}:', ratio)
                
                clips.remove(clips[idx])
            else:
                idx += 1 
            
            # reset index
            if idx >= len(clips):
                idx = 0 

        # add dictionary with the ratios per clip to the dictionary with all cases
        point_cases[case] = point_clips

        # save the file after all clips for a single case were annotated
        file = open(os.path.join(info_folder, point_dict_name), 'wb')
        pickle.dump(point_cases, file)
        file.close()