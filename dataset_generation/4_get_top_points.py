"""
Create file with all top point annotations for evaluation of instance detection.
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import numpy as np
import pandas as pd
from tqdm import tqdm
from pygeoif import from_wkt

from utils.config import images_folder, info_folder, annotations_folder
from utils.conversion_utils import process_point
from utils.labelset_utils import get_top_point


def get_top_points(annotations: pd.core.frame.DataFrame, processing_info: tuple, width: int, height: int) -> list:
    """ 
    Args:
        annotations:  collection of all annotations for one specific frame.
        processing_info:  information about how the image was processed to correct the annotated points.
        width:  width of frame in pixels.
        height: height of frame in pixels.

    Returns:
        top_points:  the top point coordinate for each annotation.
    """ 
    
    top_points = []
    for annotation in eval(annotations.wkt_geoms.tolist()[0]):
        # get the annotated points 
        points = from_wkt(annotation).coords  
        # rescale the points from the 0-100 range to the image-specific range
        points = [(points[i][0]/100*width, points[i][1]/100*height) for i in np.arange(len(points))]
        # convert the points to Cartesian coordinates if specified
        points = [process_point(point, processing_info) for point in points]

        # find the top point
        top_point = get_top_point(points)
        
        top_points.append(top_point)

    return top_points


if __name__ == '__main__':

    # get the directory with all annotated images
    input_folder = os.path.join(images_folder, 'datasets', 'frames_1')

    # define paths to the annotation and processing information
    shape_path = os.path.join(info_folder, 'shape_dictionary.pkl')
    processing_path = os.path.join(info_folder, 'processing_dictionary.pkl')
    annotation_results = os.path.join(annotations_folder, 'B-line_expert_annotation.csv')

    # define the name for csv file with top point information that is created
    output_name = 'top_point_annotations.csv'

    # --------------------------------------------------------------------------------

    # get the dictionary with processing information and dataframe with annotation results
    shape_dict = pd.read_pickle(shape_path)
    processing_dict = pd.read_pickle(processing_path)
    annotation_df = pd.read_csv(annotation_results)

    # create a dictionary to keep track of the paths to images and corresponding top point annotations
    data_dict = {'case': [], 'clip': [], 'frame': [], 'split': [], 'top_points': []}

    # get a list with all case name folders
    split_folders = os.listdir(input_folder)

    for split in split_folders:

        print(f'Split: {split}')

        # for each case, get the image names in the case folder
        images = os.listdir(os.path.join(input_folder, split, 'pos'))

        for image in tqdm(images):
            # decompose the image name to extract the clip number and frame number
            name = os.path.splitext(image)[0]
            _, case, clip, frame = name.split('_')
            case_name = 'Case-'+case
            
            # get height and width of the frame
            height, width = shape_dict[case_name][clip]

            # select all annotations for the frame
            frame_annotations = annotation_df[
                (annotation_df['case'] == case_name) &
                (annotation_df['clip'] == int(clip)) &
                (annotation_df['frame'] == int(frame))
            ]
            top_points = get_top_points(frame_annotations, processing_dict[case_name][clip], width, height)

            # add the information to the data_dict
            data_dict['case'].append(case_name)
            data_dict['clip'].append(clip)
            data_dict['frame'].append(frame)
            data_dict['split'].append(split)
            data_dict['top_points'].append(top_points)

    # save the data dictionary as a csv file
    data_df = pd.DataFrame(data_dict, columns=list(data_dict.keys()))
    data_df.to_csv(os.path.join(annotations_folder, output_name), index=False)

