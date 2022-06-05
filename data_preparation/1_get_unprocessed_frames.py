"""
Save all frames of the raw lung ultrasound clips (.mp4) as separate images (.jpg by default).
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import pickle
import concurrent.futures
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from imageio import get_reader, imwrite

from utils.config import raw_folder, images_folder, info_folder


def convert_clip(clip: str, output_dir: str, output_type: str) -> tuple:
    """ 
    Args:
        clip:  path to clip.
        output_dir:  path to output directory where all individual frames will be stored.
        output_type:  datatype that is used to store the frames.
    
    Returns:
        shape:  image shape (height, width) extracted from metadata. 
        i:  last index indicating the number of frames in a clip.  
    """
    # read the clip and get the shape
    reader = get_reader(clip)
    shape = reader.get_meta_data()['source_size'][::-1] # flip dimensions, result is (height, width)

    # loop over frames and save them as individual images 
    # (rgb is converted to greyscale and image is stored as unsigned byte format, with values in [0, 255])
    for i, frame in enumerate(reader):
        imwrite(os.path.join(output_dir, os.path.basename(clip).replace('.mp4', f'_{str(i).zfill(3)}{output_type}')), img_as_ubyte(rgb2gray(frame)))

    return shape, i


if __name__ == '__main__':

    # define directories, paths, and filenames
    input_folder = os.path.join(raw_folder)
    output_folder = os.path.join(images_folder, 'unprocessed_frames (jpg)')
    shape_dict_name = 'shape_dictionary.pkl'
    frame_dict_name = 'frames_dictionary.pkl'

    # define output data type for images
    output_datatype = '.jpg'

    # --------------------------------------------------------------------------------

    # create the output directory if it does not exist yet
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    # create dictionaries to collect shape and number of frames of each clip per case
    shape_cases, frame_cases = {}, {}

    # get all case names from the input folder
    cases = sorted([case for case in os.listdir(input_folder) if case.lower().startswith('case')])

    for case in tqdm(cases):
        # define the output directory and create it if it does not exist yet
        case_output = os.path.join(output_folder, case)
        if not os.path.isdir(case_output):
            os.mkdir(case_output)
        
        # create dictionaries to collect the shape and number of frames for each clip
        shape_clips, frame_clips = {}, {}

        # define the path to the directory with all clips
        clips_folder = os.path.join(input_folder, case)
        # natural sort the clips
        clips = natsorted(glob(clips_folder + '/*.mp4'))

        # set some inputs of the convert_clip function
        convert = lambda clips: convert_clip(clips, case_output, output_datatype)

        # handle clips using multithreading for speedup
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # convert the clips to separate frames
            results = executor.map(convert, clips)
            # add results to the dictionaries
            for i, result in enumerate(results):
                shape_clips[os.path.splitext(os.path.split(clips[i])[1].split('_')[2])[0]] = result[0]
                frame_clips[os.path.splitext(os.path.split(clips[i])[1].split('_')[2])[0]] = result[1]+1

        shape_cases[case] = shape_clips
        frame_cases[case] = frame_clips

    # save frame shape dictionary variable
    file = open(os.path.join(info_folder, shape_dict_name), 'wb')
    pickle.dump(shape_cases, file)
    file.close()

    # save number of frames dictionary variable
    file = open(os.path.join(info_folder, frame_dict_name), 'wb')
    pickle.dump(frame_cases, file)
    file.close()

        