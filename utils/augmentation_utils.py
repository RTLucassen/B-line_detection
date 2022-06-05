"""
Utility functions and classes for creating a dataset object and the corresponding
generator for training of a neural network using Pytorch.
"""

from typing import Callable

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import torch
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
from math import ceil
from natsort import natsorted
from scipy.ndimage import gaussian_filter
from skimage.transform import AffineTransform, warp

from utils.config import info_folder


# --------------------------------  GENERAL  --------------------------------

def one_hot(index: int, N_classes: int = 2) -> torch.Tensor:
    """
    Args:
        index:  class index that should be one in one-hot encoding.
        N_classes:  total number of classes, which is equal to the length of the encoding.  
    
    Returns:
        encoding:  one-hot encoding.
    """
    # check if the index is inside the number of classes
    if index < 0 or index >= N_classes:
        raise ValueError('Index must be larger than 0 and smaller than number of classes')
    # create and return encoding
    encoding = torch.zeros(N_classes)
    encoding[index] = 1
    return encoding

# -----------------------------  AUGMENTATIONS  -----------------------------

# For clearity and brevity in the implementations of augmentations below,
# all functions expect an image volume X with shape (batch, channels, X, Y, ...)
# and datatype torch.Tensor, which is augmented and returned.
# Label y can either be of the same shape and type as X for segmentation,
# or it can be a value between 0-1 for classification (which is not modified during the augmentation,
# but allows for reuse of the augmentation functions)

def add_intensity(X, y, settings):
    """ 
    Add a random value sampled from a uniform distribution to all pixels in the image.
    """
    # get the argument from settings if available, else use the default value
    intensity_range = (-0.1, 0.1) if 'intensity_range' not in settings else settings['intensity_range']
    # check if the tuple is ordered as (lower bound, upper bound), otherwise correct it
    if intensity_range[0] > intensity_range[1]:
        intensity_range = (intensity_range[1], intensity_range[0])
    # get random intensity values
    value = random.uniform(*intensity_range)
    return threshold(X+value), y

def multiply_intensity(X, y, settings):
    """ 
    Multiply all intensity values in an image with a constant.
    """
    # get the argument from settings if available, else use the default value
    max_factor = 1.1 if 'max_factor' not in settings else settings['max_factor']
    if max_factor >= 1:
        factor = random.uniform(1,max_factor)
    else:
        factor = random.uniform(max_factor, 1)
    if random.randint(0,1) == 0:
        factor = 1/factor
    return threshold(X*factor), y

def gamma_shift(X, y, settings):
    """ 
    Apply a gamma shift to change the contrast of the image.
    """
    # get the argument from settings if available, else use the default value
    gamma_range = (0.67, 1.5) if 'gamma_range' not in settings else settings['gamma_range']
    # check if the tuple is ordered as (lower bound, upper bound), otherwise correct it
    if gamma_range[0] > gamma_range[1]:
        gamma_range = (gamma_range[1], gamma_range[0])
    # sample gamma by first deciding if the value should be above or below 1
    if gamma_range[0] < 1 and gamma_range[1] > 1:
        gamma = random.uniform(gamma_range[0], 1) if random.randint(0,1) == 0 else random.uniform(1, gamma_range[1])
    else:
        gamma = random.uniform(*gamma_range)
    return X**gamma, y

def gaussian_blur(X, y, settings):
    """ 
    Apply gaussian blurring to the image.
    """
    # get the argument from settings if available, else use the default value
    sigma_range = (0.0, 1.0) if 'sigma_range' not in settings else settings['sigma_range']
    # check if the tuple is ordered as (lower bound, upper bound), otherwise correct it
    if sigma_range[0] > sigma_range[1]:
        sigma_range = (sigma_range[1], sigma_range[0])
    # randomly sample a sigma value and blur the image
    sigma = random.uniform(*sigma_range)
    blurred_X = torch.from_numpy(gaussian_filter(X, sigma=sigma))
    return threshold(blurred_X), y

def gaussian_noise(X, y, settings):
    """ 
    Add Gaussian noise to the image.
    """
    # get the argument from settings if available, else use the default value
    max_sigma = 0.02 if 'max_sigma' not in settings else settings['max_sigma']
    # randomly sample a sigma value and sample noise to add to the image
    sigma = random.uniform(0, max_sigma)
    noise = torch.normal(0, sigma, X.shape)
    return threshold(X+noise), y

def temporal_reflection(X, y, settings):
    """ 
    Reflect the frames in the channels dimension (i.e. later frames become earlier frames and vice versa).
    Note: this augmentation is only of interest if adjacent frames are included.
    """
    return torch.flip(X, [0]), y

def vertical_reflection(X, y, settings):
    """ 
    Reflect both the image and the label about a vertical axis in the center of the image.
    """
    X_flipped = torch.flip(X, [2])
    y_flipped = torch.flip(y, [2]) if isinstance(y, torch.Tensor) else y

    return X_flipped, y_flipped

def horizontal_translation(X, y, settings):
    """
    Translate both the image and the label horizontally.
    X (and optionally y) are assumed to consist of the dimensions (channel, height, width).
    """
    width = X.shape[2]
    # get the argument from settings if available, else use the default value    
    max_translation = 0.05 if 'max_translation' not in settings else settings['max_translation']
    t = random.randint(-round(width*max_translation), round(width*max_translation))
    if t > 0:
        translated_X = torch.zeros_like(X)
        translated_X[..., t:] = X[..., :-t]

        if isinstance(y, torch.Tensor):
            translated_y = torch.zeros_like(y)
            translated_y[0, ...] += 1
            translated_y[..., t:] = y[..., :-t]
        else:
            translated_y = y
    elif t < 0:
        translated_X = torch.zeros_like(X)
        translated_X[..., :t] = X[..., -t:]

        if isinstance(y, torch.Tensor):
            translated_y = torch.zeros_like(y)
            translated_y[0, ...] += 1
            translated_y[..., :t] = y[..., -t:]
        else:
            translated_y = y
    else:
        translated_X, translated_y = X, y

    return translated_X, translated_y

def vertical_translation(X, y, settings):
    """
    Translate both the image and the label vertically.
    X (and optionally y) are assumed to consist of the dimensions (channel, height, width).
    """
    height = X.shape[1]
    # get the argument from settings if available, else use the default value
    max_translation = 0.05 if 'max_translation' not in settings else settings['max_translation']
    t = random.randint(-round(height*max_translation), round(height*max_translation))
    if t > 0:
        translated_X = torch.zeros_like(X)
        translated_X[:, t:, :] = X[:, :-t, :]

        if isinstance(y, torch.Tensor):
            translated_y = torch.zeros_like(y)
            translated_y[0, ...] += 1
            translated_y[:, t:, :] = y[:, :-t, :]
        else:
            translated_y = y  
    elif t < 0:
        translated_X = torch.zeros_like(X)
        translated_X[:, :t, :] = X[:, -t:, :]

        if isinstance(y, torch.Tensor):
            translated_y = torch.zeros_like(y)
            translated_y[0, ...] += 1
            translated_y[:, :t, :] = y[:, -t:, :]
        else:
            translated_y = y
    else:
        translated_X, translated_y = X, y

    return translated_X, translated_y

def affine_transform(X, y, settings):
    """
    Apply scaling, shearing, and rotation to the image and the label.
    """
    # find shift to center
    origin = (X.shape[2]//2, X.shape[1]//2) # x, y 
    # get the argument from settings if available, else use the default value
    scale_prob = 0 if 'scale_prob' not in settings else settings['scale_prob']
    max_scale = 10 if 'max_scale' not in settings else settings['max_scale']

    shear_prob = 0 if 'shear_prob' not in settings else settings['shear_prob']
    max_shear = 10 if 'max_shear' not in settings else settings['max_shear']

    rotation_prob = 0 if 'rotation_prob' not in settings else settings['rotation_prob']
    max_rotation = 10 if 'max_rotation' not in settings else settings['max_rotation']

    # get a random scaling factor
    if random.uniform(0,1) <= scale_prob:
        scale_factor = random.uniform(1, max_scale) if max_scale >= 1 else random.uniform(max_scale, 1)
        scale_factor = 1/scale_factor if random.randint(0,1) == 0 else scale_factor
    else:
        scale_factor = 1

    # get a random shearing angle
    if random.uniform(0,1) <= shear_prob:
        shear_rad = random.uniform(-max_shear, max_shear)*np.pi/180  
    else:
        shear_rad = 0
    
    # get a random rotation angle
    if random.uniform(0,1) <= rotation_prob:
        rotation_rad = random.uniform(-max_rotation, max_rotation)*np.pi/180 
    else:
        rotation_rad = 0

    # check if at least one of the three transformations should be applied
    if not (scale_factor == 1 and shear_rad == 0 and rotation_rad == 0):
        # define transformation
        augmentation_transform = AffineTransform(scale=scale_factor, shear=shear_rad, rotation=rotation_rad) 
        transform = centered_transform(origin, augmentation_transform)
        # apply the transformation
        X_transformed = torch.from_numpy(warp(X.permute(1, 2, 0), inverse_map=transform)).permute(2, 0, 1)
        if isinstance(y, torch.Tensor):
            order = 0 if set(y.ravel().tolist()) == {0.0, 1.0} else 3 # account for interpolation for binary images
            y_transformed = torch.from_numpy(warp(y.permute(1, 2, 0), inverse_map=transform, order=order, mode='edge')).permute(2, 0, 1)
        else:
            y_transformed = y
    else:
        X_transformed, y_transformed = X, y

    return threshold(X_transformed), y_transformed

def occlusion(X, y, settings):
    """
    Occlude a random rectangular region of the image.
    X (and optionally y) are assumed to consist of the dimensions (channel, height, width).
    """
    shape = X.shape
    # get the argument from settings if available, else use the default value
    max_height = 0.2 if 'max_height' not in settings else settings['max_height']
    max_width = 0.2 if 'max_width' not in settings else settings['max_width']
    # sample the settings for the random occlusion
    height = random.randint(0, round(max_height*shape[1]))
    width = random.randint(0, round(max_width*shape[2]))
    x_position = random.randint(0, shape[2] - width)
    y_position = random.randint(0, shape[1] - height)
    # apply the occlusion
    X[:, y_position:(y_position+height), x_position:(x_position+width)] = 0

    return X, y

def threshold(X, minimum: float = 0.0, maximum: float = 1):
    """ 
    Helper function to set all values in the image below min to min and all values above max to max.
    """ 
    min_tensor = torch.ones(X.shape, dtype=X.dtype)*minimum
    max_tensor = torch.ones(X.shape, dtype=X.dtype)*maximum
    return torch.where(torch.where(X > max_tensor, max_tensor, X) < min_tensor, min_tensor, X)

def centered_transform(origin: tuple, transform: AffineTransform) -> AffineTransform:
    """
    Args:
        origin:  position to use as origin for the transformation as (x, y)
        transform:  affine transformation class instance 
    
    Return:
        centered_transform:  transformation with respect to the specified origin.
    """
    x_shift, y_shift = origin[0], origin[1]
    # define shifts to and from origin
    shift = AffineTransform(translation=[-x_shift, -y_shift])
    inv_shift = AffineTransform(translation=[x_shift, y_shift]) 
    # combine translation with specified transformation
    return shift+(transform+inv_shift)


# create a dictionary with variables to store all augmentation functions with their name
augmentations = {
    'affine_transform': affine_transform,
    'temporal_reflection': temporal_reflection,
    'vertical_reflection': vertical_reflection,
    'horizontal_translation': horizontal_translation,
    'vertical_translation': vertical_translation,
    'occlusion': occlusion,    
    'add_intensity': add_intensity,
    'multiply_intensity': multiply_intensity,
    'gamma_shift': gamma_shift,
    'gaussian_noise': gaussian_noise,     
    'gaussian_blur': gaussian_blur
}

# -----------------------------  DATASET CLASSES  -----------------------------

class ImageDataset(torch.utils.data.Dataset):
    corner_points_dict = pd.read_pickle(os.path.join(info_folder, 'corner_points_dictionary.pkl'))
    processing_dict = pd.read_pickle(os.path.join(info_folder, 'processing_dictionary.pkl'))

    def __init__(
        self,
        task: str, 
        df: pd.core.frame.DataFrame, 
        batch_size: int,
        negatives: int, 
        seed: float, 
        folder: str = '',
        aug_settings_dict: dict = {},
        aug_function_dict: dict = augmentations,
        positives_faction: float = 1.0,
        pretrained: bool = False,
        padding: tuple = (0, 0, 0, 0),
        drop_last: bool = False
    ) -> None:
        """ 
        Args:
            task: either 'segmentation', 'frame_classification' or 'video_classification' (influences the label output)
            df:  contains paths to all images and corresponding labels, together with the fold and class information.
            negatives:  number of negative examples per batch
            batch_size:  number of examples in a batch during training (required to allocate at least one positive example to each batch).
            seed:  seed to make the random sampling of batches reproducible.
            folder:  folder directory that is added before the paths in the dataframe (since these can be different on different platforms).
            aug_settings_dict:  contains augmentation function names as keys and the corresponding settings as values:
            positives_fraction:  fraction from all available positives that is used in one epoch.
            pretrained:  indicates if a pretrained model is used, which requires a 3-channel output to match the RGB channels.
            padding:  add zero-padding to the image, specified as the number of pixels added to the (left, right, top, bottom).
            drop_last:  if True, incomplete batches at the end of an epoch are dropped.
        """
        # store the input arguments as instance attributes
        self.task = task
        self.df = df
        self.batch_size = batch_size
        self.negatives = negatives
        self.seed = seed
        self.folder = folder
        self.aug_settings_dict = aug_settings_dict
        self.aug_function_dict = aug_function_dict
        self.positives_fraction = positives_faction
        self.pretrained = pretrained
        self.padding = padding
        self.drop_last = drop_last

        # check if the batch size value is valid 
        if self.batch_size <= 0:
            raise ValueError('Batch size must be equal or larger than one.')
        # check if there is at least one positive datapoint in a batch
        if self.batch_size - self.negatives <= 0:
            raise ValueError('Batch must at least contain one positive datapoint.')
        # check if the specified task is valid
        if self.task not in ['segmentation', 'frame_classification', 'video_classification']:
            raise ValueError('Unrecognized task')    

        # define a seeded numpy random number generator
        self.rng = np.random.default_rng(self.seed)

        # retrieve all paths to the images and corresponding labels
        self.positive_paths = natsorted(list(zip(df[df['b-lines_present'] == 'pos'].image_path.to_list(), 
                                                 df[df['b-lines_present'] == 'pos'].label_info.to_list())))
        self.negative_paths = natsorted(list(zip(df[df['b-lines_present'] == 'neg'].image_path.to_list(),
                                                 df[df['b-lines_present'] == 'neg'].label_info.to_list())))
        # get the dictionaries with the number of frames per clip for all positive and negative clips
        self.weight_dict = self.get_weight_dict()

        # get the dictionary with the augmentation function names as keys and the corresponding augmentation function as values
        self.augmentation_dict = self.get_aug_dict()

        # initialize instance variables to store image paths, corresponding weights, and to keep a record of the sampled images
        self.paths = []
        self.frame_weights = [] 
        self.record = {}

        # shuffle data (if negatives is not False, new negatives will be sampled)
        self.shuffle()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple:
        """ 
        Args:
            idx:  extract image-label pair using this index

        Returns:
            image:  image tensor, rescaled in the 0-1 range and optionally augmented.
            label:  corresponding label tensor, rescaled in the 0-1 range and optionally augmented.
        """
        # load the image and label
        image = load_image(os.path.join(self.folder, self.paths[idx][0]))
        label = load_image(os.path.join(self.folder, self.paths[idx][1])) if self.task == 'segmentation' else int(self.paths[idx][1])

        # optionally add padding to the images (and labels for segmentation tasks)
        image = torch.nn.functional.pad(image[None, ...], self.padding, 'constant', 0)[0, ...]
        label = torch.nn.functional.pad(label[None, ...], self.padding, 'replicate')[0, ...] if self.task == 'segmentation' else label        

        # apply other spatial and intensity augmentations if they were configured using the order above
        for aug in self.augmentation_dict.keys():
            image, label = self.augmentation_dict[aug](image, label) 

        # add an additional channel if the task is video classification
        image = image[None, ...] if self.task == 'video_classification' else image
        # convert the label to its one-hot encoding for classification tasks
        label = one_hot(label) if self.task in ['frame_classification', 'video_classification'] else label
        
        # repeat the image 3 times in the channel dimension 
        if self.pretrained:
            if image.shape[0] != 1:
                raise ValueError('Image must have a single channel to be repeated 3 times.')
            else:
                repeats = [3] + [1]*(len(image.shape)-1)
                image = image.repeat(*repeats)
        
        return image, label, self.frame_weights[idx], self.paths[idx][0]

    def get_weight_dict(self) -> dict:
        """
        Returns:
            total_weight_dict:  contains the weight (1/number of frames) for each positive and negative clip.
        """
        weight_dicts = []
        harmonic_means = []
        # get the frame weights by counting how many frames from each clip there are
        for paths in [self.positive_paths, self.negative_paths]:
            # create a dictionary to store counts
            count_dict = {}
            for path, _ in paths:
                # get the case and clip
                elements = os.path.split(path.replace('\\','/'))[1].split('_')
                key = (elements[1], elements[2])
                # add an example to count dict
                if key not in count_dict.keys():
                    count_dict[key] = 1
                else:
                    count_dict[key] += 1
            
            # calculate and append the harmonic mean
            counts = count_dict.values()
            value = len(counts)/sum(counts) if sum(counts) > 0 else 1 
            harmonic_means.append(value)
            
            # convert the counts to weightings (1/count)
            weight_dict = {key: (1 / count_dict[key]) for key in count_dict.keys()}
            weight_dicts.append(weight_dict)
        
        # multiply all weights from negative frames with the ratio as a correction
        ratio = harmonic_means[0]/harmonic_means[1]
        weight_dicts[1] = {key: (weight_dicts[1][key] * ratio) for key in weight_dicts[1].keys()}
        # combine positive and negative dictionary
        total_weight_dict = {**weight_dicts[0], **weight_dicts[1]}

        return total_weight_dict

    def get_aug_dict(self) -> dict:
        """
        Returns:
            augmentation_dict:  contains augmentation name as keys and the corresponding function as values
        """
        # create a dictionary to store the augmentation name and function pairs
        augmentation_dict = {}
        
        # collect the settings for the affine transformation augmentation (different from the others)
        if 'affine_transform' in self.aug_function_dict:
            settings = {}
            include = False
            for aug in ['scale', 'shear', 'rotation']:
                if aug in self.aug_settings_dict.keys():
                    settings[f'{aug}_prob'] = self.aug_settings_dict[aug][0]
                    settings = {**settings, **self.aug_settings_dict[aug][1]}
                    if settings[f'{aug}_prob'] > 0:
                        include = True    
            if include == True:        
                augmentation_dict['affine_transform'] = augmentation_wrapper(self.aug_function_dict['affine_transform'], (1.0, settings))

        # loop over the available augmentation functions (except for the affine transformation)
        for aug in self.aug_function_dict:
            if aug == 'affine_transform':
                continue
            # check if the augmentation is present in the settings specification
            # and if the probability is valid and not zero
            if aug in self.aug_settings_dict.keys():
                if self.aug_settings_dict[aug][0] < 0 or self.aug_settings_dict[aug][0] > 1:
                    raise ValueError('Probability value for augmentation must be between 0 and 1.')
                elif self.aug_settings_dict[aug][0] != 0:
                    augmentation_dict[aug] = augmentation_wrapper(self.aug_function_dict[aug], self.aug_settings_dict[aug])

        return augmentation_dict        

    def shuffle(self) -> None:
        """ 
        Create a list with randomly shuffled examples.
        If negative examples should be used, a random selection of negative examples
        with the size of a specified ratio compared to the number of positive examples are included.
        Every batch subsequent examples in the list contain at least one positive example.
        """
        # reset the instance attributes with paths to the images and labels in use
        self.paths = []
        self.frame_weights = [] 

        # create variables with shuffled positive and negative paths
        positive_indices = list(np.arange(len(self.positive_paths)))
        self.rng.shuffle(positive_indices)
        # select a fraction of all positive frames to use in the epoch (if positives_fraction < 1)
        positive_indices = positive_indices[:round(len(positive_indices)*self.positives_fraction)]

        negative_indices = list(np.arange(len(self.negative_paths)))
        self.rng.shuffle(negative_indices)
        # in case that there are less negative than positive examples, repeat the list
        if len(negative_indices) > 0: # prevent a division by zero error   
            negative_indices *= ceil(len(positive_indices)/len(negative_indices))

        # get the number of positives per batch and the number of batches
        positives = self.batch_size - self.negatives
        batches = len(positive_indices)//positives if self.drop_last else ceil(len(positive_indices)/positives)

        # add positive and negative paths to list in the correct order
        for batch in np.arange(batches):
            batch_paths = []
            # add the positive and negative frames
            batch_paths += [self.positive_paths[i] for i in positive_indices[batch*positives:(batch+1)*positives]]
            batch_paths += [self.negative_paths[i] for i in negative_indices[batch*self.negatives:(batch+1)*self.negatives]]
            self.rng.shuffle(batch_paths)

            self.paths += batch_paths

        # add the weightings to the list
        for path, _ in self.paths:
            # get the case and clip
            elements = os.path.split(path.replace('\\','/'))[1].split('_')
            key = (elements[1], elements[2])
            # append the frame weight to the list
            self.frame_weights.append(self.weight_dict[key])

    def get_class_weights(self) -> list:
        """
        Find the number of pixels belonging to each class in the dataset 
        and calculate weights to balance the classes.

        Returns:
            weights:  weight for each example in the dataset.
        """
        # define a variable to store the total number of pixels that belongs to each class
        aggregated_counts = {}

        # loop over all positive images and count the number of pixels for each class
        for path in self.paths:
            # load the labelmap
            image = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.folder, path[1]).replace('\\','/')))) / 255
            # count the pixels per class
            image_counts = torch.sum(image, dim=tuple(np.arange(len(image.shape))[1:])).tolist()
            # add the counted pixels to the total
            if aggregated_counts == {}:
                aggregated_counts = {i: image_counts[i] for i in np.arange(len(image_counts))}
            else:
                aggregated_counts = {i: aggregated_counts[i] + image_counts[i] for i in np.arange(len(image_counts))}

        # calculate the weights
        total = sum(aggregated_counts.values())
        classes = len(aggregated_counts)
        weights = [total/(classes*aggregated_counts[i]) for i in np.arange(classes)]

        return weights

    def get_record(self, directory: str, name: str = 'record.csv') -> None:
        """
        Args:
            directory:  output directory where the record csv-file is saved.
            name:  name of record csv file.
        """
        # order the data to be used as columns in the correct order
        columns = list(zip(*natsorted(self.record.items())))
        # create a dataframe out of the columns
        df = pd.DataFrame.from_dict({'image': columns[0], 'count': columns[1]})
        # write the dataframe as a csv file
        df.to_csv(os.path.join(directory, name), index=False)

    def add_to_record(self, sampled_paths: list) -> None:
        """
        Args:
            sampled_paths:  list with paths from the images that were sampled in the last batch
                            to add to the record dictionary.
        """
        for path in sampled_paths:
            if path in self.record.keys():
                self.record[path] += 1
            else:
                self.record[path] = 1


class ClipDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        images: list, 
        image_directory: str, 
        frames: int,
        overlap_fraction: float,
        apply_temporal_padding: bool, 
        pretrained: bool,
        video_dimension: bool = False,
        padding: tuple = (0, 0, 0, 0)
    ) -> None:
        """
        Args:
            images:  names of images
            image_directory:  directory where images are stored.
            frames:  total number of frames in a selected subset.
            overlap_fraction:  faction of the segment that overlaps with the previous or next segment.
            apply_temporal_padding:  add padding frames to the start and end of the clip.
            pretrained:  if True, repeat the segment three times in the channel dimension.
            video_dimension:  indicates if a 3rd dimension should be added for video models.
            padding:  add zero-padding to the image, specified as the number of pixels added to the (left, right, top, bottom).
        """
        self.images = natsorted(images)
        self.image_directory = image_directory
        self.frames = frames
        self.overlap_fraction = overlap_fraction
        self.overlap_frames = round(self.frames * self.overlap_fraction)
        self.apply_temporal_padding = apply_temporal_padding
        self.pretrained = pretrained
        self.padding = padding
        self.video_dimension = video_dimension

        # check if the overlap fraction is a valid value
        if self.overlap_fraction < 0 or self.overlap_fraction > 1:
            raise ValueError('Overlap fraction must be in the [0,1] range.')

        # check if the overlapping number of frames is equal to the total number of frames
        if self.overlap_frames >= self.frames:
            raise ValueError('Overlapping number of frames must be smaller than the total number of frames')

        # add additional images to the start and end of the clip s.t. each frame is the center of a selected subset of frames
        if self.apply_temporal_padding:
            adjacent_frames = (ceil(self.frames/2)-1, int(self.frames/2))
            self.images = [self.images[0]]*adjacent_frames[0] + self.images + [self.images[-1]]*adjacent_frames[0]

        # create a dictionary with indices as keys and image names as in a list as values
        self.data_dict = self.create_data_dict()                

    def __len__(self) -> int:
        return len(self.data_dict)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            idx:  index indicating the collection of frames to load
        
        Returns:
            item:  contains collection of frames
        """
        # get the frames and initialize item
        frames = self.data_dict[idx]
        item = None

        for frame in frames:
            # read the image and set the intensities to the 0-1 range
            image = load_image(os.path.join(self.image_directory, frame))
            # add the image to storage variable
            item = image if item == None else torch.cat((item, image), dim=0)

        # optionally add padding to the images
        item = torch.nn.functional.pad(item[None, ...], self.padding, 'constant', 0)
        # remove the first dimension for non-video models
        if not self.video_dimension:
            item = item[0, ...]

        # repeat the image 3 times in the channel dimension 
        if self.pretrained:
            if item.shape[0] != 1:
                raise ValueError('Image must have a single channel to be repeated 3 times.')
            else:
                repeats = [3] + [1]*(len(item.shape)-1)
                item = item.repeat(*repeats)

        return item

    def create_data_dict(self) -> dict:
        """
        Create a dictionary with indices as keys and a list of subsequent frame names as values.
        If there are frames at the end of the clip which do not fit into an entire selection of subsequent frames,
        add a last segment starting from the end of the clip, which will have more overlap with the previous selection.
        
        Returns:
            data_dict:  contains indices as keys and a list of subsequent frame names as values.
        """
        # define the dictionary to store collections of frames in
        data_dict = {}
        # define the states
        start = index = 0

        # generate data dictionary by looping until all selections of the specified length are made
        while start + self.frames <= len(self.images):
            data_dict[index] = self.images[start:start+self.frames]
            # update the states
            start += (self.frames-self.overlap_frames)
            index += 1

        # if the clip is shorter than the selected clip length, select all frames and repeat the last frame
        if index == 0:
            data_dict[index] = self.images + [self.images[-1]]*(self.frames - len(self.images))

        # if the last section does not perfectly fit, add another section starting from the back
        elif data_dict[index-1][-1] != self.images[-1]:
            data_dict[index] = self.images[-self.frames:]
        
        return data_dict


# ------------------------  DATASET CLASS HELPER FUNCTIONS  -----------------------

def load_image(path: str) -> torch.Tensor:
    """
    Args:
        path:  path to image
    Returns:
        image:  loaded image with required processing
    """
    image = sitk.GetArrayFromImage(sitk.ReadImage(path.replace('\\','/')))
    image = torch.from_numpy(image)/255
    # add a dimension if necessary
    image = image[None, ...] if len(image.shape) == 2 else image
    return image

# define a function to embed the probability and settings in the augmentation function
def augmentation_wrapper(function: Callable, settings: tuple) -> Callable:
    """
    Args:
        function:  performs augmentation
        settings:  tuple containing the probability for the augmentation to be applied
                   and additional settings for the augmentation function.
    Returns:
        augmentation:  augmentation function with the settings embedded                   
    """
    prob, aug_settings = settings
    def augmentation(X, y, other={}):
        f = function(X, y, {**aug_settings, **other}) if random.uniform(0,1) <= prob else (X, y)
        return f
    return augmentation