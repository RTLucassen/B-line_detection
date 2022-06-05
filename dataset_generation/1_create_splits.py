"""
Create train and test set splits. For the training set, subdivide it into different folds for cross-validation.
Save the training folds in a dictionary and the test cases in a list as a pickle variable.
"""

from typing import Any

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import pickle
import numpy as np

from utils.config import seed, raw_folder, info_folder


def check_dict(dictionary: dict, item: Any) -> list:
    """ 
    Args:
        dictionary:  any dictionary
        item:  any item that could be inside the dictionary.
    
    Returns:
        list with keys for which the corresponding argument equates the input argument item.
    """
    return [key for key in dictionary.keys() if item in dictionary[key]]


if __name__ == '__main__':

    # define test set size and number of cross-validation folds for training
    test_set_size = 23
    N_folds = 5
    split_dict_name = 'dataset_split_dictionary.pkl'

    # --------------------------------------------------------------------------------

    # define dictionary to store dataset split information
    split_dict = {}

    # define numpy random number generator
    rng = np.random.default_rng(seed)

    # get all case names and remove cases from the list which do not contain any data
    unfiltered_cases = sorted([case for case in os.listdir(raw_folder) if case.lower().startswith('case')])
    cases = [case for case in unfiltered_cases if os.listdir(os.path.join(raw_folder, case)) != []]

    # get the number of cases after filtering
    N_cases = len(cases)

    # check if the specified test set size is valid
    if test_set_size > N_cases:
        raise ValueError('Specified test set size exceeds total number of cases')

    # get the fold size of the training set
    fold_size = (N_cases-test_set_size)/N_folds 

    # warn the user if the remaining cases for training cannot be evenly divided among the different folds
    if fold_size != int(fold_size):
        print(f'Warning: the cases in the training set ({N_cases-test_set_size}) cannot be evenly divided among the {N_folds} folds.')

    # shuffle the indices and get a list with the shuffled cases
    indices = np.arange(N_cases)
    rng.shuffle(indices)
    shuffled_cases = [cases[idx] for idx in indices]

    # populate the dictionary with the cases for the test set and training folds
    split_dict['test'] = shuffled_cases[:test_set_size]
    for i, case in enumerate(shuffled_cases[test_set_size:]):
        if f'{i%N_folds}' not in split_dict.keys():
            split_dict[f'{i%N_folds}'] = [case]
        else:
            split_dict[f'{i%N_folds}'].append(case)

    # print statistics about dataset devision
    print('Number of clips per dataset split')
    for key in split_dict.keys():
        clips = 0
        for case in split_dict[key]:
            clips += len(os.listdir(os.path.join(raw_folder, case)))
        print(f'{key}:\t{clips} clips')
        print(sorted(split_dict[key]))

    # save the split dictionary variable   
    file = open(os.path.join(info_folder, split_dict_name), 'wb')
    pickle.dump(split_dict, file)
    file.close()