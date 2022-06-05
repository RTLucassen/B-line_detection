"""
Configure project root path and create project subfolders if they do not exist yet.
"""

import os

folders = []
project_root = None # change to project root path

if project_root == None:
    raise ValueError('First specify the project root')
    
# the raw folder is where the original clips which remain untouched are stored
raw_folder = os.path.join(project_root, 'raw')
folders.append(raw_folder)

# the intermediate folder is where the processed data and models are stored
intermediate_folder = os.path.join(project_root, 'intermediate')
folders.append(intermediate_folder)

# the data folder is where the images, arrays, and sheets are stored
data_folder = os.path.join(intermediate_folder, 'data')
folders.append(data_folder)

# the sheets folder is where spreadsheets with experiment results can be stored
sheets_folder = os.path.join(intermediate_folder, 'sheets')
folders.append(sheets_folder)

# the info folder is where the information about the data (e.g. frame rates) is stored
info_folder = os.path.join(data_folder, 'info')
folders.append(info_folder)

# the images folder is where (processed) images and label maps are stored
images_folder = os.path.join(data_folder, 'images')
folders.append(images_folder)

# the arrays folder is where the numpy arrays with image data are stored
arrays_folder = os.path.join(data_folder, 'arrays')
folders.append(arrays_folder)

# the annotations folder is where the annotation data is stored
annotations_folder = os.path.join(data_folder, 'annotations')
folders.append(annotations_folder)

# the models folder is where the trained deep learning models are stored
models_folder = os.path.join(intermediate_folder, 'models')
folders.append(models_folder)

# define a seed number
seed = 11

if __name__ == '__main__':
    
    # create the folders if they do not exist yet
    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)
        else:
            print('folder {} exists.'.format(folder))
