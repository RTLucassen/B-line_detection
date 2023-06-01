# Deep Learning for Detection and Localization of B-Lines in Lung Ultrasound
This repository contains the code for *"Deep Learning for Detection and Localization of B-Lines in Lung Ultrasound"*. 

## Data and Network Parameters
The lung ultrasound dataset and network parameter files for the trained models are available at [Harvard Dataverse](https://doi.org/10.7910/DVN/GLCZRB).
After downloading and unzipping, all files are inside the `BEDLUS-Data` folder.

## Project Setup
1. Install all required libraries: `pip install -r requirements.txt` with Python version 3.7.
2. Run `python utils/config.py` with a specified root folder (in code) to create the project folder structure.
3. Copy all contents from `BEDLUS-Data/LUS_videos` to `project_root/raw`.
The raw-folder should contain a subfolder for each patient (e.g. Case-001) with LUS videos as mp4-files inside.
4. Copy all pkl-files with dataset information `supplementary_files/*.pkl` to `project_root/intermediate/data/info`.
5. Copy all csv-files with annotation information `supplementary_files/*.csv` to `project_root/intermediate/data/annotations`.
6. Copy all contents (or only the network parameter files for the models of interest) from `BEDLUS-Data/trained_networks` to `project_root/intermediate/models`.

## Data Preparation
To extract and preprocess the frames from the lung ultrasound videos, run `python 5_get_processed_frames.py` in the `data_preparation` folder.
A new folder `intermediate/data/images/processed_frames` is created with inside all processed frames.
All other python files in the `data_preparation` folder were used to extract the processing information (e.g. for cropping and rescaling),
which is already available in the supplementary pkl-files.

## Dataset Generation
From the preprocessed frames, datasets with labels or annotations can be created for training deep neural networks at the level of multi-frame clips, single frames, and individual pixels (i.e., segmentation). 
The dataset split is already provided in one of the supplementary files.
1. Run `python 2_create_dataset.py` in the `dataset_generation` folder with `frames = 1` or `frames = 16` (in code) to create the datasets with single frames and 16-frame clips, respectively.
2. Run `python 3_create_labelset.py` in the `dataset_generation` folder with the following combinations of variables (in code) to create the labelsets: 
    - `task = 'segmentation'` with `folder = 'frames_1_disk_4mm'` for the pixel-level
    - `task = 'classification'` with `folder = 'frames_1'` for the frame-level
    - `task = 'classification'` with `folder = 'frames_16'` for the clip-level

## Network Training
Network training details can be configured for a single training run in `single_training_run.py` which is inside the `network_training` folder. Alternatively, the training details can be specified in a csv-file (e.g., see `network_training/example_training_schedule.csv`) when running multiple consecutive experiments using `scheduled_training_runs.py` in the `network_training` folder.

## Evaluation
The evaluation folder contains all code for evaluation, separated in subfolders for each level.
