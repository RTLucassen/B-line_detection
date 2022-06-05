# Deep Learning for Detection and Localization of B-Lines
This repository contains the code for *"Deep Learning for Detection and Localization of B-Lines"*. 

## Project Setup
1. Install all required libraries: `pip install -r requirements.txt`
2. Run *repository > utils > config.py* with a specified root folder to create the project folder structure.
3. Copy all folders with raw lung ultrasound videos to *project_root > raw*.
4. Copy all pkl-files with dataset information from *repository > supplementary_files* to *project_root > intermediate > data > info*.
5. Copy all csv-files with annotation information from *repository > supplementary_files* to *project_root > intermediate > data > annotations*.
6. Copy all contents (or only the models that are of interest) to *project_root > intermediate > models*.

## Data and Network Parameters
The lung ultrasound data and network parameters files for the trained models are available from: *(add link)*.

## Data Preparation
To extract and preprocess the frames from the lung ultrasound videos, run *repository > data_preparation > 5_get_processed_frames.py*.
A new folder *project_root > intermediate > data > images > processed_frames* is created with inside all processed frames.
All other python files in *repository > data_preparation* are used to extract the processing information (e.g. for cropping and rescaling),
which is already available in the supplementary pkl-files.

## Dataset Generation
From the preprocessed frames, datasets with labels or annotations can be created for training deep neural networks at the level of multi-frame clips, single frames, and individual pixels (i.e., segmentation). 
The dataset split is already provided in one of the supplementary files.
1. Run *repository > dataset_generation > 2_create_dataset.py* with `frames=1` or `frames=16` to create the datasets with single frames and 16-frame clips, respectively.
2. Run *repository > dataset_generation > 3_create_labelset.py* with the following combinations: `task=frame_classification` with `folder='frames_1'`,  `task=video_classification` with `folder='frames_16'`, and `task=segmentation` with `folder='frames_1_disk_4mm'` to create the labelsets.

## Network Training
Network training details can be configured for a single training run in *repository > network_training > single_training_run.py.py*, 
or in a csv-file (e.g., see *repository > network_training > example_training_schedule.csv*) when running multiple experiments in a row using *repository > network_training > scheduled_training_runs.py*.

## Evaluation
The evaluation folder contains all code for evaluation, separated in subfolders for each level.