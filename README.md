# Deep Learning for Detection and Localization of B-Lines in Lung Ultrasound
This repository contains all code to support the paper ***"Deep Learning for Detection and Localization of B-Lines in Lung Ultrasound"***,
published in IEEE Journal of Biomedical and Health Informatics. 
[[`arXiv`](https://arxiv.org/abs/2302.07844)][[`JBHI`](https://ieeexplore.ieee.org/abstract/document/10143623)]

<div align="center">
  <img width="40%" alt="clip1" src=".github\clip1.gif"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img width="40%" alt="clip2" src=".github\clip2.gif">
</div>
&nbsp;
<div align="center">
  <img width="40%" alt="clip3" src=".github\clip3.gif"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img width="40%" alt="clip4" src=".github\clip4.gif">
</div>

## Data and Network Parameters
As part of this work, we curated the ***"Boston Emergency Department Lung UltraSound (BEDLUS)"*** dataset.
This dataset comprises 1,419 videos from 113 patients, along with expert annotations.
All videos are labeled as positive or negative for the presence of B-lines. For the positive videos, a total of 15,755 B-lines are annotated on 10,371 frames.

The BEDLUS dataset and network parameter files for the trained models are available at [Harvard Dataverse](https://doi.org/10.7910/DVN/GLCZRB).

## Project Setup
1. Install all required libraries (Python version 3.7): 
```
pip install -r requirements.txt
```
2. Create the project folder structure after specifying the root folder in `config.py`: 
```
python utils/config.py
```
3. After requesting access, download and unzip the dataset (`data.zip`) and network parameter files (`networks_clip-level.zip`, `networks_frame-level.zip`, `networks_pixel-level.zip`).
3. Copy all contents from `data/LUS_videos` to `project_root/raw`.
The raw-folder should contain a subfolder for each patient (e.g. Case-001) with LUS videos as mp4-files inside.
4. Copy all pkl-files with dataset information `supplementary_files/*.pkl` (in this repository) to `project_root/intermediate/data/info`.
5. Copy all csv-files with annotation information `supplementary_files/*.csv` (in this repository) to `project_root/intermediate/data/annotations`.
6. Copy all contents (or only the network parameter files for the models of interest) from `clip-level`, `frame-level`, `pixel-level` to `project_root/intermediate/models`.

## Data Preparation
Extract and preprocess the frames from the lung ultrasound videos:
```
python data_preparation/5_get_processed_frames.py
```
A new folder `intermediate/data/images/processed_frames` is created with inside all processed frames.
All other python files in the `data_preparation` folder were used to extract the processing information (e.g. for cropping and rescaling),
which is already available in the supplementary pkl-files.

## Dataset Generation
From the preprocessed frames, datasets with labels or annotations can be created for training deep neural networks at the level of multi-frame clips, single frames, and individual pixels (i.e., segmentation). 
The dataset split is already provided in one of the supplementary files.
1. Specify the dataset type in `2_create_dataset.py`: 
    - `frames = 1` for frame-level and pixel-level
    - `frames = 16` for clip-level
2. Create the dataset:
```
python dataset_generation/2_create_dataset.py
```
3. Specify the labelset type in `3_create_dataset.py`:
    - `task = 'segmentation'` with `folder = 'frames_1_disk_4mm'` for the pixel-level
    - `task = 'classification'` with `folder = 'frames_1'` for the frame-level
    - `task = 'classification'` with `folder = 'frames_16'` for the clip-level
4. Create the labelset:
```
python dataset_generation/3_create_labelset.py
```

## Network Training & Evaluation
Network training details can be configured for a single training run in `single_training_run.py`, which is inside the `network_training` folder. Alternatively, the training details can be specified in a csv-file (e.g., see `network_training/example_training_schedule.csv`) when running multiple consecutive experiments using `scheduled_training_runs.py` in the `network_training` folder.

The evaluation folder contains all code that was used for the evaluation, separated in subfolders for each level.

## Citing
If you found our work useful in your research, please consider citing our paper:
```
@article{lucassen2023deep,
  title={Deep Learning for Detection and Localization of B-Lines in Lung Ultrasound},
  author={Lucassen, Ruben T and Jafari, Mohammad H and Duggan, Nicole M and Jowkar, Nick and Mehrtash, Alireza and Fischetti, Chanel and Bernier, Denie and Prentice, Kira and Duhaime, Erik P and Jin, Mike and others},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2023},
  publisher={IEEE}
}
```
