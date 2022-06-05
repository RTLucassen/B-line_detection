"""
Collects and prints dataset information.
Using the predefined seed:

DATASET INFORMATION
Number of patients: 113
Number of LUS videos: 1,419
   Positive videos: 719 (50.7%)
   Negative videos: 700 (49.3%)
   Average frame rate: 22.2 (min: 15.0, max: 46.0)

Number of annotations: 15,755
Number of frames: 188,670
Number of frames with one or more annotations: 10,371

FOLD-SPECIFIC INFORMATION
test
Number of patients: 23
Number of clips: 307
   Positive videos: 170 (55.4%)
   Negative videos: 137 (44.6%)
      Annotated frames: 2404

0
Number of patients: 18
Number of clips: 216
   Positive videos: 105 (48.6%)
   Negative videos: 111 (51.4%)
      Annotated frames: 1616

1
Number of patients: 18
Number of clips: 197
   Positive videos: 102 (51.8%)
   Negative videos: 95 (48.2%)
      Annotated frames: 1479

2
Number of patients: 18
Number of clips: 247
   Positive videos: 118 (47.8%)
   Negative videos: 129 (52.2%)
      Annotated frames: 1565

3
Number of patients: 18
Number of clips: 225
   Positive videos: 110 (48.9%)
   Negative videos: 115 (51.1%)
      Annotated frames: 1683

4
Number of patients: 18
Number of clips: 227
   Positive videos: 114 (50.2%)
   Negative videos: 113 (49.8%)
      Annotated frames: 1624
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import pandas as pd

from utils.config import raw_folder, annotations_folder, info_folder

# define the paths
expert_classification = os.path.join(annotations_folder, 'B-line_expert_classification.csv')
expert_annotation = os.path.join(annotations_folder, 'B-line_expert_annotation.csv')
frames_path = os.path.join(info_folder, 'frames_dictionary.pkl')
frame_rates_path = os.path.join(info_folder, 'frame_rate_dictionary.pkl')
dataset_split_path = os.path.join(info_folder, 'dataset_split_dictionary.pkl')

print('\nDATASET INFORMATION')

# find the total number of available raw clips
total_raw_clips = 0
empty_folders = 0
raw_subfolders = [path for path in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, path))]

for subfolder in raw_subfolders:
    raw_clips = len([clip for clip in os.listdir(os.path.join(raw_folder, subfolder)) if os.path.splitext(clip)[-1] == '.mp4'])
    total_raw_clips += raw_clips
    if raw_clips == 0:
        empty_folders += 1

print(f'Number of patients: {len(raw_subfolders)-empty_folders:,}')
print(f'Number of LUS videos: {total_raw_clips:,}')

# check if the number of available video classifications matches with the number of raw videos
expert_classification_df = pd.read_csv(expert_classification)
if total_raw_clips != len(expert_classification_df):
    print('Warning: the number of available raw LUS videos does not match')


# find the number of positive and negative clips
positive_clips = sum(expert_classification_df['label'])
negative_clips = len(expert_classification_df)-positive_clips
print(f'   Positive videos: {positive_clips} ({positive_clips/total_raw_clips*100:0.1f}%)')
print(f'   Negative videos: {negative_clips} ({negative_clips/total_raw_clips*100:0.1f}%)')

# find the distribution of frame rates
frame_rates_distribution = {}
frame_rates_dict = pd.read_pickle(os.path.join(info_folder, frame_rates_path))
for case in frame_rates_dict.keys():
    for clip in frame_rates_dict[case]:
        frame_rate = frame_rates_dict[case][clip]
        if frame_rate not in frame_rates_distribution:
            frame_rates_distribution[frame_rate] = 1
        else:
            frame_rates_distribution[frame_rate] += 1

average_fr = sum([fr*frame_rates_distribution[fr] for fr in frame_rates_distribution.keys()])/total_raw_clips
min_fr = min(frame_rates_distribution.keys())
max_fr = max(frame_rates_distribution.keys())
print(f'   Average frame rate: {average_fr:0.1f} (min: {min_fr:0.1f}, max: {max_fr:0.1f})\n')


# find the total number of annotations
frames_dict = pd.read_pickle(frames_path)
expert_annotation_df = pd.read_csv(expert_annotation)
annotated_lines = expert_annotation_df['wkt_geoms']

frames = 0
for case in frames_dict.keys():
    for clip in frames_dict[case]:
        frames += frames_dict[case][clip]

annotated_frames = len(annotated_lines)
annotations = sum([len(eval(line)) for line in annotated_lines])

print(f'Number of annotations: {annotations:,}')
print(f'Number of frames: {frames:,}')
print(f'Number of frames with one or more annotations: {annotated_frames:,}')

print('\nFOLD-SPECIFIC INFORMATION')

# find the fold-specific information
split_dict = pd.read_pickle(dataset_split_path)

for split in split_dict:
    print(split)

    cases = split_dict[split]
    print(f'Number of patients: {len(cases)}')
    
    clips_in_fold = expert_classification_df[expert_classification_df['case'].isin(cases)]
    print(f'Number of clips: {len(clips_in_fold)}')

    # find the number of positive and negative clips
    positive_clips = sum(clips_in_fold['label'])
    negative_clips = len(clips_in_fold)-positive_clips
    annotated_frames = len(expert_annotation_df[expert_annotation_df['case'].isin(cases)])
    print(f'   Positive videos: {positive_clips} ({positive_clips/len(clips_in_fold)*100:0.1f}%)')
    print(f'   Negative videos: {negative_clips} ({negative_clips/len(clips_in_fold)*100:0.1f}%)')
    print(f'      Annotated frames: {annotated_frames}\n')