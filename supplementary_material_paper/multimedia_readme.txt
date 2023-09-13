DESCRIPTION:
The multimedia material consists of one file, multimedia.pptx,
which contains all lung ultrasound videos (in .gif format) from one patient in the test set with overlayed model predictions.
The ensemble of five EfficientNet-B0 with U-Net networks was used for predicting the landmark detection heatmap of the B-line origin locations.
The threshold for the landmark detection heatmap was set to 0.5, which is similar to how the detections were counted.

Additional information was added to each video:  
- In the top left corner, the average number of detections per frame for the entire video is reported.
- In the top right corner, the upper symbol indicates the expert video label (+: Presence of B-lines, -: Absence of B-lines),
	the lower symbol indicates the model prediction based on the average number of detections per frame (+: Predicted presence of B-lines, -: Predicted absence of B-lines).
- At the bottom, the timeline of the video is shown. The videos loop indefinitely and are played at half the original frame rate.

SIZE: 
The total size of the file is 28.1MB

PLAYER INFORMATION: 
Microsoft PowerPoint

PACKING LIST: 
multimedia.pptx

CONTACT INFORMATION:
Tina Kapur, PhD
Brigham and Women's Hospital, Harvard Medical School
Boston, Massachusetts, USA
E-mail: tkapur@bwh.harvard.edu
