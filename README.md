# CNN-project
This repository contains a small CNN hand joint recognition task that I prepared for Applied AI study group at Inzva.
Main aim is will be "keypoint detection", here images will be hands and keypoints will be joint locations on hands.

Dataset can be reached from:
  http://www.rovit.ua.es/dataset/mhpdataset/
or 
  https://www.kaggle.com/kmader/multiview-hand-pose
  
About dataset:

The dataset consist of sequences of hand photos.
Inside each sequence there are 4 color images, bounding boxes for hands in images, 3D points of joints.

-For easier application of a CNN network we need to project 3D coordinates into 2D coordinates. The code for this conversion is provided by publishers under utils.
I debugged the provided script, you can find it under utils named as 'generate2Dpoints.py'.

-We also need to generate bounding boxes for the hands in images. Script for generating bounding boxes also provided and you can find debugged version
under utils named as 'generateBBoxes.py'

## Preprocessing
There are 21 data folders in dataset. I used 11 of them with reshaping the images since using all the data did not improve results significantly but result in longer runtimes in preprocessing.
  
  * From bounding box coordinates, I cropped hands in images for better performance in model.
  
  * To prevent long runtime,I resized and turn the cropped images into arrays of shape (64,64,3)
  
  * After cropping and reshaping, we need to correct the joint coordinates to use with reshaped images.
  
  * I pickled the data as a dictionary: mydict = {'images':images,'joints':joints,'bounding':bounding}
  
  ## Model
 
 Model script includes also my Google Drive connection steps since I run the model in Google Colab environment with GPU support.
 I used Keras library from Tensorflow to build the model.
 
 I provided two models and 3 deployments to see improvements from base model to final model:

 * **Base Model**: 2 Conv layers with 'valid' padding and LeakyReLu with alpha=0.1 between conv layers, 1 MaxPooling layer, 1 Dropout layer
                   _Training_: 50 epoches, batch size=64, optimizer=adam
 
 * **Last Model**: 4 Conv layers with 'valid' padding, Dropout, LeakyRelu and BatchNormalization between layers, 2 MaxPooling layer, 1 final Dropout layer
                   _Training_: 50 epoches, batch size=64, optimizer=adam
                   
 * **Final**: Same model with  _Training_: 30 epoches, batch size=32, optimizer=adam
 

