# Human Activity Recognition from Videos

## Problem statement :

**Given a video (any file : .mp4, .avi, .MTS) , the task is to recognize, i.e , classify the activity being performed in the video.**

![Activity classification](https://cse.buffalo.edu/~jcorso/r/actionbank/files/action_bank_montage.png)

## Applications of such a system :

* Elderly & infant care
* Suspicious Activity Recognition
* Industrial manufacturing & assistance

**& many more**

## Dataset used:

10 classes from the UCF-101 dataset : https://www.crcv.ucf.edu/data/UCF101.php

## Libraries used:

```
* Numpy
* OpenCV 
* PyTorch 
```
## Methodologies :

### 1) Using CNN:

Videos can be thought as many images stitched together. Thus we can assume subsequent frames in a video are correlated with respect to their semantic contents. Hence, we can extract images from the videos & then train a CNN pretrained on ImageNet dataset to classify the images extracted from the videos.

**Accuracy achieved using this methodology 91%.**

#### Dataflow diagram :

![DFD-1](https://raw.githubusercontent.com/subhromitra/Video-analytics/master/dfd_1.JPG)

### 2) Using Spatio Temporal Classifer (CNN-LSTM):

Since, videos are temporal sequences thus we may also create a spatio-temporal classifer. I've done this by training an LSTM network on the features given by the CNN from the images of the video.

However, accuracy achieved was only 56%. 

#### Dataflow diagram :

![DFD-2](https://raw.githubusercontent.com/subhromitra/Video-analytics/master/dfd_2.JPG)

**Reasons for low accuracy :**

```
Less amount of data per class
```

## Other ways of doing it:

Other ways of doing it have been beautifully descibed in this blog: **http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review**

I hope to implement some of them in the near future !!!
