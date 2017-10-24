8 Bit runtimes
Downscale : 0.002117505995556712
SS Test : 1.8592919129878283
Upscale : 0.001217473007272929
1.0 frames
Downscale : 0.0011298349709250033
SS Test : 1.513521872984711
Upscale : 0.0006270769517868757
2.0 frames
Downscale : 0.0011487309820950031
SS Test : 1.5856113570043817
Upscale : 0.0005612199893221259
3.0 frames
Downscale : 0.0008781710057519376
SS Test : 1.5999070939724334
Upscale : 0.0005967679899185896
4.0 frames
Downscale : 0.0011475890059955418
SS Test : 1.5963401480112225
Upscale : 0.0005361019866541028
5.0 frames
Downscale : 0.0008201590389944613
SS Test : 1.5866409789887257
Upscale : 0.0005890229949727654
6.0 frames
Downscale : 0.0008638330036774278
SS Test : 1.0831055819871835
Upscale : 0.0005624120240099728
7.0 frames
Downscale : 0.0011295850272290409
SS Test : 1.0470612680073828
Upscale : 0.0004745839978568256
8.0 frames
Downscale : 0.0011427589925006032
SS Test : 1.5157521630171686
Upscale : 0.0004632920026779175
9.0 frames
Downscale : 0.0010177289950661361
SS Test : 1.2053278379607946
Upscale : 0.0007577179931104183
10.0 frames
Downscale : 0.0014837050111964345
SS Test : 1.249713953002356
Upscale : 0.0005801570368930697
11.0 frames
Downscale : 0.0018314349581487477
SS Test : 1.3025611389894038
Upscale : 0.0007479590130969882
12.0 frames
Downscale : 0.001161083986517042
SS Test : 1.3303677139920183
Upscale : 0.0005723319482058287
13.0 frames

# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
