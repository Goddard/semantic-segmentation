# Semantic Segmentation
### Introduction
In this project I am using vgg16 model that is pre-trained and the kitti dataset.  I also add fcn8 network demonstrated in the paper "Fully Convolutional Networks For Semantic Segmentation".

This code should make it easy for some one to load a pre-existing model, add layers, train, freeze, optimize, perform a graph transform to 8 bit, and also simple graph, video, and image examples of the inference.

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### To Start

##### Run
Run the following command to run the training and save the initial model and checkpoint for tensorflow:
```
python main.py
```

Run the following command to get extra information for the models and also run tensorflow tools to freeze, optimize, and transform the graphs.  This also creates the dot files and converts them to a image.:
```
python build-model.py
```

Run the following command to look at how the various graphs perform on video and images.:
```
python video.py
```

## Test Environment
All testing was done on my lab system.

OS : Kubuntu 16.04
CPU : AMD Threadripper 1950x
RAM : 32 GB with 15-15-15 latency
TensorFlow Version: 1.4.0-rc1 (compiled from source)
GPU : GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.683
Total GPU Memory: 7.92GiB


## Hyper parameters used
Epochs : 15
Batch Size : 5

# Results
![Cross Entropy Loss](runs/normal/1508797633.1374235.png?raw=true "Cross Entropy Loss")

### Normal Semantic Segmentation Speeds
With this example I do think the speeds seem to suggest that the non-8 bit version was on average 1 second slower.  The video I tested on was about 1700 frames and each frame had inference performed on it and it took about .5 seconds for each frame.

Downscale : 0.002460714997141622

SS Test : 1.6782658910015016

Upscale : 0.001284024998312816

1.0 frames


Downscale : 0.001150920994405169

SS Test : 0.6872002170057385

Upscale : 0.0005270979963825084

2.0 frames


### 8 Bit Video Speeds
After freezing the graph we use a TensorFlow tool called transform_graph to convert the graph and I perform inference on images and a video.  The speed results from the first couple frames are below.

Downscale : 0.002117505995556712

SS Test : 1.8592919129878283

Upscale : 0.001217473007272929

1.0 frames


Downscale : 0.0011298349709250033

SS Test : 1.513521872984711

Upscale : 0.0006270769517868757

2.0 frames

![8 bit Example 1](runs/eight_bit/1508725963.9354362/um_000000.png?raw=true "Example 1")

![8 bit Example 2](runs/eight_bit/1508725963.9354362/um_000006.png?raw=true "Example 2")

![8 bit Example 3](runs/eight_bit/1508725963.9354362/um_000009.png?raw=true "Example 3")

### Problems
Currently for some reason I was unable to decern was why when using some of these Tensorflow tools it would alter graphs beyond use.  Currently the graphs get altered in a way that eats up all memory so I could only create about 10 test images before it would overflow.  With the 8 bit example I was able to do many more images, but still the same problem.  When saving and restoring using Tensorflows Save Model structure and restoring the checkpoint I can perform inference on all my images and run over the video.  Going to keep working on this issue even though it is somewhat unrelated to the end goals of this project.

### Frozen Graph and Examples
![Frozen Example 1](runs/freeze/1508772627.4712257/um_000009.png?raw=true "Example 1")

![Frozen Example 2](runs/freeze/1508772627.4712257/um_000017.png?raw=true "Example 2")

![Frozen Example 3](runs/freeze/1508772627.4712257/um_000090.png?raw=true "Example 3")

![Frozen Graph Example](runs/freeze/graph.dot.png?raw=true "Frozen Graph Example")

### Optimized Graph and Examples
![Optimized Example 1](runs/optimized/1508772893.5949483/um_000009.png?raw=true "Example 1")

![Optimized Example 2](runs/optimized/1508772893.5949483/um_000017.png?raw=true "Example 2")

![Optimized Example 3](runs/optimized/1508772893.5949483/um_000090.png?raw=true "Example 3")

![Optimized Graph Example](runs/optimized/graph.dot.png?raw=true "Optimized Graph Example")

### Eight Bit Graph and Examples
![8 bit Example 1](runs/eight_bit/1508801115.3053203/um_000009.png?raw=true "Example 1")

![8 bit Example 2](runs/eight_bit/1508801115.3053203/um_000017.png?raw=true "Example 2")

![8 bit Example 3](runs/eight_bit/1508801115.3053203/um_000090.png?raw=true "Example 3")

![Optimized Graph Example](runs/eight_bit/graph.dot.png?raw=true "Eight Bit Graph Example")