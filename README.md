# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

All code, including `model.py` run without any errors. Assertion statements are also put into place throuout the code to ensure the datasets and labels match up. An example of this includes

```python
assert len(all_images) == len(all_steering_angles), "Number of images and steering angles does not match (Combine Dataset)"
```

which will stop the script way before the keras model is even created if anything goes wrong with the data.

![ Simulation Autonomous Mode](images/sim-auto.gif)

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. A supplementary `model.ipynb` and `model.html` is included to view the code in model.py in a easier to read, notbook format.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The actual Keras model used for this projects can be found on lines 190-209 of model.py. The network flow can be viewed on the table below from the output of `model.summary()`. This model was partially based off of [Nvidia's](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) "End to End Deep Learning For Self Driving Cars" model with some slight modifications. The model starts with a normalization layer to normalize the images as they come in. Rather than taking the mean, a simple lamda function is put into place `lambda x: x/127.5 - 1.0` in a lambda Keras layer. This is followed by a cropping layer which removes 50 pixels from the top of the image to remove the environment scenery, and 20 pixels from the bottom of the image to remove the hood of the car. the This is followed by 4 convolutional layers, each followed by Relu activation layers. The results are then flattened and put through a dropout layer. This is followed by three fully connected layer, with the first fully connected layer having an another activation and dropout layer after it. All these layers were then trained by a Adam optimizer. 

The highlight of this arcitecture is that four convolutional layers are used (with corresponding activations functions). The added dropout layers also help prevent overfitting of the model, improving it's performance when working with live data from running the simulation in autonomous mode.  


|Layer (type)        |         Output Shape       |       Param #  |
| ------------------ | -------------------------- | -------------- |
|lambda_1 (Lambda)    |        (None, 160, 320, 3)  |     0        |
|cropping2d_1 (Cropping2D)  |  (None, 90, 320, 3)    |    0         |
|conv2d_1 (Conv2D)     |       (None, 23, 80, 32)    |    6176      |
|activation_1 (Activation)  |  (None, 23, 80, 32)    |    0         |
|conv2d_2 (Conv2D)      |      (None, 6, 20, 64)       |  131136   |
|activation_2 (Activation)   | (None, 6, 20, 64)       |  0        |
|conv2d_3 (Conv2D)       |     (None, 3, 10, 128)        |131200   |
|activation_3 (Activation)    |(None, 3, 10, 128)        |0        |
|conv2d_4 (Conv2D)       |     (None, 3, 10, 128)    |    65664     |
|activation_4 (Activation) |   (None, 3, 10, 128)    |    0         |
|flatten_1 (Flatten)     |     (None, 3840)       |       0         |
|dropout_1 (Dropout)     |     (None, 3840)        |      0         |
|dense_1 (Dense)         |     (None, 128)        |       491648    |
|activation_5 (Activation)|    (None, 128)          |     0        |
|dropout_2 (Dropout)      |    (None, 128)           |    0        |
|dense_2 (Dense)          |    (None, 128)          |     16512    |
|dense_3 (Dense)          |    (None, 1)           |      129       |

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Also the image generator helps prevent overfitting by randomly flipping images horizontally along with flipping the corresponding steering angle, and also randomly translating them. This way the network is not just looking at the exact pixel for pixel images every time. This is addressed in the code with the `randomize_translation()` and `randomize_horizontal_flip()` functions.

Having a large data set (I had over 24,000 images) also helped prevent overfitting. The number of epochs was kept relatively low, with only 7 epochs, which also contributes to the prevention of overfitting.

#### 3. Model parameter tuning

The parameters used to train the optimizer can be found on lines 26-27 of `model.py`. The batch size was set to 128, with 7 epochs, and a default Adam optimizer learning rate of 0.001. 

#### 4. Appropriate training data

Training data was mainly collected from Track 1 of the simulation. Over 24,000 images were recorded from the simulation while driving it in manual mode. This include images from the center, left, and right cameras. The angle of the steering wheel was also recorded with these images and used as the "labels" for the neural network to define what steering angle the car should be at when it sees particular characteristics in an image.

Training data was modified to flip and translate images and steering angles. This lead to more diversified data since the track only consists of left turns. Recording data while driving around the track in the opposite direction also helped diversify the data. More info on generated data can be found below. 

### Final Results And Implementation

#### 1. Solution Design Approach

Data Collection was a crucial part of this project. Over 24,000 images of data was taken driving around in the simulation while in manual mode. This included doing several laps around the course, as well as recording more data with the car flipped around facing the other direction. Flippin the car around and driving it is important since the track mainly consists of left turns, which would lead to a bias in the model to prefer left turning more than anything else. The horizontal flipping of images and steering angles also helped address this issue. 

Data was saved to the `/opt/` directory of the Udacity workspace, but since this directory gets wiped after every reboot, data was stored on my GitHub repository: https://github.com/808brick/Behavioral-Cloning under the data folder. Data was then cloned to the workspace everytime it was booted, the directory which the `driving_log.csv` pointed to had to be changed to point to the full directory of the images for use with my `model.py` script. Thus, helper scripts were created, specifically `modify_file_paths_csv.py` which renamed all the image paths in the csv file to point to the directory specified. 

When running `model.py`, the model will begin training on the datset and will write 3 files: `model.h5`, `model.json`, `model_simple_save.h5`. `model.h5` and `model.json` are reserved in the case you wanted to retrain with the same model with the saved weights. `model_simple_save.h5` is the actual model used when running the simulation in autonomous mode with `drive.py`. 

The final step was to run the simulator to see how well the car was driving around track one. The model performed excelently, with the car not falling off the track at all, nor even showing a hint of almost doing so. The results from my trial run can be viewed in `video.mp4`. I could not get the supplied `video.py` script to work with the simulation, so I reverted to recording the trial run with a screen recorder and then increased the speed of the video 4x for brevity and to decrease the size of the video using the linux `mencoder` terminal command. 


#### 2. Training Set & Training Process

Part of the data collected was split into a validation set. This was done using the `train_test_split()` function from the sklearn module. The function was set to split 20% of the data to be reserved for validation. This can be found on line 92 of `model.py`. Assertion statements follow this operation to ensure that the dataset and labels are split properly.  

This dataset was then put into data generators that split the images into batches and shuffled the data. The `train_generator_yield()` function is used to feed in the training images, while occasionally horizontally flipping the image (and steering angle) and performing a randomized translation on the image so that the model does not become overfitted by seeing the same exact image with every epoch. The validation images were put into the `valid_generator_yield()` function, which just shuffles the data and feeds it as a validation batch. These functions can be found in `model.py` on lines 129-180. 

With that, the model would be trained. Past 7 epochs, very little change in the accuracy displayed by Keras was seen, leading to the use of only 7 epochs. The resulting trained model worked excellently when running the simulation in autonomous mode.  
