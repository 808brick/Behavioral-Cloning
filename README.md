# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

![ Simulation Autonomous Mode](images/car_driving.gif)

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. A supplementary `model.ipynb` and `model.html` is included to view the code in model.py in a easier to read, notbook format.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

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

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was mainly collected from Track 1 of the simulation. Over 24,000 images were recorded from the simulation while driving it in manual mode. This include images from the center, left, and right cameras. The angle of the steering wheel was also recorded with these images and used as the "labels" for the neural network to define what steering angle the car should be at when it sees particular characteristics in an image.

Training data was modified to flip and translate images and steering angles. This lead to more diversified data since the track only consists of left turns. Recording data while driving around the track in the opposite direction also helped diversify the data. More info on generated data can be found below. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
