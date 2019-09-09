##############################
#	Imports
##############################

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from collections import deque
import keras
import keras.layers as layers
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Activation, Dropout,Convolution2D,MaxPooling2D,Flatten,Lambda

# import os 
# dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)

##############################
#	Training Parameters
##############################

EPOCHS = 7
BATCH_SIZE = 128


##############################
#	Load In Data
##############################

csv_filename = '/opt/Behavioral-Cloning/data/driving_log.csv'

# CSV Data = [Center, left, right, steering, throttle, brake, speed]

left_images = deque()
right_images = deque()
center_images = deque()
steering_angles = deque()

with open(csv_filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    print("Reading in CSV data ...", end="")
    for row in csv_reader:
        center_image_path = row[0].replace(" ", "")
        left_image_path = row[1].replace(" ", "")
        right_image_path = row[2].replace(" ", "")
        
        center_images.append(plt.imread(center_image_path))
        left_images.append(plt.imread(left_image_path))
        right_images.append(plt.imread(right_image_path))
        
        steering_angles.append(row[3])
    print(" Done")

assert len(steering_angles) == len(left_images), "Number of images and steering angles does not match"


##############################
#	Combine Datasets
##############################

all_images = deque()
all_steering_angles = deque()

print("Combining Datasets ... ", end="")

for i in range(len(steering_angles)):
    
    all_images.append(center_images[i])
    all_images.append(left_images[i])
    all_images.append(right_images[i])
    
    for j in range(3):
        all_steering_angles.append(float(steering_angles[i]))
        

print("Done")

assert len(all_images) == len(all_steering_angles), "Number of images and steering angles does not match (Combine Dataset)"



##############################
#	Validation Split
##############################

print("Creating Validation Dataset ... ", end="")

train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(all_images, all_steering_angles, test_size=0.2)

assert len(train_dataset) == len(train_labels), "Number of images and steering angles does not match (Train Split)"
assert len(valid_dataset) == len(valid_labels), "Number of images and steering angles does not match (Valid Split)"

print("Done")

##############################
#	Image Data Generator
##############################

def randomize_translation(image):
    rows,cols,depth = image.shape

    # allow translation up to px pixels in x and y directions
    max_translation_pixels = int(rows/12)
    x_dist,y_dist = np.random.randint(-max_translation_pixels,max_translation_pixels,2)

    M = np.float32([[1,0,x_dist],[0,1,y_dist]])
    dst = cv2.warpAffine(image,M,(cols,rows))

    translated_image = dst[:,:,np.newaxis]

    return translated_image

def randomize_horizontal_flip(image, angle, thresh_angle=0.05):
#     if abs(angle) >= thresh_angle:
            
    #Flip image and steering angle every so often
    if np.random.randint(2) == 1:

        image = cv2.flip(image, 1)
        angle = -angle
        
    return image, angle
    

def train_generator_yield(X_dataset, y_labels, batch_size=64):  
    
    dataset_len = len(y_labels)
    
    while True:
    
        image_batch = deque()
        steering_batch = deque()
        
        X_dataset, y_labels = shuffle(X_dataset, y_labels)
    
        for i in range(batch_size):
            image = X_dataset[i]
            angle = y_labels[i]
            
            image, angle = randomize_horizontal_flip(image, angle)
            image = randomize_translation(image)
            
            image_batch.append(image)
            steering_batch.append(angle)

        
        image_batch = np.squeeze(np.asarray(image_batch))
        steering_batch = np.squeeze(np.asarray(steering_batch))


        yield [image_batch, steering_batch]
        

def valid_generator_yield(X_dataset, y_labels, batch_size=64):  
    
    dataset_len = len(y_labels)
    
    while True:
    
        image_batch = deque()
        steering_batch = deque()
        
        X_dataset, y_labels = shuffle(X_dataset, y_labels)
    
        for i in range(batch_size):
            image = X_dataset[i]
            angle = y_labels[i]
            
            image_batch.append(image)
            steering_batch.append(angle)
        
        image_batch = np.squeeze(np.asarray(image_batch))
        steering_batch = np.squeeze(np.asarray(steering_batch))


        yield [image_batch, steering_batch]



##############################
#	Keras Model
##############################

print("Creating Keras Model: \n\n")

model = keras.Sequential()

model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(160,320,3)))
model.add(layers.Cropping2D(cropping=((50,20), (0,0))))
# model.add(layers.Cropping2D(cropping=((50,20), (0,0)),input_shape=(160,320,3)))
model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
model.add(Activation('relu'))
model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(1))


model.summary()

print("\n\n")



##############################
#	Train Model
##############################

train_generator = train_generator_yield(train_dataset, train_labels, BATCH_SIZE)
valid_generator = train_generator_yield(valid_dataset, valid_labels, BATCH_SIZE)

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

model.fit_generator(train_generator, nb_epoch=EPOCHS,\
                    steps_per_epoch=int(len(train_labels)/BATCH_SIZE), verbose=1, \
                    validation_data=valid_generator, validation_steps=len(valid_labels)/BATCH_SIZE)


# model.save("model.h5")

# Save model data
model.save_weights('./model.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)

    
model.save("./model_simple_save.h5")
    
print("Model Saved")