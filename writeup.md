#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./center.jpg "Center lane driving"
[image3]: ./recovery_start.jpg "Recovery Start Image"
[image4]: ./recovery_end.jpg "Recovery End Image"
[image5]: ./more.jpg "More Data on a section"
[image6]: ./reverse.jpg "Driving in Reverse"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the model creation and training code as a Jupyter notebook
* model.py containing a copy of th eabove Jupyter notebook
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 - A video of the car driving in autonomous mode in track 1
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.ipynb/model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model builds on the well-known VGG19 network used for image recognition. The convolution layers of VGG19 are retained and new fully connected layers are added to perform feature extraction. When deciding on the size of the fully connected layers, I referred to the [NVIDIA literature](https://arxiv.org/pdf/1604.07316v1.pdf) and maintained the propotion of features coming out of convolution layers to the number of features in the fully connected layers.
The non-linear activation function used is RELU. Batch normalization is used such that the variance of the non-linear layers is equal to the variance of the input.

The model accepts 
- images of size `224 X 64`
- The data needs to be normalized and zero-centered before inputting to the model.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Driving on the track in the reverse direction was also done in certain sections to reduce the left turn bias.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to identify key features in images and then calculate steering angle based on these features. Convolutional layers are used for image feature extraction. Instead of designing my own convolutional layers, I used the well-known VGG19 network. Feature extraction is performed by adding custon fully connected layers which provide the predicted steering angle measurement. Weight initialization is performed using the Xavier/Glorot method. The error function used is MSE (Measn Squared Error) to measure the error between the actual steerign angle and the predicted steering angle.

The model was validated against a validation set (30% of the original data set) to verify that there was no over-fitting to the training data.

Once there was a model with low mean squared error on both training and validation sets, the following steps were done iteratively:

1. Run the simulator with the model in autonomous model. Identify sections where the car is verring of-course and collect more data on good driving behavior for those sections
2. Re-train the model with new data (weights from previous runs were re-used)


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network (VGG19) with the following layers. and layer sizes.
Here is the model summary:

| Layer (type)                 | Output Shape          | Param #   | Connected to                |             
| ---------------------------- | --------------------- | --------- | --------------------------- |
| input_1 (InputLayer)         | (None, 64, 224, 3)    | 0         |                             |      
| block1_conv1 (Convolution2D) | (None, 64, 224, 64)   | 1792      |  input_1[0][0]              |      
| block1_conv2 (Convolution2D) |  (None, 64, 224, 64)  | 36928     |  block1_conv1[0][0]         |      
| block1_pool (MaxPooling2D)   |  (None, 32, 112, 64)  | 0         |  block1_conv2[0][0]         |      
| block2_conv1 (Convolution2D) |  (None, 32, 112, 128) | 73856     |  block1_pool[0][0]          |      
| block2_conv2 (Convolution2D) |  (None, 32, 112, 128) | 147584    |  block2_conv1[0][0]         |      
| block2_pool (MaxPooling2D)   |  (None, 16, 56, 128)  | 0         |  block2_conv2[0][0]         |      
| block3_conv1 (Convolution2D) |  (None, 16, 56, 256)  | 295168    |  block2_pool[0][0]          |      
| block3_conv2 (Convolution2D) |  (None, 16, 56, 256)  | 590080    |  block3_conv1[0][0]         |      
| block3_conv3 (Convolution2D) |  (None, 16, 56, 256)  | 590080    |  block3_conv2[0][0]         |      
| block3_conv4 (Convolution2D) |  (None, 16, 56, 256)  |  590080   |   block3_conv3[0][0]        |       
| block3_pool (MaxPooling2D)   |  (None, 8, 28, 256)   |  0        |   block3_conv4[0][0]        |       
| block4_conv1 (Convolution2D) |  (None, 8, 28, 512)   | 1180160   |  block3_pool[0][0]          |      
| block4_conv2 (Convolution2D) |  (None, 8, 28, 512)   | 2359808   |  block4_conv1[0][0]         |      
| block4_conv3 (Convolution2D) |  (None, 8, 28, 512)   | 2359808   |  block4_conv2[0][0]         |      
| block4_conv4 (Convolution2D) |  (None, 8, 28, 512)   | 2359808   |  block4_conv3[0][0]         |      
| block4_pool (MaxPooling2D)   |  (None, 4, 14, 512)   | 0         |  block4_conv4[0][0]         |      
| block5_conv1 (Convolution2D) |  (None, 4, 14, 512)   | 2359808   |  block4_pool[0][0]          |      
| block5_conv2 (Convolution2D) |  (None, 4, 14, 512)   | 2359808   |  block5_conv1[0][0]         |      
| block5_conv3 (Convolution2D) |  (None, 4, 14, 512)   | 2359808   |  block5_conv2[0][0]         |      
| block5_conv4 (Convolution2D) |  (None, 4, 14, 512)   | 2359808   |  block5_conv3[0][0]         |      
| block5_pool (MaxPooling2D)   |  (None, 2, 7, 512)    | 0         |  block5_conv4[0][0]         |      
| custom_flatten (Flatten)     |  (None, 7168)         | 0         |  block5_pool[0][0]          |      
| custom_fc1 (Dense)           |  (None, 512)          | 3670528   |  custom_flatten[0][0]       |     
| batchnormalization_1         |  (None, 512)          | 2048      |  custom_fc1[0][0]           |      
| activation_1 (Activation)    |  (None, 512)          | 0         |  batchnormalization_1[0][0] |     
| dropout_1 (Dropout)          |  (None, 512)          | 0         |  activation_1[0][0]         |      
| custom_fc2 (Dense)           |  (None, 64)           | 32832     |  dropout_1[0][0]            |      
| batchnormalization_2         |  (None, 64)           | 256       |  custom_fc2[0][0]           |      
| activation_2 (Activation)    |  (None, 64)           | 0         |  batchnormalization_2[0][0] |      
| dropout_2 (Dropout)          |  (None, 64)           | 0         |  activation_2[0][0]         |      
| custom_fc3 (Dense)           |  (None, 10)           | 650       |  dropout_2[0][0]            |      
| batchnormalization_3         |  (None, 10)           | 40        |  custom_fc3[0][0]           |      
| activation_3 (Activation)    |  (None, 10)           | 0         |  batchnormalization_3[0][0] |      
| custom_predictions (Dense)   |  (None, 1)            | 11        |  activation_3[0][0]         |      

Total params: 23,730,749  
Trainable params: 3,705,193  
Non-trainable params: 20,025,556  

Here is a visualization of the architecture

![Model][image1]

#### 3. Creation of the Training Set & Training Process

In addition to the data set provided in the project, good driving behavior was captured by ecording two laps on track one using center lane driving. 
Here is an example image of center lane driving:

![Center lane driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to to move towards center from the track edges. For example:

![Recovery Start][image3]
![Recovery End][image4]

As mentioned previously, turns where the model was having difficulty were recorded to provide more data on these turns. For example:  
![Section with more data][image5]

Driving on the track in the reverse direction was also done in certain sections to reduce the left turn bias. For example:  
![Driving in reverse][image6]

To combat the bias of driving straight in the simulator most of the time, about half of the data with 0 steering angle measurement was removed. Stratification was used in train/validation split so that the zero measurements did not dominate either of the data sets. 

The images from left and right cameras were used with a correction factor added/subtracted to the corresponding center camera measurements. 

The images are cropped to remove unwanted features and resized to the following size: `224 X 64`

During training time, the following augmentations were done at random:
1. Horizontal flipping to reduce the left turn bias in the data collected
2. Contrast stretching to improve the contrast of the images.
3. Gaussian blurring to enable the model to train with blurred features.

__Data statistics:__  
Number of images: 44662  
Training samples: 31263  
Validation samples: 13399 (30%)  

A python generator was used for memory efficiency. At at the begining of every epoch, the data is shuffled at random and batches of the desired size are yielded.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The early stopping callback of keras was used to stop training once the loss stops to reduce by a magnitude of `1e-4`. I used an adam optimizer so that manually training the learning rate wasn't necessary.
