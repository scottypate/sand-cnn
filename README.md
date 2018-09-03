# Image Segmentation with TensorFlow

Convolutional Neural Network built using OpenCV and Tensorflow for [this](https://www.kaggle.com/c/tgs-salt-identification-challenge) Kaggle competition. 

This is an implementation of "U-Net" described in [this](https://arxiv.org/abs/1505.04597) paper. The name "U-Net" comes from the "U" shape of the convolution and upsampling layers in the network. The "left" side of the "U" are convolution layers in which image shrinkage occurs as seen in the .gif below:

![](https://gph.is/1Hc5NVr)


### Net.py
The `Net` class of this application is responsible for the structure and the flow of data through the neural network. This class allows training of a neural network with the `train` method and access to the saved results of the training session with the `predict` method.

###Convolution.py
The `Convolution` class of this application is responsible for the operations that occur on the data within the structure defined in the `Net.py` class. This includes the definition of activation functions and the 