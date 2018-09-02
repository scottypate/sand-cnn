import tensorflow as tf
import cv2
import numpy as np
import os
import pandas as pd
from Net import Net

net = Net()

app_dir = os.path.dirname(os.path.abspath(__file__))

train_image_helper = pd.read_csv(filepath_or_buffer=app_dir + '/data/train.csv')

# Load the image data
train_masks = [] 
train_images = []
for item in train_image_helper.id:
    image = cv2.imread(filename=app_dir + '/data/train/images/' + item + '.png', flags=cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(filename=app_dir + '/data/train/masks/' + item + '.png', flags=cv2.IMREAD_GRAYSCALE)
    train_images.append(np.expand_dims(cv2.resize(image, (256, 256)), axis=2))
    train_masks.append(np.expand_dims(cv2.resize(mask, (256, 256)), axis=2))

# Run the training process
net.execute(
    batch_size=100,
    features=train_images,
    labels=train_masks,
    n_epochs=1,
)
