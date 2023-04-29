import sys
import argparse
import logging
import os.path

import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.metrics
import joblib
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm 
import PIL
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import random

def get_data(filename):
    data = pd.read_csv(filename)
    return data

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
TRAIN_SIZE = 20
X_train = np.zeros((TRAIN_SIZE,IMG_WIDTH,IMG_HEIGHT,3), dtype=np.uint8)
Y_train = np.zeros((TRAIN_SIZE,IMG_WIDTH,IMG_HEIGHT,1), dtype=bool)
df = get_data("dog.csv")
for index, row in df.iterrows(): 
    if index >= TRAIN_SIZE:
        break
    print(index) 
    img = tf.io.read_file(row["images"])
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize_with_pad(img,IMG_WIDTH,IMG_HEIGHT)
    img = img.numpy()
    X_train[index] = img 
    mask = tf.io.read_file(row["labels"])
    mask = tf.io.decode_jpeg(mask)
    mask = tf.image.resize_with_pad(mask,IMG_WIDTH,IMG_HEIGHT)
    mask = tf.cast(mask, dtype=tf.bool)
    mask = mask.numpy()
    Y_train[index] = mask

image_x = random.randint(0, TRAIN_SIZE-1)
imsave("test.jpg",X_train[image_x])
imsave("test2.jpg",Y_train[image_x])