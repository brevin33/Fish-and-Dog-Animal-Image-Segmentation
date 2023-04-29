#!/usr/bin/env python3
# python3 cnn_classification.py -t TMNIST_Data-train-train.csv -T TMNIST_Data-train-test.csv -l labels -m best99-model.joblib

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
import random
from cnn_common import *
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import random

################################################################
#
# CNN functions
#

hyperParams = {"conLayers": [3,4],
                "conLayersSize": [3,3,2,2,2],
                "conLayersFilterNumber": [32,32,20.16,20,20],
                "conLayersKernal": [2,3,3,3],
                "dropoutAmount": [0.01,0.1,0.2,0.3,0.1,0.1,0.1],
              }

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
TRAIN_SIZE = 180

def create_model(my_args, input_shape):
    #Build the model
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    layer = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    downScalingLayer = []

    #downscaling layers
    layer = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(layer)
    for i in range(random.choice(hyperParams["conLayers"])):
        layerDepthMuliplier = pow(2,i)
        for j in range(random.choice(hyperParams["conLayersSize"])-1):
            kernal = random.choice(hyperParams["conLayersKernal"])
            layer = tf.keras.layers.Conv2D(random.choice(hyperParams["conLayersFilterNumber"])*layerDepthMuliplier, (kernal, kernal), activation='relu', kernel_initializer='he_normal', padding='same')(layer)
        layer = tf.keras.layers.Dropout(random.choice(hyperParams["dropoutAmount"]))(layer)
        kernal = random.choice(hyperParams["conLayersKernal"])
        layer = tf.keras.layers.Conv2D(random.choice(hyperParams["conLayersFilterNumber"])*layerDepthMuliplier, (kernal, kernal), activation='relu', kernel_initializer='he_normal', padding='same')(layer)
        downScalingLayer.append(layer)
        layer = tf.keras.layers.MaxPooling2D((2, 2))(layer)

    #lowest layer
    for i in range(random.choice(hyperParams["conLayersSize"])-1):
        kernal = random.choice(hyperParams["conLayersKernal"])
        layer = tf.keras.layers.Conv2D(random.choice(hyperParams["conLayersFilterNumber"])*layerDepthMuliplier, (kernal, kernal), activation='relu', kernel_initializer='he_normal', padding='same')(layer)
    layer = tf.keras.layers.Dropout(random.choice(hyperParams["dropoutAmount"]))(layer)
    kernal = random.choice(hyperParams["conLayersKernal"])
    layer = tf.keras.layers.Conv2D(random.choice(hyperParams["conLayersFilterNumber"])*layerDepthMuliplier, (kernal, kernal), activation='relu', kernel_initializer='he_normal', padding='same')(layer)

    #upscaling layers
    for i in range(len(downScalingLayer)):
        layerDepthMuliplier = pow(2,len(downScalingLayer)-i-1)
        layer = tf.keras.layers.Conv2DTranspose(random.choice(hyperParams["conLayersFilterNumber"])*layerDepthMuliplier, (2, 2), strides=(2, 2), padding='same')(layer)
        layer = tf.keras.layers.concatenate([downScalingLayer[len(downScalingLayer)-i-1], layer])
        for i in range(random.choice(hyperParams["conLayersSize"])-1):
            kernal = random.choice(hyperParams["conLayersKernal"])
            layer = tf.keras.layers.Conv2D(random.choice(hyperParams["conLayersFilterNumber"])*layerDepthMuliplier, (kernal, kernal), activation='relu', kernel_initializer='he_normal', padding='same')(layer)
        layer = tf.keras.layers.Dropout(random.choice(hyperParams["dropoutAmount"]))(layer)
        kernal = random.choice(hyperParams["conLayersKernal"])
        layer = tf.keras.layers.Conv2D(random.choice(hyperParams["conLayersFilterNumber"])*layerDepthMuliplier, (kernal, kernal), activation='relu', kernel_initializer='he_normal', padding='same')(layer)
   
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(layer)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    adam = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=adam, loss='binary_focal_crossentropy', metrics=['accuracy'])
    return model

def do_cnn_fit(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, Y = load_data(my_args, train_file)
    #bestModel = joblib.load("troutModel.joblib")
    #bestAccuracy = getAccuracy(my_args,bestModel)
    bestModel = None
    bestAccuracy = 0
    for i in range(1):
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        #model = create_model(my_args, X.shape[1:])
        model = joblib.load("dogModel2.joblib")
        #model.summary()
        model.fit(X, Y, epochs=50, verbose=1, callbacks=[early_stopping], validation_split=my_args.validation_split)
        accuracy = getAccuracy(my_args,model)
        print(accuracy)
        if accuracy >= bestAccuracy:
            bestAccuracy = accuracy
            bestModel = model
            joblib.dump(bestModel, "dogModel2.joblib")
    joblib.dump(bestModel, "dogModel2.joblib")
    print(bestAccuracy)
    return
#
# CNN functions
#
################################################################

################################################################
#
# Evaluate existing models functions
#
def sklearn_metric(y, yhat):
    cm = sklearn.metrics.confusion_matrix(y, yhat)
    ###
    header = "+"
    for col in range(cm.shape[1]):
        header += "-----+"
    rows = [header]
    for row in range(cm.shape[0]):
        row_str = "|"
        for col in range(cm.shape[1]):
            row_str += "{:4d} |".format(cm[row][col])
        rows.append(row_str)
    footer = header
    rows.append(footer)
    table = "\n".join(rows)
    print(table)
    print()
    ###
    if cm.shape[0] == 2:
        precision = sklearn.metrics.precision_score(y, yhat)
        recall = sklearn.metrics.recall_score(y, yhat)
        f1 = sklearn.metrics.f1_score(y, yhat)
        print("precision: {}".format(precision))
        print("recall: {}".format(recall))
        print("f1: {}".format(f1))
    else:
        report = sklearn.metrics.classification_report(y, yhat)
        print(report)
    return

def getAccuracy(my_args,model):
    train_file = my_args.train_file
    test_file = get_test_filename(my_args.test_file, train_file)
    basename = get_basename(train_file)
    X_test, y_test = load_data(my_args, test_file)
    #X_test = pipeline.transform(X_test) # .todense()
    #X_test = np.reshape(X_test,[X_test.shape[0],28,28,1])
    yhat_test = np.array(np.where(model.predict(X_test) > 0.5, 1, 0).astype(bool))
    image_x = random.randint(0, 30)
    imsave("dog.jpg",X_test[image_x])
    imsave("dogTestReal.jpg",y_test[image_x])
    imsave("dogTestPredicted.jpg",yhat_test[image_x])
    y_test = np.array(y_test).ravel()
    yhat_test = yhat_test.ravel()
    accuracy_score = sklearn.metrics.accuracy_score(y_test, yhat_test)
    return accuracy_score

def show_score(my_args):
    model = joblib.load("dogModel2.joblib")
    model.summary()
    print("test or validation acceracy: " + str(getAccuracy(my_args,model)))
    image_x = random.randint(0, TRAIN_SIZE-1)



def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Image Classification with CNN')
    parser.add_argument('action', default='cnn-fit',
                        choices=[ "cnn-fit", "score" ], 
                        nargs='?', help="desired action")

    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file with test data (default is constructed from train file name)")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")

    #
    # Pipeline configuration
    #
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',         '-l', default="label",   type=str,   help="column name for label")
    parser.add_argument('--use-polynomial-features', '-p', default=0,         type=int,   help="degree of polynomial features.  0 = don't use (default=0)")
    parser.add_argument('--use-scaler',    '-s', default=0,         type=int,   help="0 = don't use scaler, 1 = do use scaler (default=0)")
    parser.add_argument('--categorical-missing-strategy', default="",   type=str,   help="strategy for missing categorical information")
    parser.add_argument('--numerical-missing-strategy', default="",   type=str,   help="strategy for missing numerical information")
    parser.add_argument('--print-preprocessed-data', default=0,         type=int,   help="0 = don't do the debugging print, 1 = do print (default=0)")

    
    parser.add_argument('--shuffle',                       action='store_true',  help="Shuffle data when loading.")
    parser.add_argument('--no-shuffle',    dest="shuffle", action='store_false', help="Do not shuffle data when loading.")
    parser.set_defaults(shuffle=True)

    #
    # hyper parameters
    #
    parser.add_argument('--validation-split', default=0.1,         type=float,   help="validation split fraction (default=0.1)")

    # debugging/observations
    parser.add_argument('--show-test',     '-S', default=0,         type=int,   help="0 = don't show test loss, 1 = do show test loss (default=0)")


    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    
    return my_args


def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'cnn-fit':
        do_cnn_fit(my_args)
    elif my_args.action == 'score':
        show_score(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))

    return

if __name__ == "__main__":
    main(sys.argv)

    
