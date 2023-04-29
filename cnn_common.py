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
from skimage.io import imread, imshow, imsave
import PIL


IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
TRAIN_SIZE = 180

################################################################
#
# Data/File functions
#
def get_feature_and_label_names(my_args, data):
    label_column = my_args.label
    feature_columns = my_args.features

    if label_column in data.columns:
        label = label_column
    else:
        label = ""

    features = []
    if feature_columns is not None:
        for feature_column in feature_columns:
            if feature_column in data.columns:
                features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column != label:
                features.append(feature_column)

    return features, label

def get_data(filename):
    data = pd.read_csv(filename)
    return data

def load_data(my_args, filename):
    df = get_data(filename)
    X = np.zeros((TRAIN_SIZE,IMG_WIDTH,IMG_HEIGHT,3), dtype=np.uint8)
    Y = np.zeros((TRAIN_SIZE,IMG_WIDTH,IMG_HEIGHT,1), dtype=bool)
    for index, row in df.iterrows(): 
        if index >= TRAIN_SIZE:
            break
        img = tf.io.read_file(row["images"])
        img = tf.io.decode_png(img)
        img = tf.image.resize_with_pad(img,IMG_WIDTH,IMG_HEIGHT)
        img = tf.cast(img, dtype=tf.int8)
        X[index] = img 
        mask = tf.io.read_file(row["labels"])
        mask = tf.io.decode_png(mask)
        mask = tf.image.resize_with_pad(mask,IMG_WIDTH,IMG_HEIGHT)
        mask = tf.cast(mask, dtype=tf.bool)
        mask = mask.numpy()
        Y[index] = mask
    return X, Y

def get_test_filename(test_file, filename):
    if test_file == "":
        basename = get_basename(filename)
        test_file = "{}-test.csv".format(basename)
    return test_file

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))

    stub = "-train"
    if basename[len(basename)-len(stub):] == stub:
        basename = basename[:len(basename)-len(stub)]

    return basename

def tensor_to_png(tensor):
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    image = PIL.Image.fromarray(tensor)
    image.save('name.png', format='PNG')

def tensor_gray_to_png(tensor):
    tensor = tf.image.grayscale_to_rgb(tensor)
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    image = PIL.Image.fromarray(tensor)
    image.save('name.png', format='PNG')

def get_model_filename(model_file, filename):
    if model_file == "":
        basename = get_basename(filename)
        model_file = "{}-model.joblib".format(basename)
    return model_file
#
# Data/File functions
#
################################################################


################################################################
#
# Pipeline classes and functions
#
class PipelineNoop(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Just a placeholder with no actions on the data.
    """
    
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

class Printer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Pipeline member to display the data at the current stage of the transformation.
    """
    
    def __init__(self, title):
        self.title = title
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("{}::type(X)".format(self.title), type(X))
        print("{}::X.shape".format(self.title), X.shape)
        if not isinstance(X, pd.DataFrame):
            print("{}::X[0]".format(self.title), X[0])
        print("{}::X".format(self.title), X)
        return X

class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, do_predictors=True, do_numerical=True):
        self.mCategoricalPredictors = []
        self.mNumericalPredictors = [ "{}".format(i) for i in range(1,785) ]
        self.mLabels = ["labels"]
        self.do_numerical = do_numerical
        self.do_predictors = do_predictors
        
        if do_predictors:
            if do_numerical:
                self.mAttributes = self.mNumericalPredictors
            else:
                self.mAttributes = self.mCategoricalPredictors                
        else:
            self.mAttributes = self.mLabels
            
        return

    def getCategoricalPredictors(self):
        return self.mCategoricalPredictors

    def getNumericalPredictors(self):
        return self.mNumericalPredictors

    def fit( self, X, y=None ):
        # no fit necessary
        return self

    def transform( self, X, y=None ):
        # only keep columns selected
        values = X[self.mAttributes]
        return values

def make_numerical_feature_pipeline(my_args):
    items = []
    
    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))
    if my_args.numerical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.numerical_missing_strategy)))

    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        items.append(("scaler", sklearn.preprocessing.StandardScaler()))

    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Numerical Preprocessing")))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline

def make_categorical_feature_pipeline(my_args):
    items = []
    
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))

    if my_args.categorical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.categorical_missing_strategy)))
    ###
    ### sklearn's decision tree classifier requires all input features to be numerical
    ### one hot encoding accomplishes this.
    ###
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))

    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Categorial Preprocessing")))

    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_feature_pipeline(my_args):
    """
    Numerical features and categorical features are usually preprocessed
    differently. We split them out here, preprocess them, then merge
    the preprocessed features into one group again.
    """
    items = []
    dfs = DataFrameSelector()
    if len(dfs.getNumericalPredictors()) > 0:
        items.append(("numerical", make_numerical_feature_pipeline(my_args)))
    if len(dfs.getCategoricalPredictors()) > 0:
        items.append(("categorical", make_categorical_feature_pipeline(my_args)))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline

def make_pseudo_fit_pipeline(my_args):
    """
    Pipeline that can be used for prepreocessing of data, but
    the model is blank because the model is a Tensorflow network.
    """
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Final Preprocessing")))
    items.append(("model", None))
    return sklearn.pipeline.Pipeline(items)
#
# Pipeline functions
#
################################################################

