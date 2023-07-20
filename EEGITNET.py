import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, SpatialDropout1D, SpatialDropout2D, BatchNormalization
from tensorflow.keras.layers import Flatten, InputSpec, Layer, Concatenate, AveragePooling2D, MaxPooling2D, Reshape, Permute
from tensorflow.keras.layers import Conv2D, LSTM , SeparableConv2D, DepthwiseConv2D, ConvLSTM2D, LayerNormalization
from tensorflow.keras.layers import TimeDistributed, Lambda, AveragePooling1D, GRU, Attention, Dot, Add, Conv1D, Multiply
from tensorflow.keras.constraints import max_norm, unit_norm 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.utils import plot_model

import random
import time
import numpy as np
import pandas as pd
import math
import mne
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import scipy
from scipy import stats, fft, signal
import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from timeit import default_timer as timer
from tabulate import tabulate
import seaborn as sns

def SQ (in_tensor):
    return tf.squeeze(in_tensor, axis=1)

def Select(in_tensor):
    return in_tensor[:,-1,:]

def Network(Chans, Samples, out_type = 'single', num_class = 4):
    n_ff = [2,4,8]    # Number of frequency filters for each inception module of EEG-ITNet
    n_sf = [1,1,1]    # Number of spatial filters in each frequency sub-band of EEG-ITNet
    out_class = num_class
    Input_block = Input(shape = (Chans, Samples, 1))
    #========================================================================================   
    # EEG-ITNet
    #========================================================================================  
    drop_rate = 0.2

    block1 = Conv2D(n_ff[0], (1, 16), use_bias = False, activation = 'linear', padding='same',
                    name = 'Spectral_filter_1')(Input_block)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias = False, padding='valid', depth_multiplier = n_sf[0], activation = 'linear',
                                depthwise_constraint = tf.keras.constraints.MaxNorm(max_value=1),
                            name = 'Spatial_filter_1')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)

    #================================

    block2 = Conv2D(n_ff[1], (1, 32), use_bias = False, activation = 'linear', padding='same',
                    name = 'Spectral_filter_2')(Input_block)
    block2 = BatchNormalization()(block2)
    block2 = DepthwiseConv2D((Chans, 1), use_bias = False, padding='valid', depth_multiplier = n_sf[1], activation = 'linear',
                                depthwise_constraint = tf.keras.constraints.MaxNorm(max_value=1),
                            name = 'Spatial_filter_2')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)

    #================================

    block3 = Conv2D(n_ff[2], (1, 64), use_bias = False, activation = 'linear', padding='same',
                    name = 'Spectral_filter_3')(Input_block)
    block3 = BatchNormalization()(block3)
    block3 = DepthwiseConv2D((Chans, 1), use_bias = False, padding='valid', depth_multiplier = n_sf[2], activation = 'linear',
                                depthwise_constraint = tf.keras.constraints.MaxNorm(max_value=1), 
                                name = 'Spatial_filter_3')(block3)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)

    #================================

    block = Concatenate(axis = -1)([block1, block2, block3]) 

    #================================

    block = AveragePooling2D((1, 4))(block)
    block_in = Dropout(drop_rate)(block)

    #================================

    paddings = tf.constant([[0,0], [0,0], [3,0], [0,0]])
    block = tf.pad(block_in, paddings, "CONSTANT")
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.pad(block, paddings, "CONSTANT")
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_in, block])


    paddings = tf.constant([[0,0], [0,0], [6,0], [0,0]])
    block = tf.pad(block_out, paddings, "CONSTANT")
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.pad(block, paddings, "CONSTANT")
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block])


    paddings = tf.constant([[0,0], [0,0], [12,0], [0,0]])
    block = tf.pad(block_out, paddings, "CONSTANT")
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.pad(block, paddings, "CONSTANT")
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block]) 


    paddings = tf.constant([[0,0], [0,0], [24,0], [0,0]])
    block = tf.pad(block_out, paddings, "CONSTANT")
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.pad(block, paddings, "CONSTANT")
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block]) 

    #================================

    block = block_out

    #================================

    block = Conv2D(28, (1,1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = AveragePooling2D((4,1), data_format='Channels_first')(block)
    block = Dropout(drop_rate)(block) 
    embedded = Flatten()(block)
    out = Dense(out_class, activation = 'softmax', kernel_constraint = max_norm(0.25))(embedded)

    #========================================================================================
    return Model(inputs = Input_block, outputs = out)