from __future__ import print_function

from StringIO import StringIO

import numpy as np
import sys
import argparse
import pickle  # for handling the new data source
import h5py  # for saving the model
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from datetime import datetime  # for filename conventions

from keras.layers import GlobalAveragePooling2D, Reshape, Activation
from keras.layers.merge import concatenate
from MobileNet.depthwiseconv import DepthwiseConvolution2D

from tensorflow.python.lib.io import file_io  # for better file I/O


def multiinput_generator(full, med, low, label):
    '''custom generator to be passed to main training
       note samplewise std normalization + batch size
    '''
    while True:
        # shuffled indices
        idx = np.random.permutation(full.shape[0])
        # create image generator
        datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                     samplewise_center=False,  # set each sample mean to 0
                                     featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                     samplewise_std_normalization=False,  # divide each input by its std
                                     zca_whitening=False)  # randomly flip images
        batches = datagen.flow(full[idx], label[idx], batch_size=1, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            yield [batch[0], med[idx[idx0:idx1]], low[idx[idx0:idx1]]], batch[1]
            idx0 = idx1
            if idx1 >= full.shape[0]:
                break


def train_test_split(array, proportion=0.8):
    '''non randomised train split
    '''
    index = int(len(array) * proportion)
    train = array[:index]
    test = array[index:]
    return train, test


def radian_to_angle(radian_array):
    '''converts original radian to angle which
       will be error metric
    '''
    return (radian_array * 180 / np.pi) - 90


def reverse_mean_std(standardized_array, prev_mean, prev_std):
    '''undo transformation in order to calculate
       angle loss
    '''
    de_std = standardized_array * prev_std
    de_mean = de_std + prev_mean
    return de_mean


def generator_train(full, med, low, labels, kernel_size, filters, top_neurons, dropout):
    '''main entry point
       calls customised  multiinput generator
       and tests angle loss
    '''
    full = [x.astype('float32') for x in full]
    full = np.array([(x / 255) for x in full])
    med = [x.astype('float32') for x in med]
    med = np.array([x / 255 for x in med])
    low = [x.astype('float32') for x in low]
    low = np.array([x / 255 for x in low])
    model = multires_mobilenet(full, med, low, filters, kernel_size, top_neurons, dropout)
    train_full, test_full = train_test_split(full)
    train_med, test_med = train_test_split(med)
    train_low, test_low = train_test_split(low)
    labels_angles = radian_to_angle(labels)
    train_orig_lab, test_orig_lab = train_test_split(labels_angles)
    mean_ = None
    std_ = None
    labels_standardised = (labels_angles - (-45))/(90)
    train_labels, test_labels = train_test_split(labels_standardised)
    model.fit_generator(multiinput_generator(train_full, train_med, train_low, train_labels),
                        steps_per_epoch=6000,
                        epochs=15)
    return model, test_full, test_med, test_low, test_labels, mean_, std_, test_orig_lab


def calculate_error(model, test_full, test_med, test_low, test_labels, mean_, std_,
                    kernel_size, filters, top_neurons, dropout, train_files, test_orig_lab):
    std_angles = model.predict([test_full, test_med, test_low])
    std_angles *= 90
    std_angles += (-45)
    unstd_angles = std_angles#reverse_mean_std(std_angles, mean_, std_)
    print('angles')
    for x in unstd_angles:
        print(x)
    #error = unstd_angles - test_labels
    error = abs((unstd_angles) - (test_orig_lab))
    for x in error:
        print(x)
    mean_error_elevation = np.mean(abs(error[:, 0]))
    mean_error_zenith = np.mean(abs(error[:, 1]))
    print('\n' * 10)
    print('kernel size: {}'.format(kernel_size))
    print('filters: {}'.format(filters))
    print('zenith: {}'.format(mean_error_zenith))
    print('elevation: {}'.format(mean_error_elevation))
    print('\n' * 10)
    print('BAD CONFIG')
    file_content = """glorot long good bw 
                      no pooling: kernel_size: {}, filters: {}, 
                      elevation: {}, zenith: {}, top_neurons: {},
                      dropout_both_layers: {} \n""".format(kernel_size,
                                                           filters,
                                                           mean_error_elevation,
                                                           mean_error_zenith,
                                                           top_neurons,
                                                           dropout)

    with file_io.FileIO(train_files + '/mobileresults.txt', mode="a") as f:
        f.write(file_content)

    return mean_error_elevation, mean_error_zenith

def mean_std_norm(array):
    '''standardization for labels
    '''
    mean_ = np.mean(array)
    std_ = np.std(array)
    standardized = (array - mean_) / std_
    return standardized, mean_, std_


def convblock(input, alpha, filters, kernel_size):
    x = Conv2D(int(filters * alpha), (kernel_size, kernel_size), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='glorot_normal')(input)
    x = BatchNormalization()(x)
    x = Activation(activation=LeakyReLU(0.1))(x)
    return x


def depthconvblock(input, filters, alpha=1):
    print(input.get_shape())
    x = DepthwiseConvolution2D(int(filters * alpha), (3, 3), depth_multiplier = 1, 
                               strides=(1, 1), padding='same', use_bias=False, kernel_initializer='glorot_normal')(input)
    print(x.get_shape())
    x = BatchNormalization()(x)
    x = Activation(activation=LeakyReLU(0.1))(x)
    print(x.get_shape())
    x = Conv2D(int(filters * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='glorot_normal')(x)
    print(x.get_shape())
    x = BatchNormalization()(x)
    print(x.get_shape())
    x = Activation(activation=LeakyReLU(0.1))(x)
    return x

def mobilenet(data, res, alpha=1,
              include_top=False,
              dropout=0.25,
              filters=10, kernel_size=3):
    '''downsized mobilenet configured
       for regression so top half of 
       network differs from original
       implementation
    '''
    if include_top:
        input = Input(shape=data.shape[1:])
    else:
        input = data
    x = convblock(input, alpha, filters, kernel_size)
    x = depthconvblock(x, alpha, filters)
    x = depthconvblock(x, alpha, filters)
    x = depthconvblock(x, alpha, filters)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dropout(dropout)
        output_layer = Dense(2, activation='linear')(x)
        model = Model(input, output_layer, name='mobilenet')
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    else:
        return x

def multires_mobilenet(full, med, low, filters, kernel_size, top_neurons, dropout):
    '''uses three mobile nets as configured
       in mobilenet() and concatenates output
       before fully-connected layers
    '''
    input_fullres = Input(full.shape[1:], name ='input_fullres')
    input_medres = Input(med.shape[1:], name ='input_medres')
    input_lowres = Input(low.shape[1:], name ='input_lowres')

    fullres_mobilenet = mobilenet(input_fullres, 'full', dropout=dropout, filters=filters, kernel_size=kernel_size)
    medres_mobilenet = mobilenet(input_medres, 'med', dropout=dropout, filters=filters, kernel_size=kernel_size)
    lowres_mobilenet = mobilenet(input_lowres, 'low', dropout=dropout, filters=filters, kernel_size=kernel_size)

    merged_branches = concatenate([fullres_mobilenet, medres_mobilenet, lowres_mobilenet])
    merged_branches = Dense(top_neurons, activation=LeakyReLU(0.1))(merged_branches)
    merged_branches = Dropout(dropout)(merged_branches)
    merged_branches = Dense(int(top_neurons /2), activation=LeakyReLU(0.1))(merged_branches)
    merged_branches = Dropout(dropout)(merged_branches)
    merged_branches = Dense(2,activation='linear')(merged_branches)

    model = Model(inputs=[input_fullres, input_medres ,input_lowres],
                  outputs=[merged_branches])
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model



def train_model(train_files='hand-data', job_dir='./tmp/test1', kernel_size=5,
                filters=16, top_neurons=128, dropout=0.5, **args):
    """ main entry point for processing args and training model
    """
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('-----------------------')
    print('Using train_file located at {}'.format(train_files))
    print('Using logs_path located at {}'.format(logs_path))
    print('-----------------------')
    print('-----------------------')
    print('-----------------------')
    print('-----------------------')
    print('-----------------------')
    print(args)


    # wrong names for now.....
    kernel_size = int(kernel_size)
    filters = int(filters)
    top_neurons = int(top_neurons)
    dropout = float(dropout)

    # wrong names for now.....
    imagesio = StringIO(file_io.read_file_to_string(train_files+'/AllImagesBW.npy'))
    imagesio64 = StringIO(file_io.read_file_to_string(train_files+'/AllImagesBW64.npy'))
    imagesio32 = StringIO(file_io.read_file_to_string(train_files+'/AllImagesBW32.npy'))
    labelsio = StringIO(file_io.read_file_to_string(train_files+'/AllAngles.npy'))

    full = np.load(imagesio)
    full = np.reshape(full, [len(full), 128, 128, 1])
    med = np.load(imagesio64)
    med = np.reshape(med, [len(med), 64, 64, 1])
    low = np.load(imagesio32)
    low = np.reshape(low, [len(low), 32, 32, 1])
    labels = np.load(labelsio)


    model, test_full, test_med, test_low, test_labels, mean_, std_, test_orig_lab = generator_train(full,
                                                                                     med,
                                                                                     low,
                                                                                     labels,
                                                                                     kernel_size,
                                                                                     filters,
                                                                                     top_neurons,
                                                                                     dropout)

    error = calculate_error(model, test_full, test_med, test_low, test_labels,
                            mean_, std_, kernel_size, filters, top_neurons,
                            dropout, train_files, test_orig_lab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-files',
                        help='GCS or local paths to training data',
                        required=True)

    parser.add_argument('--job-dir',
                        help='GCS location to write checkpoints and export models',
                        required=True)

    parser.add_argument('--kernel_size',
                        help='param for cnn')

    parser.add_argument('--filters',
                        help='param for cnn')

    parser.add_argument('--top_neurons',
                        help='param for cnn')

    parser.add_argument('--dropout',
                        help='param for cnn')

    args = parser.parse_args()
    arguments = args.__dict__
    print(arguments)
    train_model(**arguments)
