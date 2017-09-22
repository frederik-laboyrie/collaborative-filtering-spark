from __future__ import print_function

from StringIO import StringIO

import argparse
import pickle  # for handling the new data source
import h5py  # for saving the model
import keras
import tensorflow as tf
import numpy as np
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,\
                         AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from datetime import datetime  # for filename conventions

from tensorflow.python.lib.io import file_io 

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"

def get_input_shape(data):
    num_samples = data.shape[0]
    channels = 3
    img_rows = data.shape[2]
    img_cols = data.shape[3]
    return (num_samples, img_rows, img_cols, channels)

def reshape(data):
    return np.reshape(data, get_input_shape(data))

def subsample(data, labels, nb_samples):
    return data[:nb_samples], labels[:nb_samples]

def resizer(arrays, size, method):
    return tf.map_fn(lambda array: 
                     tf.image.resize_images(array,
                                            [size, size],
                                            method=method), 
                     arrays)

def singleres_to_multires(arrays, size1=64, size2=32, 
                          method=tf.image.ResizeMethod.BILINEAR):
    with tf.Session() as session:
        size1_arrays = resizer(arrays, size1, method).eval()
        size2_arrays = resizer(arrays, size2, method).eval()
    return [arrays, size1_arrays, size2_arrays]

def load_multires(images, labels):
    images_reshape = reshape(images)
    images_ss, labels_ss = subsample(images_reshape, labels, 2000)
    multires_images = singleres_to_multires(images_ss)
    return multires_images, labels_ss

def fire_module(x, fire_id, leaky, res, squeeze_param=16, expand_param=64):
    '''implementation of fire module as in
       SqueezeNet paper consisting of squeeze
       and expand phases. x represents input
       from previous layer.
    '''
    if leaky:
        relu_type = LeakyReLU()
        relu_name = 'leaky'
    else:
        relu_type = 'relu'
        relu_name = 'standard'

    s_id = 'fire' + str(fire_id) + '/'

    x = Conv2D(squeeze_param, (1, 1), padding='valid', name=s_id + sq1x1 + res)(x)
    x = Activation(relu_type, name=str(s_id) + str(relu_name) + sq1x1 + res)(x)

    left = Conv2D(expand_param, (1, 1), padding='valid', name=s_id + exp1x1 + res)(x)
    left = Activation(relu_type, name=str(s_id) + str(relu_name) + exp1x1 + res)(left)

    right = Conv2D(expand_param, (3, 3), padding='same', name=s_id + exp3x3 + res)(x)
    right = Activation(relu_type, name=str(s_id) + str(relu_name) + exp3x3 + res)(right)

    x = concatenate([left, right], name=str(s_id) + str(relu_name) + 'concat' + res)
    return x

def squeezenet(data, leaky, exclude_top, res):
    '''squeezenet implementation
       with structure as in original
       paper. note bottleneck replaces
       number of classes as this is
       used for regression.
    '''
    bottleneck = 128

    if exclude_top:
        input = data
    else:
        input = Input(shape=data.shape[1:])

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1' + res)(input)
    x = Activation(LeakyReLU(), name='relu_conv1' + res)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1' + res)(x)

    x = fire_module(x, fire_id=2, leaky=leaky, res=res, squeeze_param=16, expand_param=64)
    x = fire_module(x, fire_id=3, leaky=leaky, res=res, squeeze_param=16, expand_param=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3' + res)(x)

    x = fire_module(x, fire_id=4, leaky=leaky, res=res, squeeze_param=32, expand_param=128)
    x = fire_module(x, fire_id=5, leaky=leaky, res=res, squeeze_param=32, expand_param=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5' + res)(x)

    x = fire_module(x, fire_id=6, leaky=leaky, res=res, squeeze_param=48, expand_param=192)
    x = fire_module(x, fire_id=7, leaky=leaky, res=res, squeeze_param=48, expand_param=192)
    x = fire_module(x, fire_id=8, leaky=leaky, res=res, squeeze_param=64, expand_param=256)
    x = fire_module(x, fire_id=9, leaky=leaky, res=res, squeeze_param=64, expand_param=256)
    x = Dropout(0.5, name='drop9' + res)(x)

    x = Conv2D(bottleneck, (1, 1), padding='valid', name='conv10' + res)(x)
    x = Activation('relu', name='relu_conv10' + res)(x)
    x = GlobalAveragePooling2D()(x)

    if exclude_top:
        return x

    else:
        output_layer = Dense(2, activation='linear')(x)
        model = Model(input, output_layer, name='squeezenet')
        model.compile(loss='mean_absolute_error', optimizer='adam')

        return model

def multires_squeezenet(multires_data, leaky):
    '''uses three full size squeezenets
       and concatenates output into
       small final fully-connected layers.
    '''
    input_fullres = Input(multires_data[0].shape[1:], name ='input_fullres')
    input_medres = Input(multires_data[1].shape[1:], name ='input_medres')
    input_lowres = Input(multires_data[2].shape[1:], name ='input_lowres')

    fullres_squeezenet = squeezenet(input_fullres, leaky=leaky, exclude_top=True, res='full')
    medres_squeezenet = squeezenet(input_medres, leaky=leaky, exclude_top=True, res='med')
    lowres_squeezenet = squeezenet(input_lowres, leaky=leaky, exclude_top=True, res='low')

    merged_branches = concatenate([fullres_squeezenet, medres_squeezenet, lowres_squeezenet])
    merged_branches = Dense(128, activation=LeakyReLU())(merged_branches)
    merged_branches = Dropout(0.5)(merged_branches)
    merged_branches = Dense(2,activation='linear')(merged_branches)

    model = Model(inputs=[input_fullres, input_medres ,input_lowres],
                  outputs=[merged_branches])
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model

def multiinput_generator(full, med, low, label):
    while True:
        # shuffled indices    
        idx = np.random.permutation(full.shape[0])
        # create image generator
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=True,  # divide each input by its std
                zca_whitening=False)  # randomly flip images
        batches = datagen.flow( full[idx], label[idx], batch_size=8, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            yield [batch[0], med[ idx[ idx0:idx1 ] ], low[ idx[ idx0:idx1 ] ]], batch[1]
            idx0 = idx1
            if idx1 >= full.shape[0]:
                break



def train_model(train_files='hand-data',
                job_dir='./tmp/test1',**args):
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('-----------------------')
    print('Using train_file located at {}'.format(train_files))
    print('Using logs_path located at {}'.format(logs_path))
    print('-----------------------')

    imagesio = StringIO(file_io.read_file_to_string(train_files+'/AllImages.npy'))
    labelsio = StringIO(file_io.read_file_to_string(train_files+'/AllAngles.npy'))

    images = np.load(imagesio)
    labels = np.load(labelsio)

    multires_data, labels = load_multires(images, labels)

    multires_data = [x.astype('float32') for x in multires_data]
    multires_data = [x / 255 for x in multires_data]
    model = multires_squeezenet(multires_data, True)
    full = multires_data[0]
    med = multires_data[1]
    low = multires_data[2]
    history = model.fit_generator(multiinput_generator(full, med, low, labels),
                                                       steps_per_epoch=32,
                                                       epochs=100)
    model.save('model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-files',
                        help='GCS or local paths to training data',
                        required=True)

    parser.add_argument('--job-dir',
                        help='GCS location to write checkpoints and export models',
                        required=True)
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)

