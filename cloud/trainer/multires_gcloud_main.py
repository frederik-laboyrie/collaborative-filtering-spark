from __future__ import print_function

from data_preprocessing import load_standardized_multires
from multires_CNN import multires_CNN
import sys
import argparse
import pickle  # for handling the new data source
import h5py  # for saving the model
import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from datetime import datetime  # for filename conventions

from tensorflow.python.lib.io import file_io  # for better file I/O

def reshape(data):
    return np.reshape(data, get_input_shape(data))

def subsample(data, labels, nb_samples):
    return data[:nb_samples], labels[:nb_samples]

def per_image_standardization(arrays):
    '''only works for 1 call before
       ValueError tf graph > 2gb.
    '''
    sess = tf.InteractiveSession()
    standardized_tensors = tf.map_fn(lambda array: 
                                     tf.image.per_image_standardization(array), 
                                     arrays)
    standardized_images = standardized_tensors.eval()
    return standardized_images

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

def multires_CNN(filters, kernel_size, multires_data):
    '''uses Functional API for Keras 2.x support.
       multires data is output from load_standardized_multires()
    '''
    input_fullres = Input(multires_data[0].shape[1:], name = 'input_fullres')
    fullres_branch = Conv2D(filters, (kernel_size, kernel_size),
                     activation = LeakyReLU())(input_fullres)
    fullres_branch = MaxPooling2D(pool_size = (2,2))(fullres_branch)
    fullres_branch = BatchNormalization()(fullres_branch)
    fullres_branch = Flatten()(fullres_branch)

    input_medres = Input(multires_data[1].shape[1:], name = 'input_medres')
    medres_branch = Conv2D(filters, (kernel_size, kernel_size),
                     activation=LeakyReLU())(input_medres)
    medres_branch = MaxPooling2D(pool_size = (2,2))(medres_branch)
    medres_branch = BatchNormalization()(medres_branch)
    medres_branch = Flatten()(medres_branch)

    input_lowres = Input(multires_data[2].shape[1:], name = 'input_lowres')
    lowres_branch = Conv2D(filters, (kernel_size, kernel_size),
                     activation = LeakyReLU())(input_lowres)
    lowres_branch = MaxPooling2D(pool_size = (2,2))(lowres_branch)
    lowres_branch = BatchNormalization()(lowres_branch)
    lowres_branch = Flatten()(lowres_branch)

    merged_branches = concatenate([fullres_branch, medres_branch, lowres_branch])
    merged_branches = Dense(128, activation=LeakyReLU())(merged_branches)
    merged_branches = Dropout(0.5)(merged_branches)
    merged_branches = Dense(2,activation='linear')(merged_branches)

    model = Model(inputs=[input_fullres, input_medres ,input_lowres],
                  outputs=[merged_branches])
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model

def train_model(train_file='hand-data/AllImages.npy',
                train_labels='hand-data/AllLabels.npy',
                job_dir='./tmp/test1',**args):
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('-----------------------')
    print('Using train_file located at {}'.format(train_file))
    print('Using logs_path located at {}'.format(logs_path))
    print('-----------------------')

    images = np.load(train_file)
    labels = np.load(train_labels)

    images_reshape = reshape(images)
    images_ss, labels_ss = subsample(images_reshape, labels, 500)
    standardized_images = per_image_standardization(images_ss)

    standardized_images, labels = load_standardized_singleres()
    multires_data = singleres_to_multires(standardized_images)

    model = multires_CNN(16, 5, multires_data)
    full = multires_data[0]
    med = multires_data[1]
    low = multires_data[2]
    history = model.fit([full,med,low],labels, epochs = 40)
    model.save('model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file',
                        help='GCS or local paths to training data',
                        required=True)

    parser.add_argument('--train-labels',
                        help='GCS or local paths to training labels',
                        required=True)

    parser.add_argument('--job-dir',
                        help='GCS location to write checkpoints and export models',
                        required=True)

    args = parser.parse_args()
    arguments = args.__dict__
    
    train_model(**arguments)