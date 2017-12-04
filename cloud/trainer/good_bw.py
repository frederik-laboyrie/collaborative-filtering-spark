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
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from datetime import datetime  # for filename conventions

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
    model = multires_CNN(filters, kernel_size, top_neurons, dropout, full, med, low)
    train_full, test_full = train_test_split(full)
    train_med, test_med = train_test_split(med)
    train_low, test_low = train_test_split(low)
    labels_angles = radian_to_angle(labels)
    train_orig_lab, test_orig_lab = train_test_split(labels_angles)
    #labels_standardised, mean_, std_ = mean_std_norm(labels_angles)
    #labels_standardised +=2
    #labels_standardised /=2
    mean_ = None
    std_ = None
    labels_standardised = (labels_angles - (-45))/(90)
    train_labels, test_labels = train_test_split(labels_standardised)
    tb_logs = keras.callbacks.TensorBoard(log_dir='gs://hand-data/logs/',
                                            histogram_freq=1,
                                            write_graph=True)
    model.fit_generator(multiinput_generator(train_full, train_med, train_low, train_labels),
                        validation_data = ([test_full[:100], test_med[:100], test_low[:100]], test_labels[:100]),
                        steps_per_epoch=6000,
                        epochs=10,
                        callbacks=[tb_logs])
    #model.save('model_1.h5')
    
    # Save model.h5 on to google storage
    #with file_io.FileIO('model_1.h5', mode='r') as input_f:
    #    with file_io.FileIO('gs://hand-data/' + 'model_1.h5', mode='w+') as output_f:
    #        output_f.write(input_f.read())
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
    file_content = """tb false long more leaky bigger max min VANILLA double layer not pool not batch norm + xavier everywhere + 
                      no pooling: kernel_size: {}, filters: {}, 
                      elevation: {}, zenith: {}, top_neurons: {},
                      dropout_both_layers: {} \n""".format(kernel_size,
                                                           filters,
                                                           mean_error_elevation,
                                                           mean_error_zenith,
                                                           top_neurons,
                                                           dropout)

    with file_io.FileIO(train_files + '/vanilla2.txt', mode="a") as f:
        f.write(file_content)

    return mean_error_elevation, mean_error_zenith


def mean_std_norm(array):
    '''standardization for labels
    '''
    mean_ = np.mean(array)
    std_ = np.std(array)
    standardized = (array - mean_) / std_
    return standardized, mean_, std_


def multires_CNN(filters, kernel_size, top_neurons, dropout, full, med, low):
    '''uses Functional API for Keras 2.x support.
       multires data is output from load_standardized_multires()
    '''
    input_fullres = Input(full.shape[1:], name='input_fullres')
    fullres_branch = Conv2D(filters, (kernel_size, kernel_size), activation=LeakyReLU(alpha=0.1), kernel_initializer='glorot_normal')(input_fullres)
    #fullres_branch = LeakyReLU(alpha=0.1)(fullres_branch)
    #fullres_branch = MaxPooling2D(pool_size=(2, 2))(fullres_branch)
    #fullres_branch = BatchNormalization()(fullres_branch)
    #fullres_branch = Conv2D(filters, (kernel_size, kernel_size),
    #                       activation=LeakyReLU(alpha=0.25))(fullres_branch)
    #fullres_branch = MaxPooling2D(pool_size=(2, 2))(fullres_branch)
    #fullres_branch = BatchNormalization()(fullres_branch)
    fullres_branch = Conv2D(filters, (kernel_size, kernel_size), activation=LeakyReLU(alpha=0.1), kernel_initializer='glorot_normal')(fullres_branch)
    #fullres_branch = LeakyReLU(alpha=0.1)(fullres_branch)
    #fullres_branch = Conv2D(filters, (kernel_size, kernel_size),
    #                        activation=LeakyReLU(alpha=0.25), kernel_initializer='glorot_normal')(fullres_branch)
    #fullres_branch = Conv2D(filters, (kernel_size, kernel_size),
    #                        activation=LeakyReLU(alpha=0.25), kernel_initializer='glorot_normal')(fullres_branch)
    #fullres_branch = MaxPooling2D(pool_size=(2, 2))(fullres_branch)
    #fullres_branch = BatchNormalization()(fullres_branch)
    fullres_branch = Flatten()(fullres_branch)

    input_medres = Input(med.shape[1:], name='input_medres')
    medres_branch = Conv2D(filters, (kernel_size, kernel_size), activation=LeakyReLU(alpha=0.1), kernel_initializer='glorot_normal')(input_medres)
    #medres_branch = LeakyReLU(alpha=0.1)(medres_branch)
    #medres_branch = MaxPooling2D(pool_size=(2, 2))(medres_branch)
    #medres_branch = BatchNormalization()(medres_branch)
    #medres_branch = Conv2D(filters, (kernel_size, kernel_size),
    #                       activation=LeakyReLU(alpha=0.25))(medres_branch)
    #medres_branch = MaxPooling2D(pool_size=(2, 2))(medres_branch)
    #medres_branch = BatchNormalization()(medres_branch)
    medres_branch = Conv2D(filters, (kernel_size, kernel_size), activation=LeakyReLU(alpha=0.1), kernel_initializer='glorot_normal')(medres_branch)
    #medres_branch = LeakyReLU(alpha=0.1)(medres_branch)
    #medres_branch = Conv2D(filters, (kernel_size, kernel_size),
    #                       activation=LeakyReLU(alpha=0.25), kernel_initializer='glorot_normal')(medres_branch)
    #medres_branch = Conv2D(filters, (kernel_size, kernel_size),
    #                       activation=LeakyReLU(alpha=0.25), kernel_initializer='glorot_normal')(medres_branch)
    #medres_branch = MaxPooling2D(pool_size=(2, 2))(medres_branch)
    #medres_branch = BatchNormalization()(medres_branch)
    medres_branch = Flatten()(medres_branch)

    input_lowres = Input(low.shape[1:], name='input_lowres')
    lowres_branch = Conv2D(filters, (kernel_size, kernel_size), activation=LeakyReLU(alpha=0.1), kernel_initializer='glorot_normal')(input_lowres)
    #lowres_branch = LeakyReLU(alpha=0.1)(lowres_branch)
    #lowres_branch = MaxPooling2D(pool_size=(2, 2))(lowres_branch)
    #lowres_branch = BatchNormalization()(lowres_branch)
    #lowres_branch = Conv2D(filters, (kernel_size, kernel_size),
    #                       activation=LeakyReLU(alpha=0.25))(lowres_branch)
    #lowres_branch = MaxPooling2D(pool_size=(2, 2))(lowres_branch)
    #lowres_branch = BatchNormalization()(lowres_branch)
    lowres_branch = Conv2D(filters, (kernel_size, kernel_size), activation=LeakyReLU(alpha=0.1), kernel_initializer='glorot_normal')(lowres_branch)
    #lowres_branch = LeakyReLU(alpha=0.1)(lowres_branch)
    #lowres_branch = Conv2D(filters, (kernel_size, kernel_size),
    #                       activation=LeakyReLU(alpha=0.25), kernel_initializer='glorot_normal')(lowres_branch)
    #lowres_branch = MaxPooling2D(pool_size=(2, 2))(lowres_branch)
    #lowres_branch = BatchNormalization()(lowres_branch)
    lowres_branch = Flatten()(lowres_branch)

    merged_branches = concatenate([fullres_branch, medres_branch, lowres_branch])
    merged_branches = Dense(top_neurons, activation=LeakyReLU(alpha=0.1),
                            kernel_initializer='glorot_normal')(merged_branches)
    #merged_branches = LeakyReLU(alpha=0.1)(merged_branches)
    merged_branches = Dropout(dropout)(merged_branches)
    merged_branches = Dense(int(top_neurons/2), activation=LeakyReLU(alpha=0.1),
                            kernel_initializer='glorot_normal')(merged_branches)
   # merged_branches = LeakyReLU(alpha=0.1)(merged_branches)
    merged_branches = Dropout(dropout)(merged_branches)
    merged_branches = Dense(2, activation='linear')(merged_branches)

    model = Model(inputs=[input_fullres, input_medres, input_lowres],
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