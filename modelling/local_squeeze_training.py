
import numpy as np
import sys
import argparse
import pickle  # for handling the new data source
import h5py  # for saving the model
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Input, Activation
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from datetime import datetime  # for filename conventions


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
                                     samplewise_std_normalization=True,  # divide each input by its std
                                     zca_whitening=False)  # randomly flip images
        batches = datagen.flow(full[idx], label[idx], batch_size=8, shuffle=False)
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


def generator_train(full, med, low, labels, kernel_size, filters, top_neurons, dropout, squeeze_param, bottleneck, leaky=True):
    '''main entry point
       calls customised  multiinput generator
       and tests angle loss
    '''
    full = [x.astype('float32') for x in full]
    full = np.array([x / 255 for x in full])
    med = [x.astype('float32') for x in med]
    med = np.array([x / 255 for x in med])
    low = [x.astype('float32') for x in low]
    low = np.array([x / 255 for x in low])

    model = multires_squeezenet(full, med, low, leaky, filters,
                                kernel_size, top_neurons, dropout,
                                squeeze_param, bottleneck)

    train_full, test_full = train_test_split(full)
    train_med, test_med = train_test_split(med)
    train_low, test_low = train_test_split(low)
    labels_angles = radian_to_angle(labels)
    train_orig_lab, test_orig_lab = train_test_split(labels_angles)
    labels_standardised, mean_, std_ = mean_std_norm(labels_angles)
    train_labels, test_labels = train_test_split(labels_standardised)
    model.fit_generator(multiinput_generator(train_full, train_med, train_low, train_labels),
                        steps_per_epoch=8,
                        epochs=75)
    return model, test_full, test_med, test_low, test_labels, mean_, std_


def calculate_error(model, test_full, test_med, test_low, test_labels, mean_, std_,
                    kernel_size, filters, top_neurons, dropout, squeeze_param, bottleneck):
    std_angles = model.predict([test_full, test_med, test_low])
    unstd_angles = reverse_mean_std(std_angles, mean_, std_)
    error = unstd_angles - test_labels
    mean_error_elevation = np.mean(abs(error[:, 0]))
    mean_error_zenith = np.mean(abs(error[:, 1]))
    print('\n' * 10)
    print('kernel size: {}'.format(kernel_size))
    print('filters: {}'.format(filters))
    print('zenith: {}'.format(mean_error_zenith))
    print('elevation: {}'.format(mean_error_elevation))
    print('\n' * 10)
    file_content = """deep, highres pooling, 11 strides, double layer, 3 fire modules SQUEEZE, expandfactor3: kernel_size: {}, filters: {}, 
                      elevation: {}, zenith: {}, top_neurons: {},
                      dropout_both_layers: {}, squeeze_param: {},
                      bottleneck: {} \n""".format(kernel_size,
                                                  filters,
                                                  mean_error_elevation,
                                                  mean_error_zenith,
                                                  top_neurons,
                                                  dropout,
                                                  squeeze_param,
                                                  bottleneck)

    with open('squeezeresults.txt', mode="a") as f:
        f.write(file_content)

    return mean_error_elevation, mean_error_zenith


def mean_std_norm(array):
    '''standardization for labels
    '''
    mean_ = np.mean(array)
    std_ = np.std(array)
    standardized = (array - mean_) / std_
    return standardized, mean_, std_


def fire_module(x, fire_id, leaky, res, squeeze_param, expand_param,
                sq1x1="squeeze1x1", exp1x1="expand1x1", exp3x3="expand3x3"):
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

def squeezenet(data, leaky, exclude_top, res, squeeze_param, filters, kernel_size, bottleneck, pooling):
    '''squeezenet implementation
       with structure as in original
       paper. note bottleneck replaces
       number of classes as this is
       used for regression.
    '''
    if exclude_top:
        input = data
    else:
        input = Input(shape=data.shape[1:])

    x = Conv2D(filters, (kernel_size, kernel_size), strides=(1, 1), padding='valid', name='conv1' + res)(input)
    x = Activation(LeakyReLU(), name='relu_conv1' + res)(x)

    if pooling:
        x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), name='pool1' + res)(x)

    x = fire_module(x, fire_id=2, leaky=leaky, res=res, squeeze_param=int(squeeze_param), expand_param=int(squeeze_param)*3)
    x = fire_module(x, fire_id=3, leaky=leaky, res=res, squeeze_param=int(squeeze_param), expand_param=int(squeeze_param)*3)

    if pooling:
        x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), name='pool3' + res)(x)

    x = fire_module(x, fire_id=4, leaky=leaky, res=res, squeeze_param=int(squeeze_param), expand_param=int(squeeze_param)*3)
    x = fire_module(x, fire_id=5, leaky=leaky, res=res, squeeze_param=int(squeeze_param), expand_param=int(squeeze_param)*3)

    if pooling:
        x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), name='pool5' + res)(x)

    x = fire_module(x, fire_id=6, leaky=leaky, res=res, squeeze_param=int(squeeze_param), expand_param=int(squeeze_param)*3)
    x = fire_module(x, fire_id=7, leaky=leaky, res=res, squeeze_param=int(squeeze_param), expand_param=int(squeeze_param)*3)
    x = fire_module(x, fire_id=8, leaky=leaky, res=res, squeeze_param=int(squeeze_param), expand_param=int(squeeze_param)*3)
    x = fire_module(x, fire_id=9, leaky=leaky, res=res, squeeze_param=int(squeeze_param), expand_param=int(squeeze_param)*3)
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


def multires_squeezenet(full, med, low, leaky, filters, kernel_size,
                        top_neurons, dropout, squeeze_param, bottleneck):
    '''uses three full size squeezenets
       and concatenates output into
       small final fully-connected layers.
    '''
    input_fullres = Input(full.shape[1:], name='input_fullres')
    input_medres = Input(med.shape[1:], name='input_medres')
    input_lowres = Input(low.shape[1:], name='input_lowres')

    fullres_squeezenet = squeezenet(input_fullres, leaky=leaky, exclude_top=True, res='full',
                                    squeeze_param=int(squeeze_param), kernel_size=int(kernel_size),
                                    filters=int(filters), bottleneck=int(bottleneck), pooling=True)

    medres_squeezenet = squeezenet(input_medres, leaky=leaky, exclude_top=True, res='med',
                                   squeeze_param=int(squeeze_param), kernel_size=int(kernel_size),
                                   filters=int(filters), bottleneck=int(bottleneck), pooling=False)

    lowres_squeezenet = squeezenet(input_lowres, leaky=leaky, exclude_top=True, res='low',
                                   squeeze_param=int(squeeze_param), kernel_size=int(kernel_size),
                                   filters=int(filters), bottleneck=int(bottleneck), pooling=False)

    merged_branches = concatenate([fullres_squeezenet, medres_squeezenet, lowres_squeezenet])
    merged_branches = Dense(top_neurons, activation=LeakyReLU())(merged_branches)
    merged_branches = Dropout(float(dropout))(merged_branches)
    merged_branches = Dense(int(top_neurons/2), activation=LeakyReLU())(merged_branches)
    merged_branches = Dense(2, activation='linear')(merged_branches)

    model = Model(inputs=[input_fullres, input_medres, input_lowres],
                  outputs=[merged_branches])
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model




def train_model(kernel_size=3,
                filters=15, top_neurons=128, dropout=0.4, squeeze_param=3,
                bottleneck=64):
    """ main entry point for processing args and training model
    """

    kernel_size = int(kernel_size)
    filters = int(filters)
    top_neurons = int(top_neurons)
    dropout = float(dropout)
    squeeze_param = int(squeeze_param)
    bottleneck = int(bottleneck)

    # wrong names for now.....
    imagesio = 'AllImages.npy'
    imagesio64 = 'AllAngles64.npy'
    imagesio32 = 'AllAngles32.npy'
    labelsio = 'AllAngles.npy'

    full = np.load(imagesio)
    full = np.reshape(full, [len(full), 128, 128, 3])
    med = np.load(imagesio64)
    med = np.reshape(med, [len(med), 64, 64, 3])
    low = np.load(imagesio32)
    low = np.reshape(low, [len(low), 32, 32, 3])
    labels = np.load(labelsio)

    model, test_full, test_med, test_low, test_labels, mean_, std_ = generator_train(full,
                                                                                     med,
                                                                                     low,
                                                                                     labels,
                                                                                     kernel_size,
                                                                                     filters,
                                                                                     top_neurons,
                                                                                     dropout,
                                                                                     squeeze_param,
                                                                                     bottleneck)

    error = calculate_error(model, test_full, test_med, test_low, test_labels,
                            mean_, std_, kernel_size, filters, top_neurons,
                            dropout, squeeze_param, bottleneck)

if __name__ == '__main__':
    train_model()
