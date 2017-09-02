import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

def get_input_shape(data):
    num_samples = data.shape[0]
    channels = 3
    img_rows = data.shape[2]
    img_cols = data.shape[3]
    return (num_samples,img_rows, img_cols, channels)

def reshape(data):
    return np.reshape(data, get_input_shape(data))

def subsample(data, labels, nb_samples):
    return data[:nb_samples], labels[:nb_samples]

def vanilla_CNN(filters, kernel_size, data):
    model = Sequential()
    model.add(Conv2D(filters, (kernel_size, kernel_size), 
                     input_shape = get_input_shape(data)[1:],
                     activation = LeakyReLU()))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters, (kernel_size, kernel_size), 
                     activation = LeakyReLU()))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation = LeakyReLU()))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation ='linear'))
    model.compile(loss = 'mean_absolute_error', optimizer='adam')
    return model

