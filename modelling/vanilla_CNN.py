from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from data_preprocessing import get_input_shape

def vanilla_CNN(filters, kernel_size, data):
    model = Sequential()
    model.add(Conv2D(filters, (kernel_size, kernel_size), 
                     input_shape = data.shape[1:],
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