from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

def multires_CNN(filters, kernel_size, multires_data):
    '''multires data is output
       from load_standardized_multires()
    '''
    fullres_branch = Sequential()
    fullres_branch.add(Conv2D(filters, (kernel_size, kernel_size), 
                     input_shape = multires_data[0].shape[1:],
                     activation = LeakyReLU()))
    fullres_branch.add(MaxPooling2D(pool_size = (2,2)))
    fullres_branch.add(BatchNormalization())
    fullres_branch.add(Flatten())

    medres_branch = Sequential()
    medres_branch.add(Conv2D(filters, (kernel_size, kernel_size), 
                     input_shape = multires_data[1].shape[1:],
                     activation = LeakyReLU()))
    medres_branch.add(MaxPooling2D(pool_size = (2,2)))
    medres_branch.add(BatchNormalization())
    medres_branch.add(Flatten())

    lowres_branch = Sequential()
    lowres_branch.add(Conv2D(filters, (kernel_size, kernel_size), 
                     input_shape = multires_data[2].shape[1:],
                     activation = LeakyReLU()))
    lowres_branch.add(MaxPooling2D(pool_size = (2,2)))
    lowres_branch.add(BatchNormalization())
    lowres_branch.add(Flatten())

    merged_branches = Merge([fullres_branch, medres_branch, lowres_branch], mode='concat')

    model = Sequential()
    model.add(merged_branches)
    model.add(Dense(128, activation = LeakyReLU()))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation ='linear'))
    model.compile(loss = 'mean_absolute_error', optimizer='adam')
    return model