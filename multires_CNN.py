from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

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
                     activation = LeakyReLU())(input_medres)
    medres_branch = MaxPooling2D(pool_size = (2,2))(medres_branch)
    medres_branch = BatchNormalization()(medres_branch)
    medres_branch = Flatten()(medres_branch)

    input_lowres = Input(multires_data[1].shape[1:], name = 'input_lowres')
    lowres_branch = Conv2D(filters, (kernel_size, kernel_size),
                     activation = LeakyReLU())(input_lowres)
    lowres_branch = MaxPooling2D(pool_size = (2,2))(lowres_branch)
    lowres_branch = BatchNormalization()(lowres_branch)
    lowres_branch = Flatten()(lowres_branch)

    merged_branches = concatenate([fullres_branch, medres_branch, lowres_branch])
    merged_branches = Dense(128, activation = LeakyReLU())(merged_branches)
    merged_branches = Dropout(0.5)(merged_branches)
    merged_branches = Dense(2,activation ='linear')(merged_branches)

    model = Model(inputs=[input_fullres,input_medres,input_lowres],outputs=[merged_branches])
    model.compile(loss = 'mean_absolute_error', optimizer='adam')

    return model