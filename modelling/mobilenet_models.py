'''uses mobilenet components
   to build custom mobilenets
   differing from original
   implementation
'''

from keras.layers import Conv2D, GlobalAveragePooling2D, Reshape,\
                         Dropout, Flatten, Dense, Input, Activation
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from MobileNet.depthwiseconv import DepthwiseConvolution2D

def convblock(input, alpha):
    x = Conv2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def depthconvblock(input, alpha):
    x = DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(32 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def mobilenet(data, res, alpha,
              include_top,
              dropout=0.25,
              depth_multiplier=1,
              strides=(2,2)):
    '''downsized mobilenet configured
       for regression so top half of 
       network differs from original
       implementation
    '''
    if include_top:
        input = Input(shape=data.shape[1:])
    else:
        input = data
    x = convblock(input, alpha)
    x = depthconvblock(x, alpha)
    x = depthconvblock(x, alpha)
    x = depthconvblock(x, alpha)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dropout(dropout)
        output_layer = Dense(2, activation='linear')(x)
        model = Model(input, output_layer, name='mobilenet')
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    else:
        return x

def multires_mobilenet(multires_data):
    '''uses three mobile nets as configured
       in mobilenet() and concatenates output
       before fully-connected layers
    '''
    input_fullres = Input(multires_data[0].shape[1:], name ='input_fullres')
    input_medres = Input(multires_data[1].shape[1:], name ='input_medres')
    input_lowres = Input(multires_data[2].shape[1:], name ='input_lowres')

    fullres_mobilenet = mobilenet(input_fullres, 'full', 1, False)
    medres_mobilenet = mobilenet(input_medres, 'med', 1, False)
    lowres_mobilenet = mobilenet(input_lowres, 'low', 1, False)

    merged_branches = concatenate([fullres_mobilenet, medres_mobilenet, lowres_mobilenet])
    merged_branches = Dense(128, activation=LeakyReLU())(merged_branches)
    merged_branches = Dropout(0.5)(merged_branches)
    merged_branches = Dense(2,activation='linear')(merged_branches)

    model = Model(inputs=[input_fullres, input_medres ,input_lowres],
                  outputs=[merged_branches])
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model
