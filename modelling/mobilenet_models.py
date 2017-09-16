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
from MobileNet.mobilenet_components import _conv_block, _depthwise_conv_block

def conv_block3(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), activation=LeakyReLU()):
    channel_axis = 1
    filters = float(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def mobilenet(data, res, alpha,
              activation,
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
    print(activation)
    print(alpha)
    x = _conv_block(inputs=input, filters=32, alpha=alpha, strides=(2,2), activation='relu')
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=2)
    x = GlobalAveragePooling2D()(x)
    #x = Flatten()(x)

    if include_top:
        x = Dropout(dropout)
        output_layer = Dense(2, activation='linear')(x)
        model = Model(input, output_layer, name='mobilenet')
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    else:
        return x

def multires_mobilenet(multires_data, activation):
    '''uses three mobile nets as configured
       in mobilenet() and concatenates output
       before fully-connected layers
    '''
    input_fullres = Input(multires_data[0].shape[1:], name ='input_fullres')
    input_medres = Input(multires_data[1].shape[1:], name ='input_medres')
    input_lowres = Input(multires_data[2].shape[1:], name ='input_lowres')

    fullres_mobilenet = mobilenet(input_fullres, 'full', 1, activation, False)
    medres_mobilenet = mobilenet(input_medres, 'med', 1, activation, False)
    lowres_mobilenet = mobilenet(input_lowres, 'low', 1, activation, False)

    merged_branches = concatenate([fullres_mobilenet, medres_mobilenet, lowres_mobilenet])
    merged_branches = Dense(128, activation=LeakyReLU())(merged_branches)
    merged_branches = Dropout(0.5)(merged_branches)
    merged_branches = Dense(2,activation='linear')(merged_branches)

    model = Model(inputs=[input_fullres, input_medres ,input_lowres],
                  outputs=[merged_branches])
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model
