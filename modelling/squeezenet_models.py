'''implementations of squeezenet
   largely based on Refik Can Malli's
   implementation but for regression
   and generalised to scaled up to
   multiresolution networks
'''

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,\
                         Dropout, Flatten, Dense, Input, Activation
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"

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

def squeezenet(data, standalone, leaky, modular, res):
    '''squeezenet implementation
       with structure as in original
       paper. note bottleneck replaces
       number of classes as this is
       used for regression.
    '''
    bottleneck = 128

    if modular:
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

    if standalone:
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
    input_fullres = Input(multires_data[0].shape[1:], name = 'input_fullres')
    input_medres = Input(multires_data[1].shape[1:], name = 'input_medres')
    input_lowres = Input(multires_data[2].shape[1:], name = 'input_lowres')

    fullres_squeezenet = squeezenet(input_fullres, standalone=True, leaky=leaky, modular=True, res='full')
    medres_squeezenet = squeezenet(input_medres, standalone=True, leaky=leaky, modular=True, res='med')
    lowres_squeezenet = squeezenet(input_lowres, standalone=True, leaky=leaky, modular=True, res='low')

    merged_branches = concatenate([fullres_squeezenet, medres_squeezenet, lowres_squeezenet])
    merged_branches = Dense(128, activation=LeakyReLU())(merged_branches)
    merged_branches = Dropout(0.5)(merged_branches)
    merged_branches = Dense(2,activation='linear')(merged_branches)

    model = Model(inputs=[input_fullres, input_medres ,input_lowres],
                  outputs=[merged_branches])
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model
