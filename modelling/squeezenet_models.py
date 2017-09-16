from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,\
                         Dropout, Flatten, Dense, Input, Activation
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"

def fire_module(x, fire_id, leaky, squeeze_param=16, expand_param=64):
    '''implementation of fire module as in
       SqueezeNet paper consisting of squeeze
       and expand phases. x represents input
       from previous layer.
    '''
    if leaky:
        relu_type = LeakyReLU()
    else:
        relu_type = 'relu'

    s_id = 'fire' + str(fire_id) + '/'

    x = Conv2D(squeeze_param, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + str(relu_type) + sq1x1)(x)

    left = Conv2D(expand_param, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + str(relu_type) + exp1x1)(left)

    right = Conv2D(expand_param, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + str(relu_type) + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x

def single_res_squeeznet(data, standalone):
    '''squeezenet implementation
       with structure as in original
       paper
    '''
    input = Input(shape=data.shape[1:])

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)

    x = Conv2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)

    if standalone:
        return x

    else:
        output_layer = Dense(2, activation='linear')(x)
        model = Model(input, output_layer, name='squeezenet')
        model.compile(loss='mean_absolute_error', optimizer='adam')

        return model