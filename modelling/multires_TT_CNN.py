from TensorTrain.TTLayer import TT_Layer
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.regularizers import l2

def multires_TT_CNN(filters, kernel_size, tt_input_shape,
                    tt_output_shape, tt_ranks, multires_data):

    input_fullres_tt = Input(multires_data[0].shape[1:], name='input_fullres_tt')

    fullres_branch = Conv2D(filters, (kernel_size, kernel_size),
                            activation=LeakyReLU())(input_fullres_tt)
    fullres_branch = MaxPooling2D(pool_size=(2, 2))(fullres_branch)
    fullres_branch = BatchNormalization()(fullres_branch)
    fullres_branch = Flatten()(fullres_branch)

    input_medres_tt = Input(multires_data[1].shape[1:], name='input_medres_tt')

    medres_branch = Conv2D(filters, (kernel_size, kernel_size),
                           activation=LeakyReLU())(input_medres_tt)
    medres_branch = MaxPooling2D(pool_size=(2, 2))(medres_branch)
    medres_branch = BatchNormalization()(medres_branch)
    medres_branch = Flatten()(medres_branch)

    input_lowres_tt = Input(multires_data[2].shape[1:], name='input_lowres_tt')

    lowres_branch = Conv2D(filters, (kernel_size, kernel_size),
                           activation=LeakyReLU())(input_lowres_tt)
    lowres_branch = MaxPooling2D(pool_size=(2, 2))(lowres_branch)
    lowres_branch = BatchNormalization()(lowres_branch)
    lowres_branch = Flatten()(lowres_branch)

    merged_branches = concatenate([fullres_branch, medres_branch, lowres_branch])

    TT_merged_layer_1 = TT_Layer(tt_input_shape=tt_input_shape,
                                 tt_output_shape=tt_output_shape,
                                 tt_ranks=tt_ranks,
                                 activation=LeakyReLU(),
                                 use_bias=True,
                                 kernel_regularizer=l2(.001), )(merged_branches)
    TT_merged_layer_1 = Dropout(0.5)(TT_merged_layer_1)

    TT_merged_layer_2 = TT_Layer(tt_input_shape=tt_input_shape, 
                                 tt_output_shape=tt_output_shape,
                                 tt_ranks=tt_ranks,
                                 activation=LeakyReLU(), 
                                 use_bias=True,
                                 kernel_regularizer=l2(.001), )(TT_merged_layer_1)

    output_layer = Dense(2, activation='linear')(TT_merged_layer_2)

    model = Model(inputs=[input_fullres_tt, input_medres_tt, input_lowres_tt],
                  outputs=[output_layer])
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model