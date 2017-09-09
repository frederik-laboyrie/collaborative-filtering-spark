from TensorTrain.TTLayer import TT_Layer
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def singleres_TT_CNN(filters, kernel_size, 
                     tt_input_shape, tt_output_shape, tt_ranks):

  input_fullres = Input(data.shape[1:], name = 'input_fullres')

  conv_layer_1 = Conv2D(filters,
                        (kernel_size, kernel_size),
                        activation=LeakyReLU())(input_fullres)
  conv_layer_1 = MaxPooling2D(pool_size = (2,2))(conv_layer_1)
  conv_layer_1 = BatchNormalization()(conv_layer_1)

  conv_layer_2 = Conv2D(filters,
                       (kernel_size, kernel_size),
                       activation=LeakyReLU())(conv_layer_1)
  conv_layer_2 = MaxPooling2D(pool_size = (2,2))(conv_layer_2)
  conv_layer_2 = BatchNormalization()(conv_layer_2)
  conv_layer_2 = Flatten()(conv_layer_2)

  TT_layer_3 = TT_Layer(tt_input_shape=tt_input_shape, 
                        tt_output_shape=tt_output_shape,
                        tt_ranks=tt_ranks,
                        activation=LeakyReLU(), 
                        use_bias=True,
                        kernel_regularizer=l2(.001), )(conv_layer_2)
  TT_layer_3 = Dropout(0.5)(TT_layer_3)

  TT_layer_4 = TT_Layer(tt_input_shape=tt_input_shape, 
                        tt_output_shape=tt_output_shape,
                        tt_ranks=tt_ranks,
                        activation=LeakyReLU(), 
                        use_bias=True,
                        kernel_regularizer=l2(.001), )(TT_layer_3)

  output_layer = Dense(2,activation ='linear')(TT_layer_4)

  model = Model(inputs=[input_fullres], outputs=[output_layer])
  model.compile(loss='mean_absolute_error', optimizer='adam')

  return model