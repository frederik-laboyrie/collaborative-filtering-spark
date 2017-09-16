from data_preprocessing import load_standardized_multires
from mobilenet_models import mobilenet, multires_mobilenet
from keras.layers.advanced_activations import LeakyReLU

import sys

def main():
    multires_data, labels = load_standardized_multires()
    model = multires_mobilenet(multires_data)
    full = multires_data[0]
    med = multires_data[1]
    low = multires_data[2]
    model.fit([full,med,low],labels, epochs = 10)

if __name__ == '__main__':
    main()