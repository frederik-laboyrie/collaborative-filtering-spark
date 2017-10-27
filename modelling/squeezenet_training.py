from data_preprocessing import load_standardized_multires, load_multires
from squeezenet_models import squeezenet, multires_squeezenet
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
from math import pi
from numpy import mean, std

def radian_to_angle(radian_array):
    '''converts original radian to angle which
       will be error metric
    '''
    return (radian_array * 180 / pi) - 90 

def mean_std_norm(array):
    '''standardization for labels
    '''
    mean_ = mean(array)
    std_ = std(array)
    standardized = (array - mean_) / std_
    return standardized, mean_, std_

def reverse_mean_std(standardized_array, prev_mean, prev_std):
    '''undo transformation in order to calculate
       angle loss
    '''
    de_std = standardized_array * prev_std
    de_mean = de_std + prev_mean
    return de_mean

def multiinput_generator(full, med, low, label):
    '''custom generator to be passed to main training
       note samplewise std normalization + batch size
    '''
    while True:
        # shuffled indices
        idx = np.random.permutation(full.shape[0])
        # create image generator
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=True,  # divide each input by its std
                zca_whitening=False)  # randomly flip images
        batches = datagen.flow(full[idx], label[idx], batch_size=32, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            yield [batch[0], med[idx[idx0:idx1]], low[idx[idx0:idx1]]], batch[1]
            idx0 = idx1
            if idx1 >= full.shape[0]:
                break


def generator_main():
    '''main entry point
       calls customised  multiinput generator
    '''
    multires_data, labels = load_multires()
    multires_data = [x.astype('float32') for x in multires_data]
    multires_data = [x / 255 for x in multires_data]
    model = multires_squeezenet(multires_data, True)
    full = multires_data[0]
    med = multires_data[1]
    low = multires_data[2]
    model.fit_generator(multiinput_generator(full, med, low, labels),
                        steps_per_epoch=32,
                        epochs=10)


def main():
    '''previous non generator
       remaining here in case needed later
    '''
    multires_data, labels = load_standardized_multires()
    model = multires_squeezenet(multires_data, True)
    full = multires_data[0]
    med = multires_data[1]
    low = multires_data[2]
    model.fit([full, med, low], labels, epochs=10)

if __name__ == '__main__':
    generator_main()
