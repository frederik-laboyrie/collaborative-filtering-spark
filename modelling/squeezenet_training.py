from data_preprocessing import load_standardized_multires, load_multires
from squeezenet_models import squeezenet, multires_squeezenet
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
from math import pi
from numpy import mean, std
from data_preprocessing import radian_to_angle, mean_std_norm, reverse_mean_std, train_test_split


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
       and tests angle loss
    '''
    multires_data, labels = load_multires()
    multires_data = [x.astype('float32') for x in multires_data]
    multires_data = [x / 255 for x in multires_data]
    model = multires_squeezenet(multires_data, True)
    full = multires_data[0]
    med = multires_data[1]
    low = multires_data[2]
    train_full, test_full = train_test_split(full)
    train_med, test_med = train_test_split(med)
    train_low, test_low = train_test_split(low)
    labels_angles = radian_to_angle(labels)
    train_orig_lab, test_orig_lab = train_test_split(labels_angles)
    labels_standardised, mean_, std_ = mean_std_norm(labels_angles)
    train_labels, test_labels = train_test_split(labels_standardised)
    model.fit_generator(multiinput_generator(train_full, train_med, train_low, train_labels),
                        steps_per_epoch=32,
                        epochs=1)
    std_angles = model.predict([test_full, test_med, test_low])
    unstd_angles = reverse_mean_std(std_angles, mean_, std_)
    error = unstd_angles - test_labels
    mean_error_elevation = np.mean(abs(error[:, 0]))
    mean_error_zenith = np.mean(abs(error[:, 1]))
    return mean_error_elevation, mean_error_zenith


def generator_main_without_test():
    '''main entry point
       calls customised  multiinput generator
    '''
    multires_data, labels = load_multires()
    multires_data = [x.astype('float32') for x in multires_data]
    multires_data = [x / 255 for x in multires_data]
    model = multires_squeezenet(multires_data, True)
    full = multires_data[0],
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
