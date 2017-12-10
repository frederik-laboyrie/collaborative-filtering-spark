import numpy as np
import h5py
import keras
from keras.models import load_model
import time

DATA_PATH = '/home/frederik/gitdisst/hand-orientation-inference/modelling/'
MODEL_PATH = '/home/frederik/gitdisst/hand-orientation-inference/models/'
SAMPLE_SIZE = 1000

def standardise_and_reshape_data(full, med, low):
    full = np.reshape(full, [len(full), 128, 128, 1])
    med = np.reshape(med, [len(med), 64, 64, 1])
    low = np.reshape(low, [len(low), 32, 32, 1])
    full = [x.astype('float32') for x in full]
    full = np.array([(x / 255) for x in full])
    med = [x.astype('float32') for x in med]
    med = np.array([x / 255 for x in med])
    low = [x.astype('float32') for x in low]
    low = np.array([x / 255 for x in low])
    return full, med, low

def subsample(array, sample_size=SAMPLE_SIZE):
	return array[:sample_size]

def time_results_serving(full, med, low, model):
	start = time.time()
	good_predictions = model.predict([full, med, low])
	end = time.time()
	return end - start


full = subsample(np.load(DATA_PATH + 'AllBW.npy'))
med = subsample(np.load(DATA_PATH + 'AllImagesBW64.npy'))
low = subsample(np.load(DATA_PATH + 'AllImagesBW32.npy'))

full, med, low = standardise_and_reshape_data(full, med, low)

model = load_model(MODEL_PATH + 'vanilla_15.h5')

time = time_results_serving(full, med, low, model)

examples_per_second = SAMPLE_SIZE / time

model.summary()