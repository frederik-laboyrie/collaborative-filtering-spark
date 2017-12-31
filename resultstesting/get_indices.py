import numpy as np
import h5py
import keras
from keras.models import load_model
import pickle

DATA_PATH = '/home/frederik/gitdisst/hand-orientation-inference/modelling/'
MODEL_PATH = '/home/frederik/gitdisst/hand-orientation-inference/models/'

def labels_to_angles(labels):
    labels_angles = (labels * 180 / np.pi) - 90
    return labels_angles

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

def get_best_and_worst_indices(errors, output_name, good_threshold=3, poor_threshold=20):
    under_indices_elevation = np.where(errors[:,0] < good_threshold)
    under_indices_zenith = np.where(errors[:,1] < good_threshold)
    over_indices_elevation = np.where(errors[:,0] > poor_threshold)
    over_indices_zenith = np.where(errors[:,1] < poor_threshold)
    return {output_name: [{'good_elev':under_indices_elevation},
                          {'good_zen':under_indices_zenith},
                          {'bad_elev':over_indices_elevation},
                          {'bad_zen':over_indices_zenith}]}


full_res = np.load(DATA_PATH + 'AllBW.npy')
med_res = np.load(DATA_PATH + 'AllImagesBW64.npy')
low_res = np.load(DATA_PATH + 'AllImagesBW32.npy')
labels = np.load(DATA_PATH + 'AllAngles.npy')

good_model = load_model(MODEL_PATH + 'vanilla_15.h5')
bad_model = load_model(MODEL_PATH + 'vanilla_15_bad.h5')

full, med, low = standardise_and_reshape_data(full_res, med_res, low_res)

test_index = int(len(full_res) * 0.8)
full = full[test_index:]
med = med[test_index:]
low = low[test_index:]
labels = labels[test_index:]

good_predictions = good_model.predict([full, med, low])
bad_predictions = bad_model.predict([full, med, low])

good_angles = (good_predictions * 90) - 45
bad_angles = (bad_predictions * 90) - 45
true_angles = labels_to_angles(labels)

good_error = abs(true_angles - good_angles)
bad_error = abs(true_angles - bad_angles)


good_vanilla_indices = get_best_and_worst_indices(good_error, output_name='vanilla_good',
                                                  good_threshold=3, poor_threshold=20)
# not really relevant because constant predictor
bad_vanilla_indices = get_best_and_worst_indices(bad_error, output_name='vanilla_bad',
                                                  good_threshold=3, poor_threshold=20)

with open('good_vanilla_indices.pickle', 'wb') as handle:
    pickle.dump(good_vanilla_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bad_vanilla_indices.pickle', 'wb') as handle:
    pickle.dump(bad_vanilla_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)



