import numpy as np
import h5py
import tensorflow as tf
from sklearn.cluster import KMeans


def unpack_mat(filename):
    f = h5py.File(filename, 'r')
    arrays = {}
    for k, v in f.items():
        arrays[k] = np.array(v)
    values = arrays[filename[:-4]]
    return values


def get_input_shape(data):
    num_samples = data.shape[0]
    channels = 3
    img_rows = data.shape[2]
    img_cols = data.shape[3]
    return (num_samples, img_rows, img_cols, channels)


def reshape(data):
    return np.reshape(data, get_input_shape(data))


def subsample(data, labels, nb_samples):
    return data[:nb_samples], labels[:nb_samples]


def per_image_standardization(arrays):
    '''only works for 1 call before
       ValueError tf graph > 2gb.
    '''
    sess = tf.InteractiveSession()
    standardized_tensors = tf.map_fn(lambda array: 
                                     tf.image.per_image_standardization(array), 
                                     arrays)
    standardized_images = standardized_tensors.eval()
    return standardized_images


def resizer(arrays, size, method):
    return tf.map_fn(lambda array: 
                     tf.image.resize_images(array,
                                            [size, size],
                                            method=method), 
                     arrays)


def singleres_to_multires(arrays, size1=64, size2=32, 
                          method=tf.image.ResizeMethod.BILINEAR):
    with tf.Session() as session:
        size1_arrays = resizer(arrays, size1, method).eval()
        size2_arrays = resizer(arrays, size2, method).eval()
    return [arrays, size1_arrays, size2_arrays]


def load_data():
    images = np.load('AllImages.npy')
    labels = np.load('AllAngles.npy')
    return images, labels


def load_standardized_singleres():
    '''temporarily subsampling within here
       as tf graph > 2gb issue workaround
    '''
    images, labels = load_data()
    images_reshape = reshape(images)
    images_ss, labels_ss = subsample(images_reshape, labels, 500)
    standardized_images = per_image_standardization(images_ss)
    return standardized_images, labels_ss


def load_multires():
    images, labels = load_data()
    images_reshape = reshape(images)
    images_ss, labels_ss = subsample(images_reshape, labels, 2000)
    multires_images = singleres_to_multires(images_ss)
    return multires_images, labels_ss


def load_standardized_multires():
    standardized_images, labels = load_standardized_singleres()
    multires_images = singleres_to_multires(standardized_images)
    return multires_images, labels


def get_buckets_kmeans(n_clusters, labels_file='AllAngles.npy'):
    labels = np.load(labels_file)
    clusters = KMeans(n_clusters=n_clusters, init='k-means++',
                      n_init=10, max_iter=300, tol=0.0001).fit(labels)
    return clusters.labels_

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


def train_test_split(array, proportion=0.8):
    '''non randomised train split
    '''
    index = int(len(array) * proportion)
    train = array[:index]
    test = array[index:]
    return train, test
