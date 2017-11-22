""" preprocess beforehand to store on cloud to manage memory better during training """

import tensorflow as tf
import numpy as np

def resizer(arrays, size, method):
    return tf.map_fn(lambda array: 
                     tf.image.resize_images(array,
                                            [size, size],
                                            method=method), 
                     arrays)

def get_input_shape(data):
    num_samples = data.shape[0]
    channels = 3
    img_rows = data.shape[2]
    img_cols = data.shape[3]
    return (num_samples, img_rows, img_cols, channels)

def new_res(arrays, size, method=tf.image.ResizeMethod.BILINEAR):
    reshaped_arrays = reshape(arrays)
    with tf.Session() as session:
        resized_array = resizer(reshaped_arrays, size, method).eval()
    return resized_array

def reshape(data):
    return np.reshape(data, get_input_shape(data))

images = np.load('AllImages.npy')

res64 = new_res(images, 64)

res64_reshape = np.reshape(res64, [len(res64), 3, 64, 64])

np.save('AllImages64.npy', res64_reshape)

res32 = new_res(images, 32)

res32_reshape = np.reshape(res32, [len(res32), 3, 32, 32])

np.save('AllImages32.npy', res32_reshape)
