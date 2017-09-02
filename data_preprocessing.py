import numpy as np
import h5py
import tensorflow as tf

def unpack_mat(filename):
    f = h5py.File(filename,'r')
    arrays={}
    for k, v in f.items():
         arrays[k] = np.array(v)
    values = arrays[filename[:-4]]
    return values

def per_image_standardization(arrays):
    '''only works for 1 call before
       ValueError tf graph > 2gb.
       make work around to clear graph
    '''
    sess = tf.InteractiveSession()
    standardized_tensors = tf.map_fn(lambda array: 
                                    tf.image.per_image_standardization(array), 
                                    arrays)
    standardized_images = standardized_tensors.eval()
    return standardized_images

def load_data(images='AllImages.Mat',labels='AllLabels.Mat'):
	images = unpack_mat(images)
	labels = unpack_mat(labels)
	standardized_images = per_image_standardization(images)
	return standardized_images, labels


