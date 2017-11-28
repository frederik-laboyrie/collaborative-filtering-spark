""" preprocess beforehand to store on cloud to manage memory better during training """
import numpy as np
from scipy.misc import imresize

images = np.load('AllBW.npy')

images64 = np.array([imresize(image, (64, 64)) for image in images])
images32 = np.array([imresize(image, (32, 32)) for image in images])

np.save('AllImagesBW64.npy', images64)
np.save('AllImagesBW32.npy', images32)
