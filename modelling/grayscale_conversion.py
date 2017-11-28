import numpy as np

rgb = np.load('AllImages.npy')
rgb_rolled = np.array([np.rollaxis(image, 0, 3) for image in rgb])
 
bw = []
count = 0
for image in rgb_rolled:
    grey = np.zeros((image.shape[0], image.shape[1]))  # init 2D numpy array
    # get row number
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
            grey[rownum][colnum] = np.average(image[rownum][colnum])
    bw.append(np.array(grey))
    count += 1
    print(count)

bw = np.array(bw)
np.save('AllBW.npy', bw)
