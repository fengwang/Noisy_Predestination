import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

from instance_normalization import InstanceNormalization

from keras_utils import read_model
model_path = '/raid/feng/cache/self2noise/model_noisy_to_clean_model_2/s2n_2_128'
model = read_model(model_path)

import numpy as np
import tifffile
image_path = '/raid/feng/experimental_data/experimental/s9.tif'
noisy_images = tifffile.imread( image_path )
noisy_images = np.reshape(noisy_images, noisy_images.shape+(1,))
noisy_images = noisy_images / ( np.amax(noisy_images) + 1.0e-10 )
results = model.predict(noisy_images, batch_size=1, verbose=1)
results = np.squeeze( results )
results = results / np.amax(results)
results = np.asarray(results*65535.0, dtype='uint16')

tifffile.imsave( f'{image_path}_ssd_result.tif', results )

