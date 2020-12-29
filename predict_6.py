import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tifffile
from instance_normalization import InstanceNormalization
from keras_utils import read_model
model = read_model( '/raid/feng/cache/self2noise/model_noisy_to_clean_model_6/s2n_6_147' )

image_path_0 = '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_large.tif_part_0.tif'
noisy_images_0 = tifffile.imread( image_path_0 )
print( f'loaded {image_path_0}' )

image_path_1 = '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_large.tif_part_1.tif'
noisy_images_1 = tifffile.imread( image_path_1 )
print( f'loaded {image_path_1}' )

image_path_2 = '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_large.tif_part_2.tif'
noisy_images_2 = tifffile.imread( image_path_2 )
print( f'loaded {image_path_2}' )

image_path_3 = '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_large.tif_part_3.tif'
noisy_images_3 = tifffile.imread( image_path_3 )
print( f'loaded {image_path_3}' )

noisy_images_ = np.concatenate( (noisy_images_0, noisy_images_1, noisy_images_2, noisy_images_3), axis=0 )
print( 'merged' )
#tifffile.imsave( '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_largee.tif', noisy_images_, compress=6 )
#print( 'merged tif saved' )

noisy_images = noisy_images_.reshape( noisy_images_.shape + (1,) )
noisy_images = noisy_images / (np.amax( noisy_images ) + 1.0e-10)

predictions = model.predict( noisy_images, batch_size=1, verbose=1 )
predictions = np.squeeze( predictions )
predictions = np.asarray( predictions * 65535.0, dtype='uint16' )
print( 'prediction finished' )

tifffile.imsave( '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_large_denoised.tif', predictions, compress=6 )

from message import send_message

send_message( 'prediction 6 done' )
