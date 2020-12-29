import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tqdm
import copy

training_index = 5

from message import send_message
from message import send_photo
send_message( f'self2noise training {training_index} started' )

from models_5 import make_model
# mcnn : input - noisy, outpus: [zero_128, zero_64, zero_32, zero_16]
mcnn, s2n = make_model()

from tensorflow.keras.optimizers import RMSprop
mcnn.compile(loss='mae', optimizer=RMSprop())

model_directory = f'/raid/feng/cache/self2noise/model_noisy_to_clean_model_{training_index}'
if not os.path.exists(model_directory):
    os.mkdir(model_directory)

from skimage import io
import numpy as np
import tifffile
import imageio

image_path_0 = '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_large.tif_part_0.tif'
noisy_images_0 = tifffile.imread( image_path_0 )
image_path_1 = '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_large.tif_part_1.tif'
noisy_images_1 = tifffile.imread( image_path_1 )
image_path_2 = '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_large.tif_part_2.tif'
noisy_images_2 = tifffile.imread( image_path_2 )
image_path_3 = '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_large.tif_part_3.tif'
noisy_images_3 = tifffile.imread( image_path_3 )

noisy_images_ = np.concatenate( (noisy_images_0, noisy_images_1, noisy_images_2, noisy_images_3), axis=0 )

send_message( f'noisy image of shape {noisy_images_.shape} loaded' )
n, row, col = noisy_images_.shape
dim_scaler = 128
factor = int( row / dim_scaler )
noisy_images = np.zeros( (n*factor*factor, dim_scaler, dim_scaler), dtype=noisy_images_.dtype )
for r in range( factor ):
    for c in range( factor ):
        offset = r * factor + c
        noisy_images[offset*n:(1+offset)*n] = noisy_images_[:,r*dim_scaler:(r+1)*dim_scaler, c*dim_scaler:(c+1)*dim_scaler]
print( 'noisy image converted' )

n_noisy_images, *_ = noisy_images.shape
print( f'{n_noisy_images} noisy images loaded' )

import numpy as np
noisy_images = np.asarray( noisy_images, dtype='float32' ) / (np.amax( noisy_images ) + 1.0e-10)
noisy_images = noisy_images.reshape( noisy_images.shape + (1,) )

batch_size = 128
zeros = [np.zeros((batch_size, dim_scaler, dim_scaler, 1)), np.zeros((batch_size, dim_scaler//2, dim_scaler//2, 1)), np.zeros((batch_size, dim_scaler//4, dim_scaler//4, 1)), np.zeros((batch_size, dim_scaler//8, dim_scaler//8, 1)), np.zeros((batch_size, dim_scaler//16, dim_scaler//16, 1))]

n_loops = 192*32*32
check_intervals = 1

from keras_utils import read_model
from keras_utils import write_model
import tifffile

test_image = np.squeeze( noisy_images_[0] )
imageio.imsave( './self2noise_test.png', np.squeeze(test_image) )
send_photo( './self2noise_test.png' )

test_image = test_image.reshape( (1,) + test_image.shape + (1,) )
test_image = test_image / (np.amax(test_image) + 1.0e-10)

current_losses = None
for loop in range( n_loops ):

    for idx in range( int(n_noisy_images/batch_size) ):
        input = noisy_images[idx*batch_size:(idx+1)*batch_size]
        outputs = zeros
        current_losses = mcnn.train_on_batch( input, outputs )
        print( f'self2noise {training_index} --> {loop}/{n_loops} with minibatch {idx*batch_size}/{n_noisy_images}, losses: {current_losses}', end='\r')

    if loop != 0 and loop % check_intervals == 0:
        send_message( f'self2noise denoising of {training_index}: {loop+1}/{n_loops} done, last loss is {current_losses}.' )
        model_directory = f'/raid/feng/cache/self2noise/model_noisy_to_clean_model_{training_index}'
        write_model( f'{model_directory}/s2n_{training_index}_{loop}', s2n )

        translated_image = np.squeeze( s2n.predict( test_image ) )
        imageio.imsave( f'./self2noise_denoised_{training_index}_{loop}.png', np.squeeze(translated_image) )
        send_photo( f'./self2noise_denoised_{training_index}_{loop}.png' )

