import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tqdm
import copy

training_index = 7

from message import send_message
from message import send_photo
send_message( f'self2noise training {training_index} started' )

from models_3 import make_model
# mcnn : input - noisy, outpus: [zero_128, zero_64, zero_32, zero_16]
mcnn, s2n = make_model()

from tensorflow.keras.optimizers import RMSprop
mcnn.compile(loss='mae', optimizer=RMSprop(lr=0.001))

model_directory = f'/raid/feng/cache/self2noise/model_noisy_to_clean_model_{training_index}'
if not os.path.exists(model_directory):
    os.mkdir(model_directory)

from skimage import io
import numpy as np
import tifffile
import imageio
#image_path = '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/2019_05_08-15.32cde,2048px,6.3pm(astigmatic).tif'
image_path = '/raid/feng/noise2void/TwoPhoton_BPAE_G.raw.tif'
noisy_images_ = tifffile.imread( image_path )
noisy_images_ = np.sqrt( noisy_images_ ) # enhance
send_message( f'noisy image of shape {noisy_images_.shape} loaded' )
n, row, col = noisy_images_.shape
factor = int( row / 128 )
noisy_images = np.zeros( (n*factor*factor, 128, 128), dtype=noisy_images_.dtype )
for r in range( factor ):
    for c in range( factor ):
        offset = r * factor + c
        noisy_images[offset*n:(1+offset)*n] = noisy_images_[:,r*128:(r+1)*128, c*128:(c+1)*128]
print( 'noisy image converted' )

n_noisy_images, *_ = noisy_images.shape
print( f'{n_noisy_images} noisy images loaded' )

import numpy as np
noisy_images = np.asarray( noisy_images, dtype='float32' ) / (np.amax( noisy_images ) + 1.0e-10)
noisy_images = noisy_images.reshape( noisy_images.shape + (1,) )

batch_size = 128
zeros = [np.zeros((batch_size, 128, 128, 1)), np.zeros((batch_size, 64, 64, 1)), np.zeros((batch_size, 32, 32, 1)), np.zeros((batch_size, 16, 16, 1)), np.zeros((batch_size, 8, 8, 1))]

n_loops = 192*32*32*32
check_intervals = 128*4

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

        translated_image = np.squeeze( s2n.predict( test_image ) )
        imageio.imsave( f'./self2noise_denoised_{training_index}_{loop}.png', np.squeeze(translated_image) )
        send_photo( f'./self2noise_denoised_{training_index}_{loop}.png' )

        model_directory = f'/raid/feng/cache/self2noise/model_noisy_to_clean_model_{training_index}'
        write_model( f'{model_directory}/s2n_{training_index}_{loop}', s2n )

