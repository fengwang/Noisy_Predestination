# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# and https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
#
from instance_normalization import InstanceNormalization
from math import exp, sqrt
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import Subtract
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Subtract
import numpy as np

layer_counter = 0
def unique_name():
    global layer_counter
    layer_counter += 1
    return 'Layer_'+str(layer_counter).zfill(5)

def make_activation( input_layer, with_normalization=True ):
    if with_normalization:
        return LeakyReLU(alpha=0.2, name=unique_name())(InstanceNormalization(name=unique_name())(input_layer))
    return LeakyReLU(alpha=0.2, name=unique_name())(input_layer)

def make_pooling( input_layer, channels, with_normalization=True ):
    x = conv2d_transpose( channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid')( input_layer )
    x = make_activation( x, with_normalization )
    x = conv2d( channels, kernel_size=(3,3), activation='linear', strides=2, padding='valid')( x )
    x = make_activation( x, with_normalization )
    return x

def conv2d_transpose( *args,**kwargs ):
    if 'name' in kwargs:
        return Conv2DTranspose( *args, **kwargs )
    return Conv2DTranspose( *args, **kwargs, name=unique_name() )

def conv2d( *args,**kwargs ):
    if 'name' in kwargs:
        return Conv2D( *args, **kwargs )
    return Conv2D( *args, **kwargs, name=unique_name() )

def make_block( input_layer, channels, kernel_size=(3,3), with_normalization=True ):
    x = input_layer
    x = conv2d_transpose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x, with_normalization )
    x = conv2d( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x, with_normalization )
    return x

def make_output_block( input_layer, output_channels, kernel_size, output_activation ):
    channels = output_channels << 3
    x = input_layer
    x = conv2d_transpose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x )
    x = conv2d( output_channels, kernel_size=kernel_size, activation=output_activation, strides=1, padding='valid')( x )
    return x

def make_upsampling( input_layer, channels ):
    x = conv2d_transpose( channels, kernel_size=(4,4), activation='linear', strides=2, padding='valid')( input_layer )
    x = make_activation( x )
    x = conv2d( channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x )
    return x

def make_xception_blocks( input_layer, channels, kernel_sizes ):
    sub_channels = int( channels/len(kernel_sizes) )
    assert sub_channels * len(kernel_sizes) == channels, 'sub-channels and channels not match, adjust the channels or the size of sub-kernels'
    layer_blocks = []
    for kernel_size in kernel_sizes:
        layer_blocks.append( make_block( input_layer, sub_channels, kernel_size ) )
    return concatenate( layer_blocks )

def add( layers ):
    return Add(name=unique_name())( layers )

def make_blocks( n_blocks, input_layer, channels, kernel_size=(3,3) ):
    x = make_block( input_layer, channels, kernel_size )
    for idx in range( n_blocks ):
        x_ = make_block( x, channels, kernel_size )
        x = add( [x_, x] )
    return x

def make_model( input_channels=1, output_channels=1, transform_repeater=16, output_activation='sigmoid', name=None ):
    input_layer = Input( shape=(None, None, input_channels) )
    #input_layer = Input( shape=(128, 128, input_channels) )

    gt_128 = input_layer
    gt_64 = AveragePooling2D()( gt_128 )
    gt_32 = AveragePooling2D()( gt_64 )
    gt_16 = AveragePooling2D()( gt_32 )
    gt_8  = AveragePooling2D()( gt_16 )

    #def make_block( input_layer, channels, kernel_size=(3,3), with_normalization=True ):
    encoder_128 = make_xception_blocks( make_block( input_layer, 8, with_normalization=False ), 8, (1, 3, 5, 7) )
    encoder_64 = make_xception_blocks( make_block( make_pooling( encoder_128, 16 ), 16 ), 16, (3, 5) )
    encoder_32 = make_xception_blocks( make_block( make_pooling( encoder_64, 32 ), 32 ), 32, (3, 5) )
    encoder_16 = make_xception_blocks( make_block( make_pooling( encoder_32, 64 ), 64 ), 64, (3, 5) )
    encoder_8  = make_xception_blocks( make_block( make_pooling( encoder_16, 64 ), 64 ), 64, (3, 5) )
    encoder_4  = make_pooling( encoder_8, 64 )
    encoder_2  = make_pooling( encoder_2, 128 )
    encoder_1  = make_pooling( encoder_1, 256 )

    compressed_expression = encoder_1
    transformer = make_blocks( transform_repeater, encoder_4, 256 )

    decoder_1  = transformer
    decoder_2  = make_upsampling( decoder_1, 256 )
    decoder_4  = make_upsampling( decoder_2, 128)
    decoder_8  = make_upsampling( decoder_4, 64 )
    decoder_16 = make_xception_blocks( make_block( make_upsampling( decoder_8, 64 ), 64 ), 64, (3, 5) )
    decoder_32 = make_xception_blocks( make_block( make_upsampling( decoder_16, 32 ), 32 ), 32, (3, 5) )
    decoder_64 = make_xception_blocks( make_block( make_upsampling( decoder_32, 16 ), 16 ), 16, (3, 5) )
    decoder_128= make_xception_blocks( make_block( make_upsampling( decoder_64, 8 ), 8 ), 8, (1, 3, 5, 7) )

    output_layer_128 = make_output_block( decoder_128, output_channels, (9, 9), output_activation )
    output_layer_64  = make_output_block(  decoder_64, output_channels, (7, 7), output_activation )
    output_layer_32  = make_output_block(  decoder_32, output_channels, (5, 5), output_activation )
    output_layer_16  = make_output_block(  decoder_16, output_channels, (3, 3), output_activation )
    output_layer_8   = make_output_block(  decoder_8,  output_channels, (3, 3), output_activation )

    should_be_zero_128 = Subtract()( [gt_128, output_layer_128] )
    should_be_zero_64  = Subtract()( [gt_64,  output_layer_64 ] )
    should_be_zero_32  = Subtract()( [gt_32,  output_layer_32 ] )
    should_be_zero_16  = Subtract()( [gt_16,  output_layer_16 ] )
    should_be_zero_8   = Subtract()( [gt_8,   output_layer_8 ] )

    mcnn_model = Model( input_layer, [should_be_zero_128, should_be_zero_64, should_be_zero_32, should_be_zero_16, should_be_zero_8], name='mcnn_model' )
    self2noise_model = Model( input_layer, output_layer_128, name='self2noise_model' )

    return mcnn_model, self2noise_model


from tensorflow.keras.utils import plot_model
import imageio
import numpy as np

if __name__ == '__main__':
    mcnn, s2n = make_model()
    s2n.summary()
    mcnn.summary()



