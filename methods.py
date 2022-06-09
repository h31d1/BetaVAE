
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
from matplotlib import pyplot as plt
import model
from model import *


def preprocessor(ds, height, width, batch_size):
    """ 
    Preprocess images 
    arguments: 
        ds: tensorflow dataset
        height: target height of the image
        width: target width of the image
        batch_size: batch size of the final dataset
    implementation:
        - fix pixel values
        - resize images
        - turn colors to black and white
        - normalize pixels from range (0,255) to range (0,1)
    return:
        shuffled dataset with batches
    """
    ds = ds.map(lambda x: tf.clip_by_value(x,0,255))
    resized_ds = ds.map(lambda x: (tf.image.resize(x, (height, width))))
    final_bw_ds = resized_ds.map(lambda x: (tf.image.rgb_to_grayscale(x)))
    norm_bw_ds = final_bw_ds.map(lambda x: x / 255.)
    dataset = norm_bw_ds.shuffle(100).batch(batch_size).prefetch(buffer_size=10)
    return dataset


def show_reconstr_hor(image,vae_model):
    """
    Draw initial image and its reconstruction
    Arguments:
        image: initial image
        model: betavae or vae model which brings image into latent space and then reconstructs it
    """
    img = np.expand_dims(image, 0)
    z_mean,_,_ = vae_model.encoder.predict(img)
    prediction = vae_model.decoder.predict(z_mean)
    plt.rcParams['font.size'] = 15
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    bef_n_aft = [image,prediction[0,:,:,0]]
    titles = ['Initial image','Encoded-Decoded image']
    for i,ax in enumerate(axs.flat):
        ax.imshow(bef_n_aft[i],cmap='binary_r')
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def generate_images2(vae_model, image, latent_dim, lat_feature=0, scale=3., N=15): 
    """ image.shape = (batch, width, height, depth) = (1, 28, 28, 1)  """
    """ Encode image """
    z_mean,_,_ = vae_model.encoder.predict(image)
    """ Sample given feature """
    z_sample = z_mean.copy()
    grid_x = np.linspace(-scale, scale, N)
    samples = []
    for n in range(N):
        z_sample[0,lat_feature] = grid_x[n]
        samples.append(z_sample)
    samples = np.asarray(samples).reshape(N,latent_dim)
    """ Generate new images """
    dec_imgs = vae_model.decoder.predict(samples)
    """ Plot images """
    fig, axs = plt.subplots(nrows=1, ncols=N+1, figsize=(16, 4),
                            subplot_kw={'xticks': [], 'yticks': []})
    for i,ax in enumerate(axs.flat):
        if i==0: ax.imshow(image[0,:,:,:],cmap='gray')
        else: ax.imshow(dec_imgs[i-1,:,:,:],cmap='gray')
    plt.tight_layout()
    plt.show()