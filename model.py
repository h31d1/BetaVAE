from tkinter import Y
import tensorflow as tf 
from tensorflow import keras
from keras import Model
from keras.layers import (Input, Conv2D, Conv2DTranspose, 
                          BatchNormalization, Flatten, Dense, 
                          ReLU, Reshape, Layer)
from keras.metrics import Mean
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def Encoder(filters, latent_dim):
    encoder_inputs = Input(shape=(28, 28, 1), name='encoder_input')
    x = encoder_inputs
    for i in range(len(filters)):
        x = Conv2D(filters[i], 3, activation="relu", strides=2, padding="same", name=f'conv_layer_{i+1}')(x)
        y = BatchNormalization(name=f'conv_norm_{i+1}')(x)
        x = ReLU(name=f'conv_relu_{i+1}')(y)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def Decoder(latent_dim, filters, last_convdim):
    F = len(filters)
    flat_dim = last_convdim * last_convdim * filters[-1]
    latent_inputs = Input(shape=(latent_dim,), name='latent_input')
    x = Dense(flat_dim, activation="relu")(latent_inputs)
    x = Reshape((last_convdim, last_convdim, -1))(x)
    for i in range(F-1,0,-1):
        x = Conv2DTranspose(filters[i], 3, activation="relu", strides=2, padding="same", name=f'deconv2d_{F-i}')(x)
        y = BatchNormalization(name=f'deconv_norm_{F-i}')(x)
        x = ReLU(name=f'deconv_relu_{F-i}')(y)
    x = Conv2DTranspose(filters[-1], 3, activation="relu", strides=2, padding="same", name=f'deconv2d_{F}')(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same", name=f'sigmoid')(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder
         
    

class VAE(Model):
    
    def __init__(self, encoder, decoder, beta=1., **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        """ Metrics trackers """
        self.total_loss_tracker = Mean(name="Total_loss")
        self.rec_loss_tracker = Mean(name="Reconstruction_loss")
        self.kl_loss_tracker = Mean(name="KL_loss")
    
    @property
    def metrics(self):
        super().metrics
        return [self.total_loss_tracker,
                self.rec_loss_tracker,
                self.kl_loss_tracker]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            """ Training """
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            rec_loss = keras.losses.binary_crossentropy(data, reconstruction)
            rec_loss = tf.reduce_mean(tf.reduce_sum(rec_loss, axis=(1,2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = rec_loss + self.beta * kl_loss
        """ Gradients """
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        """ Update tracker """
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"Total_loss": self.total_loss_tracker.result(),
                "Reconstruction_loss": self.rec_loss_tracker.result(),
                "KL_loss": self.kl_loss_tracker.result(), }


def plot_latent_space(vae, n=30, figsize=15, digit_size = 28):
    """
    Display a n*n 2D manifold of digits
    linearly spaced coordinates corresponding to the 2D plot
    of digit classes in the latent space
    """
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


def plot_label_clusters(vae, data, labels):
    """
    Display a 2D plot of the digit classes in the latent space
    """
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def generate_images(vae_model, image, latent_dim, lat_feature=0, scale=3., N=15): 
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
    fig, axs = plt.subplots(nrows=1, ncols=16, figsize=(16, 1),
                            subplot_kw={'xticks': [], 'yticks': []})
    for i,ax in enumerate(axs.flat):
        if i==0: ax.imshow(image[0,:,:,:])
        else: ax.imshow(dec_imgs[i-1,:,:,:])
    plt.tight_layout()
    plt.show()