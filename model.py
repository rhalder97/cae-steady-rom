import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
Input, Dense, Conv2D, Activation, MaxPool2D,
Flatten, Reshape, Conv2DTranspose, LeakyReLU, ReLU)

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()

    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential([
      Input(shape=(64, 64, 2), name="inputs"),
      Conv2D(64, (3, 3), padding="same", activation=LeakyReLU(alpha=0.25),
            kernel_initializer=HeNormal()),
      MaxPool2D((2, 2)),
      Conv2D(32, (3, 3), padding="same", activation=LeakyReLU(alpha=0.25),
            kernel_initializer=HeNormal()),
      MaxPool2D((2, 2)),
      Flatten(),
      Dense(128, activation=LeakyReLU(alpha=0.25), kernel_initializer=HeNormal()),
      Dense(latent_dim, name="latent", activation=LeakyReLU(alpha=0.25),
            kernel_initializer=HeNormal())
    ])

    self.decoder = tf.keras.Sequential([
      Dense(128, activation=LeakyReLU(alpha=0.25), kernel_initializer=HeNormal()),
      Dense(8192, activation=LeakyReLU(alpha=0.25), kernel_initializer=HeNormal()),
      Reshape((16, 16, 32)),
      Conv2DTranspose(32, (3, 3), strides=2, padding="same",
            activation=LeakyReLU(alpha=0.25), kernel_initializer=HeNormal()),
      Conv2DTranspose(64, (3, 3), strides=2, padding="same",
            activation=LeakyReLU(alpha=0.25), kernel_initializer=HeNormal()),
      Conv2DTranspose(2, (3, 3), strides=1, padding="same",
            activation='sigmoid', name="outputs"),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
