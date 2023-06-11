import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class BIGAN:
    def __init__(self, dim, batch_size=100, epochs=100):
        self.original_dim = dim
        self.input_shape = (dim,)
        self.batch_size = batch_size
        self.epochs = epochs
        self.dense_dim = int(dim / 2)
        self.latent_dim = int(dim / 4)

    def build_discriminator(self):
        z = tf.keras.layers.Input(shape=(self.latent_dim,))

    def save_model(self, name):
        self.vae.save_weights(name)
        print("Saved model")

    def load_model(self, name):
        self.vae.load_weights(name)
