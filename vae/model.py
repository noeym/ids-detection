import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datagenerator import *


class VAE:
    def __init__(self, dim, batch_size, epochs=100):
        self.original_dim = dim
        self.input_shape = (dim,)
        self.batch_size = batch_size
        self.epochs = epochs
        self.intermediate_dim = 32
        self.latent_dim = 8

        self.create_model(
            self.original_dim,
            self.input_shape,
            self.intermediate_dim,
            self.latent_dim,
        )

    def create_model(
        self,
        original_dim,
        input_shape,
        intermediate_dim,
        latent_dim,
    ):
        def Sampling(args):
            z_mean, z_log_var = args
            batch = tf.keras.backend.shape(z_mean)[0]
            dim = tf.keras.backend.int_shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

        def train_loss(inputs, outputs):
            reconstruction_loss = tf.keras.backend.mean(
                (inputs - outputs) ** 2, axis=-1
            )
            reconstruction_loss *= original_dim
            kl_loss = (
                1
                + z_log_var
                - tf.keras.backend.square(z_mean)
                - tf.keras.backend.exp(z_log_var)
            )
            kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
            return vae_loss

        # Encoder
        inputs = tf.keras.layers.Input(shape=input_shape, name="encoder_input")
        # labels = tf.keras.layers.Input(shape=(1,), name="label_input")
        x1 = tf.keras.layers.Dense(
            intermediate_dim,
            activation="relu",
            name="x1",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            bias_regularizer=tf.keras.regularizers.l2(1e-4),
        )(inputs)
        z_mean = tf.keras.layers.Dense(
            latent_dim,
            name="z_mean",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            bias_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x1)
        z_log_var = tf.keras.layers.Dense(
            latent_dim,
            name="z_log_var",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            bias_regularizer=tf.keras.regularizers.l2(1e-4),
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )(x1)
        z = tf.keras.layers.Lambda(Sampling, output_shape=(latent_dim,), name="z")(
            [z_mean, z_log_var]
        )
        self.encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name="z_sampling")
        x2 = tf.keras.layers.Dense(
            intermediate_dim,
            activation="relu",
            name="x2",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            bias_regularizer=tf.keras.regularizers.l2(1e-4),
        )(latent_inputs)
        outputs = tf.keras.layers.Dense(
            original_dim,
            # activation="relu",
            name="decoder_output",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            bias_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x2)
        self.decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")

        # VAE Model
        # outputs = self.decoder(self.encoder(inputs)[2])
        # self.vae = tf.keras.Model(inputs, outputs, name="vae")
        # self.vae = tf.keras.Model([inputs, labels], outputs, name="vae")

        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = tf.keras.Model(inputs, outputs, name="vae")
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.vae.compile(optimizer=optimizer, loss=train_loss)

    def train(self, data, train_gen, valid_gen):
        # callbacks = [
        #     tf.keras.callbacks.ModelCheckpoint(
        #         filepath="model.h5",
        #         monitor="val_loss",
        #         verbose=1,
        #         mode="auto",
        #         save_weights_only=True,
        #         save_best_only=True,
        #     ),
        # ]
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=5,
                verbose=1,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
            ),
        ]
        # callbacks = [
        #     tf.keras.callbacks.ReduceLROnPlateau(
        #         monitor="val_loss",
        #         factor=0.2,
        #         patience=5,
        #         verbose=1,
        #         mode="auto",
        #         min_lr=0,
        #         restore_best_weights=True,
        #     ),
        # ]
        self.history = self.vae.fit(
            train_gen,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_gen,
            shuffle=True,
        )
        print("Complete traing vae")

    def calculateAnomaly(self, data, predict_gen1, predict_gen2):
        def test_loss(inputss, outputss, mean, log_var):
            reconstruction_loss = np.mean((inputss - outputss) ** 2, axis=-1)
            reconstruction_loss *= self.original_dim
            kl_losses = 1 + log_var - np.square(mean) - np.exp(log_var)
            kl_losses = np.sum(kl_losses, axis=-1)
            kl_losses *= -0.5
            vae_losses = reconstruction_loss + kl_losses
            return vae_losses

        estimated_vae_output = self.vae.predict(predict_gen1)
        mean_latent, std_latent, _ = self.encoder.predict(predict_gen2)
        return test_loss(data, estimated_vae_output, mean_latent, std_latent)

    def save_model(self):
        self.vae.save_weights("model.h5")
        print("Saved model")

    def load_model(self):
        self.vae.load_weights("model.h5")
