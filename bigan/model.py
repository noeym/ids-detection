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
        self.create_model()

    def create_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=["binary_crossentropy"], optimizer=optimizer, metrics=["accuracy"]
        )
        self.generator = self.build_generator()
        self.encoder = self.build_encoder()
        self.discriminator.trainable = False
        z = tf.keras.layers.Input(shape=(self.latent_dim,), name="inputnoise")
        input_data_ = self.generator(z)
        input_data = tf.keras.layers.Input(shape=self.input_shape, name="inputdata")
        z_ = self.encoder(input_data)

        fake = self.discriminator([z, input_data_])
        valid = self.discriminator([z_, input_data])

        self.bigan_generator = tf.keras.Model(
            [z, input_data], [fake, valid], name="bigan"
        )
        self.bigan_generator.compile(
            loss=["binary_crossentropy", "binary_crossentropy"], optimizer=optimizer
        )

    def build_encoder(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=self.input_shape))
        model.add(tf.keras.layers.Dense(self.dense_dim))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(self.latent_dim))

        input_data = tf.keras.layers.Input(shape=self.input_shape)
        z = model(input_data)

        return tf.keras.Model(input_data, z, name="encoder")

    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.dense_dim, input_dim=self.latent_dim))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(
            tf.keras.layers.Dense(np.prod(self.input_shape), activation="sigmoid")
        )
        model.add(tf.keras.layers.Reshape(self.input_shape))

        z = tf.keras.layers.Input(shape=(self.latent_dim,))
        input_data_ = model(z)

        return tf.keras.Model(z, input_data_, name="generator")

    def build_discriminator(self):
        x = tf.keras.layers.Input(shape=self.input_shape, name="inputdata")
        x_ = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x_ = tf.keras.layers.Dropout(0.5)(x)

        z = tf.keras.layers.Input(shape=self.latent_dim, name="latent_z")
        z_ = tf.keras.layers.LeakyReLU(alpha=0.2)(z)
        z_ = tf.keras.layers.Dropout(0.5)(z)

        d_in = tf.keras.layers.concatenate([z_, tf.keras.layers.Flatten()(x_)])

        model = tf.keras.layers.Dense(self.dense_dim)(d_in)
        model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)
        model = tf.keras.layers.Dropout(0.5, name="discriminator_features")(model)
        validity = tf.keras.layers.Dense(1, activation="sigmoid")(model)

        return tf.keras.Model([z, x], validity, name="discriminator")

    def train(self, data):
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        self.best_score = 9999999
        n_batches = int(np.floor(data.shape[0] / self.batch_size))
        for epoch in range(self.epochs):
            z = np.random.normal(size=(self.batch_size, self.latent_dim))
            input_data_ = self.generator.predict(z)

            idx = np.random.randint(0, data.shape[0], self.batch_size)
            input_data = data[idx]
            z_ = self.encoder.predict(input_data)
            d_loss_real = self.discriminator.train_on_batch([z_, input_data], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, input_data_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.bigan_generator.train_on_batch([z, input_data], [valid, fake])
            if d_loss[1] * 100 < self.best_score:
                self.best_score = d_loss[1] * 100
                self.save_model()
            print(
                "%d [D loss: %f , acc: %.2f%%] [G loss: %f] "
                % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0])
            )

        self.load_model()

    def calculateAnomaly(self, input_data, weight_parameter=0.1, lnorm_degree=1):
        z_ = self.encoder.predict(input_data)

        input_data_ = self.generator.predict(z_)

        delta = input_data - input_data_
        delta = delta.reshape(delta.shape[0], -1)
        gen_loss = np.linalg.norm(delta, axis=1, ord=lnorm_degree)

        valid = np.ones((input_data.shape[0], 1))
        disc_loss = self.discriminator.test_on_batch([z_, input_data_], valid)

        final_loss = (1 - weight_parameter) * gen_loss + weight_parameter * disc_loss[0]

        return final_loss

    def save_model(self):
        self.bigan_generator.save_weights("bigan_generator.h5")
        self.encoder.save_weights("encoder.h5")
        self.discriminator.save_weights("discriminator.h5")
        self.generator.save_weights("generator.h5")

    def load_model(self):
        self.bigan_generator.load_weights("bigan_generator.h5")
        self.encoder.load_weights("encoder.h5")
        self.discriminator.load_weights("discriminator.h5")
        self.generator.load_weights("generator.h5")
