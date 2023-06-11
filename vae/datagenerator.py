import tensorflow as tf
import numpy as np


class TrainDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set["train"]
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x, batch_x


class ValidationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set["valid"]
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        return (batch_x, batch_x)


class PredictTrainDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set["train"]
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x


class PredictTestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set["test"]
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x
