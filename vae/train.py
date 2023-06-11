from model import VAE
from datagenerator import *
from data import *
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np


def train_vae(data):
    print("Start Training")
    batch_size = 1000000
    epochs = 50
    model = VAE(data["train"].shape[1], batch_size, epochs)
    train_gen, valid_gen = TrainDataGenerator(
        data, batch_size
    ), ValidationDataGenerator(data, batch_size)
    model.train(train_gen, valid_gen)
    del train_gen, valid_gen
    print(
        np.quantile(
            model.calculateAnomaly(
                data["train"],
                PredictTrainDataGenerator(data, batch_size),
                PredictTrainDataGenerator(data, batch_size),
            )[0],
            0.99,
        )
    )
    return model


if __name__ == "__main__":
    disable_eager_execution()
    data = datPreProcessing.load_train()
    print("Completed Load Data")
    model = train_vae(data)
    model.save_model("Test3.h5")
    model.showHistory()
