import tensorflow as tf
import numpy as np
from data import *
from model import VAE
from datagenerator import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def getthreshold(model, batch_size):
    data = datPreProcessing.load_train()
    return np.quantile(
        model.calculateAnomaly(
            data["train"],
            PredictTrainDataGenerator(data, batch_size),
            PredictTrainDataGenerator(data, batch_size),
        )[0],
        0.99,
    )


def LoadModel(batch_size):
    data = datPreProcessing.load_test()
    model = VAE(data["test"].shape[1], batch_size=batch_size)
    model.load_model(r"Test1.h5")
    return data, model


def getPerformance(data, batch_size, threshold):
    anomaly_loss = model.calculateAnomaly(
        data["test"],
        PredictTestDataGenerator(data, batch_size),
        PredictTestDataGenerator(data, batch_size),
    )
    Y_test = np.asarray(data["label"]) != "1"
    anomaly_detected_loss = anomaly_loss > threshold

    print(threshold)
    print(anomaly_detected_loss)

    conf_matrix = confusion_matrix(Y_test, anomaly_detected_loss)

    print(conf_matrix)
    print("f1 score is ", f1_score(Y_test, anomaly_detected_loss))
    print("precision is ", precision_score(Y_test, anomaly_detected_loss))
    print("Recall is ", recall_score(Y_test, anomaly_detected_loss))
    print("Accuracy is ", accuracy_score(Y_test, anomaly_detected_loss))


if __name__ == "__main__":
    batch_size = 10000
    data, model = LoadModel(batch_size)
    threshold = getthreshold(model, batch_size)

    getPerformance(data, batch_size, threshold)
