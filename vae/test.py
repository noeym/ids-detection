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
    model.load_model(r"Test3.h5")
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
    batch_size = 1000000
    data, model = LoadModel(batch_size)
    threshold = getthreshold(model, batch_size)

    getPerformance(data, batch_size, threshold)


"""
[[    845 2352106]    100, 10000
 [    246 2352705]] 
f1 score is  0.6667000105699229
precision is  0.5000636582425947
Recall is  0.9998954504364944
Accuracy is  0.5001272869685769

"""

"""
[[    280 2352671]
 [     29 2352922]]          100,  100000
f1 score is  0.6666876341636463
precision is  0.5000266703898956
Recall is  0.9999876750514567
Accuracy is  0.5000533372773168
"""

"""                    50   100000
f1 score is  0.5321321372786165
precision is  0.5117169060822908
Recall is  0.554244011031254
Accuracy is  0.5126906595164965
"""


"""                    50   1000000


"""
