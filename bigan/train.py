from model import BIGAN
from data import *
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from test import *


def train_bigan():
    preprocess = datPreProcessing()
    data = preprocess.load_train()
    print("Completed Load Data")
    print("Start Training")
    batch_size = 100000
    epochs = 600
    model = BIGAN(data["train"].shape[1], batch_size, epochs)
    model.train(data["train"])
    model.load_model()
    anomaly_train_loss = model.calculateAnomaly(data["train"])
    test_data = preprocess.load_test()
    anomaly_test_loss = model.calculateAnomaly(test_data["test"])
    Y_test = np.asarray(test_data["test_label"]) != "Benign"
    threshold = 0.0
    best_score = 0.0
    for i in range(1, 100):
        j = np.quantile(anomaly_train_loss, i / 100)
        score = anomaly_test_loss > j
        current_score = f1_score(Y_test, score)
        if best_score < current_score:
            print(i)
            best_score = current_score
            threshold = j

    anomaly_detected = anomaly_test_loss > threshold
    print(confusion_matrix(Y_test, anomaly_detected))
    print("f1_score is ", f1_score(Y_test, anomaly_detected))
    print("precision_score is ", precision_score(Y_test, anomaly_detected))
    print("Recall_score is ", recall_score(Y_test, anomaly_detected))
    print("Accuracy_score is ", accuracy_score(Y_test, anomaly_detected))
    return model


if __name__ == "__main__":
    disable_eager_execution()
    train_bigan()
