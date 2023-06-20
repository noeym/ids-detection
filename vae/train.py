from model import VAE
from datagenerator import *
from data import *
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)


def train_vae(data):
    batch_size = 8192
    epochs = 100
    model = VAE(data["train_data"].shape[1], batch_size, epochs)
    train_gen, valid_gen = TrainDataGenerator(
        data, batch_size
    ), ValidationDataGenerator(data, batch_size)
    model.train(data, train_gen, valid_gen)
    del train_gen, valid_gen
    model.load_model()
    anomaly_train_loss = model.calculateAnomaly(
        data["train_data"],
        PredictTrainDataGenerator(data, batch_size),
        PredictEncoderTrainDataGenerator(data, batch_size),
    )
    data = preprocess.load_test()
    Y_test = np.asarray(data["test_label"]) != "Benign"
    threshold = 0.0
    best_score = 0.0
    anomaly_test_loss = model.calculateAnomaly(
        data["test"],
        PredicttestDataGenerator(data, batch_size),
        PredictEncodertestDataGenerator(data, batch_size),
    )
    for i in range(1, 100):
        j = np.quantile(anomaly_train_loss, i / 100)
        score = anomaly_test_loss > j
        current_score = f1_score(Y_test, score)
        if best_score < current_score:
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
    preprocess = datPreProcessing()
    data = preprocess.load_train()
    print("Completed Load Data")
    model = train_vae(data)
