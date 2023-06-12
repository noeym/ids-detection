import tensorflow as tf
import numpy as np
from data import *
from model import VAE
from datagenerator import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class calculateTestData:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def getthreshold(self):
        self.threshold = np.quantile(
            self.model.calculateAnomaly(
                self.train_data["train"],
                PredictTrainDataGenerator(self.train_data, self.batch_size),
                PredictTrainDataGenerator(self.train_data, self.batch_size),
            )[0],
            0.99,
        )
        del self.train_data

    def LoadModel(self):
        self.preprocess = datPreProcessing()
        self.train_data = self.preprocess.load_train()
        self.model = VAE(self.train_data["train"].shape[1], batch_size=self.batch_size)
        self.model.load_model(r"Test1.h5")

    def getPerformance(self):
        self.data = self.preprocess.load_test()
        anomaly_loss = self.model.calculateAnomaly(
            self.data["test"],
            PredictTestDataGenerator(self.data, batch_size),
            PredictTestDataGenerator(self.data, batch_size),
        )
        Y_test = np.asarray(self.data["label"]) != "1"
        anomaly_detected_loss = anomaly_loss > self.threshold

        print(self.threshold)
        print(anomaly_detected_loss)

        conf_matrix = confusion_matrix(Y_test, anomaly_detected_loss)

        print(conf_matrix)
        print("f1 score is ", f1_score(Y_test, anomaly_detected_loss))
        print("precision is ", precision_score(Y_test, anomaly_detected_loss))
        print("Recall is ", recall_score(Y_test, anomaly_detected_loss))
        print("Accuracy is ", accuracy_score(Y_test, anomaly_detected_loss))


if __name__ == "__main__":
    batch_size = 1000000
    cal = calculateTestData(batch_size)
    cal.LoadModel()
    cal.getthreshold()
    cal.getPerformance()


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


"""                    100  1000000
[[ 735417 1617534]
 [ 627784 1725167]]
f1 score is  0.6057838505582855
precision is  0.5160997049990412
Recall is  0.7331929139195844
Accuracy is  0.5228719170097464

"""
