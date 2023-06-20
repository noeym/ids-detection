import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler

ids_datatypes = {
    "Dst Port": np.uint32,
    "Protocol": np.int8,
    # "Timestamp": object,
    "Flow Duration": np.int64,
    "Tot Fwd Pkts": np.int16,
    "Tot Bwd Pkts": np.int16,
    "TotLen Fwd Pkts": np.int32,
    "TotLen Bwd Pkts": np.int32,
    "Fwd Pkt Len Max": np.int32,
    "Fwd Pkt Len Min": np.int32,
    "Fwd Pkt Len Mean": np.float64,
    "Fwd Pkt Len Std": np.float64,
    "Bwd Pkt Len Max": np.int16,
    "Bwd Pkt Len Min": np.float64,
    "Bwd Pkt Len Mean": np.float64,
    "Bwd Pkt Len Std": np.float64,
    "Flow Byts/s": np.float64,
    "Flow Pkts/s": np.float64,
    "Flow IAT Mean": np.float64,
    "Flow IAT Std": np.float64,
    "Flow IAT Max": np.int64,
    "Flow IAT Min": np.int32,
    "Fwd IAT Tot": np.int32,
    "Fwd IAT Mean": np.float32,
    "Fwd IAT Std": np.float64,
    "Fwd IAT Max": np.int32,
    "Fwd IAT Min": np.int32,
    "Bwd IAT Tot": np.int32,
    "Bwd IAT Mean": np.float64,
    "Bwd IAT Std": np.float64,
    "Bwd IAT Max": np.int64,
    "Bwd IAT Min": np.int64,
    "Fwd PSH Flags": np.int8,
    # "Bwd PSH Flags": np.int8,
    "Fwd URG Flags": np.int8,
    # "Bwd URG Flags": np.int8,
    "Fwd Header Len": np.int32,
    "Bwd Header Len": np.int32,
    "Fwd Pkts/s": np.float64,
    "Bwd Pkts/s": np.float64,
    "Pkt Len Min": np.int16,
    "Pkt Len Max": np.int32,
    "Pkt Len Mean": np.float64,
    "Pkt Len Std": np.float64,
    "Pkt Len Var": np.float64,
    "FIN Flag Cnt": np.int8,
    "SYN Flag Cnt": np.int8,
    "RST Flag Cnt": np.int8,
    "PSH Flag Cnt": np.int8,
    "ACK Flag Cnt": np.int8,
    "URG Flag Cnt": np.int8,
    "CWE Flag Count": np.int8,
    "ECE Flag Cnt": np.int8,
    "Down/Up Ratio": np.int8,
    "Pkt Size Avg": np.float32,
    "Fwd Seg Size Avg": np.float32,
    "Bwd Seg Size Avg": np.float32,
    # "Fwd Byts/b Avg": np.int8,
    # "Bwd Byts/b Avg": np.int8,
    # "Fwd Blk Rate Avg": np.int8,
    # "Bwd Blk Rate Avg": np.int8,
    # "Fwd Pkts/b Avg": np.int8,
    # "Bwd Pkts/b Avg": np.int8,
    "Subflow Fwd Pkts": np.int16,
    "Subflow Fwd Byts": np.int32,
    "Subflow Bwd Pkts": np.int16,
    "Subflow Bwd Byts": np.int32,
    "Init Fwd Win Byts": np.int32,
    "Init Bwd Win Byts": np.int32,
    "Fwd Act Data Pkts": np.int16,
    "Fwd Seg Size Min": np.int8,
    "Active Mean": np.float64,
    "Active Std": np.float64,
    "Active Max": np.int32,
    "Active Min": np.int32,
    "Idle Mean": np.float64,
    "Idle Std": np.float64,
    "Idle Max": np.int64,
    "Idle Min": np.int64,
    "Label": object,
}
used_cols = ids_datatypes.keys()


class datPreProcessing(pd.DataFrame):
    def dataReadCSV(path, uncol):
        df = pd.read_csv(path, dtype=ids_datatypes, usecols=used_cols, low_memory=False)
        return df

    def sepatrationLabel(data):
        data, label = data.drop(columns=["Label"]), data["Label"]
        return data, label

    def sepatrationOutIn(data):
        return data[data["Label"] == "1"], data[data["Label"] != "1"]

    def minmaxscale(self, data):
        self.scaler = MinMaxScaler().fit(data)
        data = self.scaler.transform(data)
        return data

    def robustscale(self, data):
        self.scaler = RobustScaler().fit(data)
        data = self.scaler.transform(data)
        return data

    def normalizer(self, data):
        self.scaler = Normalizer().fit(data)
        data = self.scaler.transform(data)
        return data

    def maxAbsscale(self, data):
        self.scaler = MaxAbsScaler().fit(data)
        data = self.scaler.transform(data)
        return data

    def standardscale(self, data):
        self.scaler = StandardScaler().fit(data)
        data = self.scaler.transform(data)
        return data

    def testscale(self, data):
        return self.scaler.transform(data)

    def load_train(self, i):
        self.unused_col = i
        print(" Start Load train Data")
        train_data = datPreProcessing.dataReadCSV(r"train.csv", self.unused_col)
        train_data = datPreProcessing.sepatrationLabel(train_data)[0]
        train_data = self.standardscale(train_data)
        return {"train": train_data}

    def load_test(self):
        print(" Start Load test Data")
        test_data = datPreProcessing.dataReadCSV(r"test.csv", self.unused_col)
        test, label = datPreProcessing.sepatrationLabel(test_data)
        test = self.scaler.transform(test)

        return {
            "test": test,
            "test_label": label,
        }
