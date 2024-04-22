import os
import numpy as np
from src.proxies import *
from sklearn.model_selection import train_test_split

class HMDA:
    def __init__(self, file_path, proxy=None, outcome=False):

        assert os.path.isfile(file_path)


        self.data  = pd.read_csv("../data/hmda/hmda.csv", dtype={
            "census_tract":str,
            "tract": str

        })

        self.continuous_features = continuous_features = [
            "loan_amount",
            "income",
        ]

        self.categorial_features = ["denial_reason-1"]
        if outcome:
            self.categorial_features.append("action_taken")

        features = self.data[self.continuous_features
                             + self.categorial_features].copy()


        #Normalize continuous features
        for c in continuous_features:
            features[c] = (features[c] - features[c].min()) / features[c].max()


        #One hot encode categorial features
        features = pd.get_dummies(features, columns=self.categorial_features)

        self.proxy = proxy
        if self.proxy is not None:
            assert isinstance(proxy, BaseProxy)
            self.bisg_probs = proxy.inference(self.data).iloc[:,-6:]
            features = features.join(self.bisg_probs)

            features = features.fillna(features.mean())

            self.X = features

            self.map_to_index = dict(zip(self.bisg_probs.columns, np.arange(len(self.bisg_probs.columns))))
            self.map_to_race = dict(zip(np.arange(len(self.bisg_probs.columns)), self.bisg_probs.columns, ))

            targets = self.data["derived_race"].replace(self.map_to_index)
            self.Y = targets
            print()

    def get_data(self, split=None, seed=None, return_bisg=False):
        assert self.proxy is not None, "must pass proxy into constructor"

        data = self.X, self.Y

        if return_bisg:
            data = self.X, self.Y, self.bisg_probs

        if split is not None:
            assert split > 0 and split < 1, "split must be valid fraction"
            data = train_test_split(*data, test_size=split, random_state=seed)

        return data

gbisg = gBisg()

X_train, X_test, Y_train, Y_test = HMDA("../data/hmda/hmda.csv", gbisg, outcome=True).get_data(split=.5)


