import io
import os
import requests
import numpy as np
from src.proxies import *
from ast import literal_eval
from sklearn.model_selection import train_test_split

class HMDA:
    def __init__(self, file_path, proxy=None, geo=None, outcome=False):

        assert os.path.isfile(file_path)

        self.races = ["white", "black", "api", "native", "multiple", "hispanic"]
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
            features[c] = (features[c] - features[c].mean()) / features[c].std()


        #One hot encode categorial features
        features = pd.get_dummies(features, columns=self.categorial_features)
        features = features.rename(columns={
            "action_taken_1": "Loan Originated",
            "action_taken_2": "Loan Appr (only)",
            "action_taken_3": "Loan Denied",
            "action_taken_4": "App. Withdraw",
            "action_taken_5": "Incomplete",
            "action_taken_6": "Puchased Loan",
            "action_taken_7": "Pre-Appr. Denied",
            "action_taken_8": "Pre-Appr (Only)",
            "denial_reason-1_1": "(Denied) Debt-To-Income",
            "denial_reason-1_2": "(Denied) Employment History",
            "denial_reason-1_3": "(Denied) Credit History Denial",
            "denial_reason-1_4": "(Denied) Collateral",
            "denial_reason-1_5": "(Denied) Insufficient Funds",
            "denial_reason-1_6": "(Denied) Unverifiable",
            "denial_reason-1_7": "(Denied) Credit App Incomplete",
            "denial_reason-1_8": "(Denied) Insurance Denied",
            "denial_reason-1_9": "(Denied) Other",
            "denial_reason-1_10": "Not Denied",
            "denial_reason-1_1111": "Not Specified",
        })

        self.proxy = proxy
        if self.proxy is not None:
            assert isinstance(proxy, BaseProxy)
            self.bisg_probs = proxy.inference(self.data).iloc[:,-6:]
            features = features.join(self.bisg_probs)

        features = features.fillna(features.mean())

        assert geo.lower() in ["tract"], "geography not supported"

        if geo.lower() == "tract":
            features["state"] = self.data["state"]
            features["tract"] = self.data["tract"]
            features["county"] = self.data["county"]




        self.X = features

        self.map_to_index = dict(zip(self.races, np.arange(len(self.races))))
        self.map_to_race = dict(zip(np.arange(len(self.races)), self.races))



        targets = self.data["derived_race"].replace(self.map_to_index)
        self.Y = targets

    def get_data(self, split=None, seed=None, return_bisg=False):

        data = self.X, self.Y

        if return_bisg:
            assert self.proxy is not None, "must pass proxy into constructor"
            data = self.X, self.Y, self.bisg_probs

        if split is not None:
            assert split > 0 and split < 1, "split must be valid fraction"
            data = train_test_split(*data, test_size=split, random_state=seed)

        return data





class Census:
    def __init__(self, api_key, year=2010, cache=True, data_path=None):
        self.key = api_key
        

        assert str(year) in ["2010", "2000"], "Year not supported"
            
        self.year = str(year)

        self.cache = cache

        if data_path is None:
            data_path = ".."

        self.data_path = data_path

    def fetch_surname_race_probs(self, rank=None):
        if rank is None:
            rank = np.iinfo(int).max

        url = "https://api.census.gov/data/{year}/surname?get=" \
              "PCT2PRACE,PCTAIAN,PCTAPI,PCTBLACK,PCTHISPANIC,PCTWHITE,COUNT,NAME" \
              "&RANK=1:{rank}&key={key}".format(
            year = self.year,
            rank = rank,
            key = self.key
        )

        response = requests.get(url).content.decode('utf-8')
        surname_df = pd.DataFrame(literal_eval(response)).replace({
            "(S)" : 0
        })

        return surname_df

    def load_surname_table(self):
        if self.cache:
            file_loc = os.path.join(self.data_path, "data/census")
            if not os.path.exists(file_loc):
                os.makedirs(file_loc)

            fname = os.path.join(file_loc, "surnames.csv")
            if os.path.isfile(fname):
                surname_table = pd.read_csv(fname)
            else:
                surname_table = self.fetch_surname_race_probs()
                surname_table.to_csv(fname, index=False)

            return surname_table
        else:
            return self.fetch_surname_race_probs()


