import io
import os

import pandas as pd
import requests
import numpy as np
from src.proxies import *
from ast import literal_eval
from src.constants import constants
from sklearn.model_selection import train_test_split

class HMDA:
    def __init__(self, file_path, proxy=None, geo=None, include_geo=False, outcome=False, races=None):

        assert os.path.isfile(file_path)

        self.races = races
        if races is None:
            self.races = ["white", "black", "api", "native", "multiple", "hispanic"]

        self.data = pd.read_csv(file_path, dtype={
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

        #Replace nans w/ average value
        features = features.fillna(features.mean())


        assert "census_tract" in self.data.columns
        if "census_tract" in self.data.columns:
            self.data["tract"] = self.data["census_tract"].apply(lambda x: str(x)[5:])
            self.data["state"] = self.data["census_tract"].apply(lambda x: str(x)[:2])
            self.data["county"] = self.data["census_tract"].apply(lambda x: str(x)[2:5])


        # assert geo.lower() in ["tract"], "geography not supported"
        # if geo.lower() == "tract" and include_geo:
        #     features["state"] = self.data["state"].astype(str)
        #     features["tract"] = self.data["tract"].astype(str)
        #     features["county"] = self.data["county"].astype(str)



        self.proxy = proxy
        if self.proxy is not None:
            assert isinstance(self.proxy, BaseProxy)
            self.bisg_probs = proxy.inference(self.data, races=self.races, probs_only=True)
            features = pd.concat([features,self.bisg_probs], axis=1)

        self.X = features


        targets = self.data["derived_race"].apply({
            lambda x: races.index(x) if x in races else len(races)
        })
        self.Y = targets.astype(int)

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
    def __init__(self, races=None, api_key=None, year=2010, cache=True, data_path=None):
        assert api_key is not None, "Must pass location of api key"
        api_key_file = open(api_key)
        key = api_key_file.readline()

        self.key = str(key)
        

        assert str(year) in ["2010", "2000"], "Year not supported"
            
        self.year = str(year)

        self.cache = cache

        if data_path is None:
            data_path = ".."

        self.data_path = data_path

        self.all_races = ['hispanic', 'white', 'black', 'native', 'api', 'multiple']
        self.races = races


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

        surname_df.columns = surname_df.iloc[0]

        surname_df = surname_df.iloc[1:]

        race_mapper = {
            "PCT2PRACE": constants.multiple,
            "PCTAIAN": constants.aian,
            "PCTAPI": constants.api,
            "PCTBLACK": constants.black,
            "PCTHISPANIC": constants.hisp,
            "PCTWHITE": constants.white
        }

        surname_df = surname_df.rename(inplace=False, columns=race_mapper)

        surname_race_probs = surname_df.loc[:,race_mapper.values()].astype(float)
        surname_df.loc[:,race_mapper.values()] = surname_race_probs/surname_race_probs.sum(axis=0)

        return surname_df


    def consolidate_race_probs(self, df, index):
        races = self.races
        all_races = self.all_races

        if races is not None:
            incl_indexes = [list(df.columns).index(r) for r in races]
            all_index = [list(df.columns).index(r) for r in all_races]

            excl_indexes = [ix for ix in all_index if ix not in incl_indexes]

            # Get the probs we want to keep
            probs = df.copy().iloc[:, incl_indexes].astype(int)

            # Drop all probs
            drop = df.drop(df.columns[all_index], axis=1, inplace=False)

            # Add probs back in
            drop = pd.concat([drop, probs], axis=1)

            if excl_indexes:
                # Get the probs we are grouping together
                other_probs = df.copy().iloc[:, excl_indexes].astype(int).sum(axis=1, min_count=1)

                # Add them back on
                drop["other"] = other_probs

        # Get all columns that contain probs that we're keeping
        drop_columns = list(drop.columns)

        # Get the indices in the above
        prob_indices = [drop_columns.index(r) for r in races + ["other"]]

        # Make dictionary index
        index_dict = dict(zip(drop.loc[:, index], np.arange(len(drop))))

        return BaseIndex(drop.iloc[:, prob_indices], index=index_dict)

    def load_surname_table(self, index="NAME"):
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
        else:
            surname_table = self.fetch_surname_race_probs()



        return self.consolidate_race_probs(surname_table, index)


    def load_geo_table(self, index, path=None):
        if path is None:
            path = "../data/census/race.csv"

        assert os.path.isfile(path), "file doesn't exist at the location"
        df = pd.read_csv(path, dtype=str)


        return self.consolidate_race_probs(df, index)



class BaseIndex:
    def __init__(self, data: pd.DataFrame, index):
        self.data = data
        self.index = index


    def __getitem__(self, item):
        return self.data.iloc[self.index[item]]