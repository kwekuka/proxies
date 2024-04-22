import surgeo
import pandas as pd
import rpy2.robjects as ro
import sklearn
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, data

class BaseProxy:
    """
    Base proxy class which implements various proxy methods
    """
    def __init__(self):
        pass

    def train(self, data: pd.DataFrame):
        pass

    def inference(self, data: pd.DataFrame):
        pass

    def validate_data(self, data) -> bool:
        pass


class gBisg(BaseProxy):

    def __init__(self):
        self.geo = surgeo.GeocodeModel(geo_level="tract")
    def validate_data(self, data) -> bool:
        assert "state" in data.columns, "Must contain a column labeled \"state\" " \
                                          "This is case sensitive"

        assert "tract" in data.columns, "Must contain a column labeled \"tract\" " \
                                        "This is case sensitive"

        assert "county" in data.columns, "Must contain a column labeled \"county\" "\
                                        "This is case sensitive"

        return data[["state","county", "tract"]]

    def inference(self, data: pd.DataFrame):
        data = self.validate_data(data)
        return self.geo.get_probabilities_tract(data)


class Bisg(BaseProxy):

    def __init__(self, geo_level="tract", surname_only=False):
        self.wru = importr('wru')
        self.surname_only = surname_only
        self.call_wru = lambda x: self.wru.predict_race(
            voter_file = x,
            census_geo = geo_level,
            surname_only = surname_only,
        )

    def validate_data(self, data) -> (pd.DataFrame):
        assert type(data) is pd.DataFrame, "Data must be pandas dataframe"

        #TODO: improve functionality to include other census.geo fields

        assert "surname" in data.columns, "Must contain a column labeled \"surname\" "\
                                          "This is case sensitive"
        if not self.surname_only:
            assert "state" in data.columns, "Must contain a column labeled \"state\" " \
                                            "This is case sensitive"

            assert "tract" in data.columns, "Must contain a column labeled \"tract\" " \
                                            "This is case sensitive"

            assert "county" in data.columns, "Must contain a column labeled \"county\" " \
                                             "This is case sensitive"

        return data


    def inference(self, data: pd.DataFrame):


        # Validate data
        data = self.validate_data(data)

        #Convert data to R
        with (ro.default_converter + pandas2ri.converter).context():
            rdf_data = ro.conversion.get_conversion().py2rpy(data)

        wru_prediction = self.call_wru(rdf_data)

        #Convert data back to python
        with (ro.default_converter + pandas2ri.converter).context():
            df_results = ro.conversion.get_conversion().rpy2py(wru_prediction)

        return df_results


class fBisg(Bisg):
    def __init__(self, geo_level="county", surname_only=False):
        self.wru = importr('wru')
        self.call_wru = lambda x: self.wru.predict_race(
            voter_file=x,
            census_geo=geo_level,
            surname_only=surname_only,
            model="fBISG"
        )



class ftBisg(BaseProxy):
    def __init__(self, base_bisg, model):

        assert isinstance(base_bisg, BaseProxy)
        self.base = base_bisg

        supported_models = [
            sklearn.linear_model.LogisticRegression,
            sklearn.ensemble.GradientBoostingClassifier,
            sklearn.ensemble.RandomForestClassifier,
        ]

        assert type(model) in supported_models, "Model not supported"

        self.model = model

    def train(self, X, Y):
        self.model.fit(X,Y)

    def inference(self, X: pd.DataFrame):
        return self.model.predict_proba(X)


class bpBisg(BaseProxy):
    def __init__(self, geo_index):
        self.geo_index = geo_index
        pass










