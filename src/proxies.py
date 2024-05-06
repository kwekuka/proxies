import numpy as np
import surgeo
import sklearn
import pymc as pm
import pandas as pd
import arviz as az
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


class BaseProxy:
    """
    Base proxy class which implements various proxy methods
    """
    def __init__(self):
        pass

    def train(self, data: pd.DataFrame):
        pass

    def inference(self, data: pd.DataFrame, races:[str] = None, drop_geo:bool = False, probs_only:bool =False):

        pass

    def validate_data(self, data) -> pd.DataFrame:
        pass

    def consolidate_race_probs(self, probs: pd.DataFrame, races: [str]) -> pd.DataFrame:
        assert set(races).issubset(set(probs.columns)), "all races must be contained in df"

        incl_indexes = [list(probs.columns).index(r) for r in races]
        all_index = np.arange(probs.shape[1])

        excl_indexes = [ix for ix in all_index if ix not in incl_indexes]

        probs_new = probs.iloc[:,incl_indexes]
        if excl_indexes: #if list is not empty
            probs_new["other"] = probs.iloc[:,excl_indexes].sum(axis=1, min_count=1)

        return probs_new


class gBisg(BaseProxy):
    #TODO replace with ACS

    def __init__(self, races: [str] = None):
        self.races = races
        self.geo = surgeo.GeocodeModel(geo_level="tract")
        self.geo_headers = ["tract", "state", "county"]

    def validate_data(self, data) -> bool:
        assert "state" in data.columns, "Must contain a column labeled \"state\" " \
                                          "This is case sensitive"

        assert "tract" in data.columns, "Must contain a column labeled \"tract\" " \
                                        "This is case sensitive"

        assert "county" in data.columns, "Must contain a column labeled \"county\" "\
                                        "This is case sensitive"

        return data[["state","county", "tract"]]





    def inference(self, data: pd.DataFrame, races:[str] = None, drop_geo=False, probs_only=False):
        geo_columns = self.validate_data(data)
        probs = self.geo.get_probabilities_tract(geo_columns).iloc[:,-6:]
        probs = self.consolidate_race_probs(probs, races)
        probs = probs.fillna(probs.mean())



        if probs_only:
            return probs
        else:
            if drop_geo:
                data = data.drop(self.geo_headers, axis=1)

            data = pd.concat([data,probs], axis=1)
            return data


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


class mlBisg(BaseProxy):
    def __init__(self, model: str, proxy: BaseProxy = None):
        self.proxy = proxy
        if self.proxy is not None:
            assert isinstance(proxy, BaseProxy), "must be implemented proxy type"
            self.proxy = proxy

        if model.lower() == "mr":
            self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                            penalty="l2", tol=1e-7, max_iter=int(1e6))

        elif model.lower() == "gb":
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                                    max_depth=2, random_state=0)

        elif model.lower() == "rf":
            self.model = RandomForestClassifier(max_depth=2, random_state=0)

        else:
            raise Exception("Model not implemented")


    def train(self, X, Y):
        if self.proxy is not None:
            proxy_input = self.proxy.validate_data(X)

            X = self.proxy.inference(proxy_input, drop_geo=True)
        X = X.to_numpy().astype(float)

        self.model.fit(X,Y)

    def inference(self, X: pd.DataFrame):
        if self.proxy is not None:
            X = self.proxy.validate_data(X)
            X = self.proxy.inference(X)

        return self.model.predict_proba(X)

    def validate_data(self, data) -> bool:
        pass



class bpBisg(BaseProxy):
    def __init__(self, geo_index):
        self.geo_index = geo_index
        pass



class bmrBisg(BaseProxy):
    def __init__(self):
        pass

    def train(self, X, Y):

        Y_target = pd.get_dummies(Y == 1).to_numpy()

        n, m = X.shape[1], Y_target.shape[1]


        with pm.Model() as bmr:
            X_data = pm.Data('X_data', X)
            y_obs_data = pm.Data('Y_data', Y)

            b = pm.Normal('intercept', mu=1, sigma=1, shape=m)
            A = pm.Normal('weights', mu=1, sigma=1, shape=(n, m))


            mu = pm.math.dot(X_data, A) + b
            probs = pm.math.softmax(mu)

            likely = pm.Categorical('obs', p=probs, observed=y_obs_data)

            trace = pm.sample(draws=1)
            idata = az.from_pymc3(trace)
            pm.traceplot(idata)





