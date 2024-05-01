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


class mlBisg(BaseProxy):
    def __init__(self, model: str):


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
        self.model.fit(X,Y)

    def inference(self, X: pd.DataFrame):
        return self.model.predict_proba(X)


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






