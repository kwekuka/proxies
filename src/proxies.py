import pandas as pd
import rpy2.robjects as ro
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



class Bisg(BaseProxy):

    def __init__(self, geo_level="tract", surname_only=False):
        self.wru = importr('wru')
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

        assert "state" in data.columns, "Must contain a column labeled \"state\" "\
                                        "This is case sensitive"

        # assert "zcta" in data.columns, "Must contain a column labeled \"zcta\" "\
        #                                 "This is case sensitive"


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



fbisg = fBisg()
bisg = Bisg()


wru = importr('wru')
voters = data(wru).fetch("voters")["voters"]
with (ro.default_converter + pandas2ri.converter).context():
  pd_from_r_df = ro.conversion.get_conversion().rpy2py(voters)

bisg.inference(pd_from_r_df)
fbisg.inference(pd_from_r_df)







