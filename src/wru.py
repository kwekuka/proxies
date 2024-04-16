import rpy2
import rpy2.robjects as ro
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, data

wru = importr('wru')
voters = data(wru).fetch("voters")["voters"]

hmm = wru.predict_race(voter_file=voters, census_geo="zcta")

with (ro.default_converter + pandas2ri.converter).context():
  pd_from_r_df = ro.conversion.get_conversion().rpy2py(voters)

pd_from_r_df

print()
