import pandas as pd
import surgeo
import numpy as np

# Instatiate your model (or all five)
fsg = surgeo.BIFSGModel()
sg = surgeo.SurgeoModel()
f = surgeo.FirstNameModel()
g = surgeo.GeocodeModel()
s = surgeo.SurnameModel()

nc_all = pd.read_csv("data/ncvoter1.txt", skiprows=1, sep='\t')
keep_columns = [4, 15, 26, 27] #These are hard coded sorry
nc_all = nc_all.iloc[:, keep_columns]
nc_all.columns = ["surname", "zip", "race", "ethnicity"]
nc = nc_all.dropna()
nc["zip"] = nc.loc[:,"zip"].astype(int)
surnames = pd.Series(nc["surname"])
zips = pd.Series(nc["zip"])
sg.get_probabilities(surnames.iloc[:2], zips.iloc[:2])