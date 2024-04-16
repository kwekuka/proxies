"""
Defining constants
"""
import numpy as np


class constants:
    white = "white"
    black = "black"
    hisp = "hispanic"

    #Asian, Native Hawaiian, Pacific Islander
    api = "api"

    #American-Indian // Alaskan Native
    aian = "aian"

    #mixed
    multiple = "multiple"

    #Native Hawaiian, pacific islander
    nwpi = "napi"

    zipcode = "zipcode"
    count = "count"
    name = "name"

    unknown = "unknown"

    race = "race"
    races = [white, black, hisp, api, aian, multiple]

    num_races = 6
    race_index = dict(zip(races, np.arange(num_races)))





