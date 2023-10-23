import pandas as pd
import requests
import yaml

from pybea.client import BureauEconomicAnalysisClient


### Data

# Credentials

settings = yaml.safe_load(open("credentials.yaml"))

bea_client = BureauEconomicAnalysisClient(settings["api_key"])

# Components of request

base = "https://apps.bea.gov/api/data/?UserID={}".format(settings["api_key"])
dataset = "&DataSetName=NIUnderlyingDetail"


def bea_data(table_number, frequency):
    method = "&method=GetData"
    ind = "&TableName=" + table_number
    freq = "&Frequency=" + frequency
    year = "&Year=ALL"
    format = "&ResultFormat=json"

    url = "{}{}{}{}{}{}{}".format(base, method, dataset, year, ind, freq, format)
    r = requests.get(url).json()

    df_raw = pd.DataFrame(r["BEAAPI"]["Results"]["Data"])
    df = df_raw.loc[:, ["LineDescription", "TimePeriod", "DataValue"]]
    df.columns = ["name", "date", "data"]
    df["date"] = pd.to_datetime(df["date"], format="%YM%m").dt.to_period("M")
    df = df.drop_duplicates().set_index(["date", "name"])
    df["data"] = pd.to_numeric(df["data"].str.replace(",", ""))
    df = df.unstack().droplevel(None, axis=1)

    return df


# Download

df_index = bea_data("U20404", frequency="M")
df_weights = bea_data("U20405", frequency="M")

df_index.to_csv("df_index.csv")
df_weights.to_csv("df_weights.csv")

# PCE Groups

pce_groups = [
    "Personal consumption expenditures",
    "PCE excluding food and energy",
    "PCE excluding energy",
    "Motor vehicles and parts",
    "Furnishings and durable household equipment",
    "Recreational goods and vehicles",
    "Other durable goods",
    "Food and beverages purchased for off-premises consumption",
    "Clothing and footwear",
    "Gasoline and other energy goods",
    "Other nondurable goods",
    "Housing",
    "Water supply and sanitation (25)",
    "Electricity and gas",
    "Health care",
    "Transportation services",
    "Recreation services",
    "Food services and accommodations",
    "Financial services and insurance",
    "Other services",
    "Final consumption expenditures of nonprofit institutions serving households (NPISHs) (132)",
]

df_index.loc[:, pce_groups]
