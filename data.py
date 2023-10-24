import pandas as pd
import requests
import yaml

from pybea.client import BureauEconomicAnalysisClient


### Data

# Credentials

settings = yaml.safe_load(open("credentials.yaml"))

bea_client = BureauEconomicAnalysisClient(settings["api_key"])

# Components of request

base_url = "https://apps.bea.gov/api/data/?UserID={}".format(settings["api_key"])
method = "GetData"
dataset = "NIUnderlyingDetail"
year = "ALL"
freq = "M"
format = "json"


def bea_data(base_url, method, dataset, year, table_number, freq, format):
    method_url = "&method=" + method
    dataset_url = "&DataSetName=" + dataset
    year_url = "&Year=" + year
    tabtable_number_url = "&TableName=" + table_number
    freq_url = "&Frequency=" + freq
    format_url = "&ResultFormat=" + format

    url = "{}{}{}{}{}{}{}".format(
        base_url,
        method_url,
        dataset_url,
        year_url,
        tabtable_number_url,
        freq_url,
        format_url,
    )
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

df_index = bea_data(base_url, method, dataset, year, "U20404", freq, format)
df_weights = bea_data(base_url, method, dataset, year, "U20405", freq, format)

df_index.to_csv("df_index.csv")
df_weights.to_csv("df_weights.csv")
