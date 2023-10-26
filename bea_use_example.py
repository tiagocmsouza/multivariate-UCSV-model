import pandas as pd
import requests
import yaml

from pybea.client import BureauEconomicAnalysisClient


### Credentials

settings = yaml.safe_load(open("./src/credentials.yaml"))
bea_client = BureauEconomicAnalysisClient(settings["api_key"])

### Datasets Available
datasets = bea_client.get_dataset_list()["BEAAPI"]["Results"]["Dataset"]

datasets = pd.DataFrame(datasets).set_index("DatasetName")


### Tables within a Dataset

# Componentes of request
base = "https://apps.bea.gov/api/data/?UserID={}".format(settings["api_key"])
get_param = "&method=GetParameterValues"
param = "TableID"

# Dataset "NIUnderlyingDetail": Standard NI underlying detail tables
dataset = "&DataSetName=NIUnderlyingDetail"

# Construct URL from parameters above
url = "{}{}{}&ParameterName={}&ResultFormat=json".format(
    base, get_param, dataset, param
)

# Request parameter information from BEA API
r = requests.get(url).json()

# Show the results as a table:
tables = pd.DataFrame(r["BEAAPI"]["Results"]["ParamValue"]).set_index("TableNumber")


### Get Table data

method = "&method=GetData"
table_name = "&TableName=" + tables.index[0]
freq = "&Frequency=Q"  # This table, specifically, is quarterly data
year = "&Year=ALL"
format = "&ResultFormat=json"

url = "{}{}{}{}{}{}{}".format(base, method, dataset, year, table_name, freq, format)

r = requests.get(url).json()

table_data = pd.DataFrame(r["BEAAPI"]["Results"]["Data"])
