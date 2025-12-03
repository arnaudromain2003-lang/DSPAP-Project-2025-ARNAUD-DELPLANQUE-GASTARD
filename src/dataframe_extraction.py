from pathlib import Path
import pandas as pd
import geopandas as gpd
import os

FOLDER_PATH = '..'
base = Path(FOLDER_PATH) / "data" / "PT" / "pt_data" # Base path for the data files

agg = "15min"  
modes = ["subway", "tramway", "bus"]

dfs = {}  # mode -> DataFrame

for mode in ['subway','tramway','bus']:
    csv_path = f"{base}/{mode}_indiv_{agg}/{mode}_indiv_{agg}.csv"
    df = pd.read_csv(csv_path,index_col = 0)
    if 'VAL_DATE' in df.columns:
        df['VAL_DATE'] = pd.to_datetime(df['VAL_DATE']) 
    else: 
        df.index = pd.to_datetime(df.index)
    dfs[mode] = df
    
df_bus = dfs["bus"]
df_tramway = dfs["tramway"]
df_subway = dfs["subway"]

# Load reference data for bus and tramway stops
df_bus["date"]=pd.to_datetime(df_bus["VAL_DATE"])
df_tramway["date"]=pd.to_datetime(df_tramway["VAL_DATE"])
df_subway["date"]=pd.to_datetime(df_subway["VAL_DATE"])

df_bus["date_only"] = df_bus["VAL_DATE"].dt.date
df_tramway["date_only"] = df_tramway["VAL_DATE"].dt.date
df_subway["date_only"] = df_subway["VAL_DATE"].dt.date

df_subway['Flow']=dfs["subway"].sum(axis=1)

df_bus=pd.DataFrame(df_bus, columns=['VAL_DATE', 'Flow'])
df_tramway=pd.DataFrame(df_tramway, columns=['VAL_DATE', 'Flow'])
df_subway=pd.DataFrame(df_subway, columns=['VAL_DATE', 'Flow'])



