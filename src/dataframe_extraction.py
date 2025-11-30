from pathlib import Path
import pandas as pd
import geopandas as gpd
FOLDER_PATH = './'

agg = "15min"  
modes = ["subway", "tramway", "bus"]
base = Path(FOLDER_PATH) / "validation_individuelle"

dfs = {}  # mode -> DataFrame

for mode in ['subway','tramway','bus']:
    csv_path = f"{base}/{mode}_indiv_{agg}/{mode}_indiv_{agg}.csv"
    df = pd.read_csv(csv_path,index_col = 0)
    if 'VAL_DATE' in df.columns:
        df['VAL_DATE'] = pd.to_datetime(df['VAL_DATE']) 
    else: 
        df.index = pd.to_datetime(df.index)
    dfs[mode] = df
    
ref_subway = pd.read_csv(f"{base}/ref_subway.csv",index_col = 0).rename(columns = {'MEAN_X' : 'lon','MEAN_Y':'lat'})
ref_tram_bus = pd.read_csv(f"{base}/ref_tram_bus.csv",index_col = 0).rename(columns = {'IDT_PNT' : 'VAL_ARRET_CODE','COO_X_WGS84':'lon','COO_Y_WGS84':'lat'})

ref_tram_bus = ref_tram_bus[['lon','lat','NOM_PNT','VAL_ARRET_CODE']]
ref_subway = ref_subway[['lon','lat','COD_TRG','LIB_STA_SIFO']]

df_bus = dfs["bus"].merge(ref_tram_bus, how = 'inner', on = 'VAL_ARRET_CODE')  
df_tramway = dfs["tramway"].merge(ref_tram_bus, how = 'inner', on = 'VAL_ARRET_CODE')

df_bus["date"]=pd.to_datetime(df_bus["VAL_DATE"])
df_tramway["date"]=pd.to_datetime(df_tramway["VAL_DATE"])

df_bus["date_only"] = df_bus["VAL_DATE"].dt.date
df_tramway["date_only"] = df_tramway["VAL_DATE"].dt.date

