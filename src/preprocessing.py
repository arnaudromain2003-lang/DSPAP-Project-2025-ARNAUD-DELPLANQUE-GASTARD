# ============================================
# Imports
# ============================================
import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys


# ============================================
# Configure the project root (Notebook version)
# ============================================

# cwd() = notebooks/
project_root = Path.cwd().parent  # → mon_projet/
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))



# ============================================
# Paths configuration
# ============================================
DATA_ROOT = project_root / "data" / "PT" / "pt_data"
AGG = "15min"
MODES = ["bus","tramway","subway"]



# ============================================
# Load all modes (subway / tramway / bus)
# ============================================  

dfs = {}  # mode -> DataFrame

for mode in MODES:
    csv_path = DATA_ROOT / f"{mode}_indiv_{AGG}" / f"{mode}_indiv_{AGG}.csv"
    df = pd.read_csv(csv_path, index_col=0,low_memory=False)
    # Ensure datetime column
    if "VAL_DATE" in df.columns:
        df["VAL_DATE"] = pd.to_datetime(df["VAL_DATE"])
    else:
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        df.rename(columns={"index":"VAL_DATE"}, inplace=True) 
    dfs[mode] = df
    


# =======================================================
# Extract cleaned DataFrames (VAL-DATE and Flow only)
# =======================================================
df_bus = dfs["bus"][["VAL_DATE", "Flow"]].copy()
df_tramway = dfs["tramway"][["VAL_DATE", "Flow"]].copy()
df_subway = dfs["subway"].copy()

# Compute subway flow as sum across all columns
df_subway = df_subway.reset_index(drop=True)  # VAL_DATE becomes a normal column (and not index)
df_subway["Flow"] = df_subway[df_subway.columns[1:]].sum(axis=1)
df_subway = df_subway[["VAL_DATE", "Flow"]]

df_subway = df_subway.rename(columns={"VAL_DATE": "date"})
df_bus = df_bus.rename(columns={"VAL_DATE": "date"})
df_tramway = df_tramway.rename(columns={"VAL_DATE": "date"})

# Assign names to DataFrames for identification
df_bus.name = "bus"
df_tramway.name = "tramway"
df_subway.name = "subway"

# ============================================
# Display shapes of loaded DataFrames
# ============================================
print("Data loaded: bus =", df_bus.shape, 
      "tramway =", df_tramway.shape, 
      "subway =", df_subway.shape)

# print('Bus dataset:',df_bus.head())
# print('Tramway dataset:',df_tramway.head())
# print('Subway: dataset',df_subway.head())

def load_data():
    """
    Load the datasets for bus, tramway, and subway modes.
    
    Returns:
        df_bus (pd.DataFrame): DataFrame containing 'date' and 'Flow' for bus mode.
        df_tramway (pd.DataFrame): DataFrame containing 'date' and 'Flow' for tramway mode.
        df_subway (pd.DataFrame): DataFrame containing 'date' and 'Flow' for subway mode.
    """
    return df_bus, df_tramway, df_subway


# Sample output:
# Data loaded: bus = (140256, 2) ,tramway = (140256 , 2), subway = (140256, 2)
# Now df_bus, df_tramway, df_subway are ready for analysis
#
# They each contain VAL_DATE and Flow columns
#
# Example usage:
# print(df_bus.head())
# print(df_tramway.head())
# print(df_subway.head())
#
# Sample output:
#             VAL_DATE  Flow
# 0 2020-01-01 00:00:00   123
# 1 2020-01-01 00:15:00   150
# 2 2020-01-01 00:30:00   130
# 3 2020-01-01 00:45:00   160   
# 4 2020-01-01 01:00:00   140
#             VAL_DATE  Flow
# 0 2020-01-01 00:00:00    80
# 1 2020-01-01 00:15:00    90
# 2 2020-01-01 00:30:00    85
# 3 2020-01-01 00:45:00    95   
# 4 2020-01-01 01:00:00    88
#             VAL_DATE  Flow
# 0 2020-01-01 00:00:00   300   
# 1 2020-01-01 00:15:00   320
# 2 2020-01-01 00:30:00   310
# 3 2020-01-01 00:45:00   330
# 4 2020-01-01 01:00:00   315   
# These DataFrames can now be used for further analysis

# ============================================
# End of src/preprocessing.py
# ============================================  



