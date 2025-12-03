"""
filters.py

List of boolean filters for TCL validation datasets.
Assumes that the `date` column is a pandas datetime (dtype datetime64[ns]).
"""

import pandas as pd


COVID_START_DATE = "2020-03-16"


def filter_no_covid(df, date_col: str = "date"):
    """Keep only data strictly before the Covid lockdown period."""
    return df[date_col] < COVID_START_DATE


def filter_christmas_holiday(df, date_col: str = "date"):
    """
    Christmas holidays:
    from 21 December to 5 January (included).
    """
    month = df[date_col].dt.month
    day = df[date_col].dt.day

    return (
        ((month == 12) & (day >= 21)) |
        ((month == 1) & (day <= 5))
    )


def filter_february_holiday(df, date_col: str = "date"):
    """
    February holidays (Zone A approx):
    from 22 February to 8 March (included).
    """
    month = df[date_col].dt.month
    day = df[date_col].dt.day

    return (
        ((month == 2) & (day >= 22)) |
        ((month == 3) & (day <= 8))
    )


def filter_fete_lumieres(df, date_col: str = "date"):
    """
    Fête des Lumières:
    from 5 to 8 December (included).
    """
    month = df[date_col].dt.month
    day = df[date_col].dt.day

    return (
        (month == 12) &
        (day >= 5) &
        (day <= 8)
    )

# ============================ FILTER APPLICATION ============================

def apply_all_filters(df, date_col="date"):
    """
    Apply all filters (Covid, Christmas, February, Fête des Lumières)
    to a single DataFrame and return a dictionary containing:
      - filtered DataFrames
      - masks if needed for composition

    Example:
        results = apply_all_filters(df_bus)
        df_bus_no_covid = results["no_covid"]
        df_bus_christmas = results["christmas"]
    """
    # Get base name for filtered DataFrames
    base_name = getattr(df, "name", "dataset")   # dataset name
    
    # Boolean masks computed on FULL df
    f_no_covid  = filter_no_covid(df, date_col)
    f_christmas = filter_christmas_holiday(df, date_col)
    f_february  = filter_february_holiday(df, date_col)
    f_lumieres  = filter_fete_lumieres(df, date_col)

    # Filtered DataFrames computed SAFELY
    df_no_covid   = df[f_no_covid]
    df_christmas  = df[f_christmas]
    df_february   = df[f_february]
    df_lumieres   = df[f_lumieres]
    df_no_holiday = df[f_no_covid & ~(f_christmas | f_february)] # To avoid reindexing issues

    # Assign .name attribute to each filtered DataFrame
    df_no_covid.name   = f"{base_name}`"
    df_christmas.name  = f"{base_name}`"
    df_february.name   = f"{base_name}`"
    df_lumieres.name   = f"{base_name}`"
    df_no_holiday.name = f"{base_name}`"

   # ----- PRINT : list of filters applied -----
    print(
        f"Applied filters for dataset '{getattr(df, 'name', 'dataset')}': "
        f"[no_covid, christmas, february, lumieres, no_holiday]"
    )
    return {
        "no_covid": df_no_covid,
        "christmas": df_christmas,
        "february": df_february,
        "lumieres": df_lumieres,
        "no_holiday": df_no_holiday,
    }

# Example usage:
# results_bus = apply_all_filters(df_bus)
# df_bus_no_covid = results_bus["no_covid"]
# df_bus_christmas = results_bus["christmas"]
# df_bus_february = results_bus["february"]
# df_bus_lumieres = results_bus["lumieres"]
# df_bus_no_holiday = results_bus["no_holiday"]
# Similarly for tramway and subway datasets