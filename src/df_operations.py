import pandas as pd

def merge_dataframes(df_tuple):
    """
    Merge multiple transport DataFrames into a single global dataset.

    Each DataFrame must contain flow data and should ideally have a .name attribute.
    The function adds a 'Transport_Type' column based on df.name, or assigns a
    generic label if no name is found.

    Args:
        df_tuple (tuple or list): Iterable containing several pandas DataFrames.

    Returns:
        pd.DataFrame: A single concatenated DataFrame with an added 'Transport_Type' column.
    """
    return (
        pd.concat(df_tuple, ignore_index=True)
          .groupby("date", as_index=False)["Flow"]
          .sum()
    )


def add_week_regularity_feature(df):
    """
    Add a column indicating the time period for each observation:
    - regular week
    - school holidays (with type)
    - COVID period
    - Fête des Lumières

    Args:
        df (pd.DataFrame): DataFrame containing transport flow data.

    Returns:
        pd.DataFrame: DataFrame with an additional 'Time_Period' column.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["date_only"] = df["date"].dt.date

    # School holiday periods in France (year 2019/2020)
    periods = {
        "Christmas holidays": (pd.to_datetime("2019-12-21").date(),
                               pd.to_datetime("2020-01-05").date()),
        "Winter holidays":    (pd.to_datetime("2020-02-15").date(),
                               pd.to_datetime("2020-03-01").date()),
        "COVID period":       (pd.to_datetime("2020-03-17").date(),
                               pd.to_datetime("2020-05-11").date()),
        "Fête des Lumières":  (pd.to_datetime("2019-12-05").date(),
                               pd.to_datetime("2019-12-08").date())
    }

    def classify_date(d):
        for period_name, (start, end) in periods.items():
            if start <= d <= end:
                return period_name
        return "Regular week"

    df["Time_Period"] = df["date_only"].apply(classify_date)
    return df
