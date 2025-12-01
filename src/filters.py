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
