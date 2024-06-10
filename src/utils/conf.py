import json
from .logger import logger

import calendar
from datetime import datetime, date, timedelta
from typing import List

@logger
def conf_reader(path):
    with open(path, "r") as f:
        mapper = json.loads(f.read())
    return mapper

@logger
def conf_writer(mapper, path):
    with open(path, "w") as f:
        json.dump(mapper, f, indent=4, sort_keys=False)
    return

@logger
def is_end_of_month(date: date) -> bool:
    """Checks if the given date is the end of the month. by adding day +1, if date is end of month then next day will be start of next month

    Args:
        date: A datetime.date object.

    Returns:
        True if the date is the end of the month, False otherwise.
    """
    next_day = date + timedelta(days=1)
    return next_day.day == 1

@logger
def get_mapper_pivot_kpi_col(pivot_columns: List, prod_lv: str) -> dict:
    """
    Generates a mapping of original column names to formatted column names based on product level and KPIs.

    Parameters:
    - pivot_columns (list): A list of unique values from the 'grouped_section_code' column used for pivoting.
    - prod_lv (str): The product level, used to customize the prefix in the formatted column names.

    Returns:
    dict: A dictionary where keys are original column names and values are formatted column names.
    """
    key_mapper_1 = ["_Spend", "_Visits", "_Units", "_SPV", "_UPV", "_SPU"]
    colname_mapper_1 = {
        f"{c}{key}": f"CAT_{prod_lv.upper()}_%{c}%{key.upper()}"
        for c in pivot_columns
        for key in key_mapper_1
    }

    key_mapper_2 = ["_PCT_Spend", "_PCT_Visits", "_PCT_Units"]
    pct_mapper_2 = ["_SPEND", "_VISITS", "_UNITS"]
    colname_mapper_2 = {
        f"{c}{key}": f"PCT_CAT_{prod_lv.upper()}_%{c}%{pct.upper()}"
        for c in pivot_columns
        for key, pct in zip(key_mapper_2, pct_mapper_2)
    }

    return {**colname_mapper_1, **colname_mapper_2}


@logger
def get_mapper_pivot_kpi_col_with_recency(pivot_columns: List, prod_lv: str, recency_lv: str) -> dict:
    """
    Generates a mapping of original column names to formatted column names based on product level and KPIs.

    Parameters:
    - pivot_columns (list): A list of unique values from the 'grouped_section_code' column used for pivoting.
    - prod_lv (str): The product level, used to customize the prefix in the formatted column names.
    - recency_num (str): recency L3, L6, L9

    Returns:
    dict: A dictionary where keys are original column names and values are formatted column names.
    """
    VALID_RECENCY_LV = ["", "L3", "L6", "L9", "l3", "l6", "l9"]
    if recency_lv not in VALID_RECENCY_LV:
        raise ValueError(f"Parameter `recency_lv` must be within {VALID_RECENCY_LV}")

    key_mapper_1 = ["_Spend", "_Visits", "_Units", "_SPV", "_UPV", "_SPU"]
    if recency_lv == "":
        colname_mapper_1 = {
        f"{c}{key}": f"CAT_{prod_lv.upper()}_%{c}%{key.upper()}"
        for c in pivot_columns
        for key in key_mapper_1
        }
    else:
        colname_mapper_1 = {
            f"{c}{key}": f"CAT_{prod_lv.upper()}_%{c}%_{recency_lv.upper()}{key.upper()}"
            for c in pivot_columns
            for key in key_mapper_1
        }

    key_mapper_2 = ["_PCT_Spend", "_PCT_Visits", "_PCT_Units"]
    pct_mapper_2 = ["_SPEND", "_VISITS", "_UNITS"]
    if recency_lv == "":
        colname_mapper_2 = {
        f"{c}{key}": f"PCT_CAT_{prod_lv.upper()}_%{c}%{pct.upper()}"
        for c in pivot_columns
        for key, pct in zip(key_mapper_2, pct_mapper_2)
    }
    else:
        colname_mapper_2 = {
            f"{c}{key}": f"PCT_CAT_{prod_lv.upper()}_%{c}%_{recency_lv.upper()}{pct.upper()}"
            for c in pivot_columns
            for key, pct in zip(key_mapper_2, pct_mapper_2)
        }

    return {**colname_mapper_1, **colname_mapper_2}