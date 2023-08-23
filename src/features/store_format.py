from utils.logger import logger
from pyspark.sql import functions as F
from datetime import datetime, timedelta
import sys
import os
from functools import reduce


def outer_join_hh(left, right):
    return left.join(right, "household_id", "outer").checkpoint()


@logger
def main(spark, prjct_nm, test):
    '''
    quarter recency
        this method will map data into date quarter recency
        30, 90, 180, 360
    :param:
        spark: spark session
        prjct_nm: project name
        test: test mode
    '''

    from .recency import format_kpi
    from . import get_txn_cust
    from src.utils import conf
    mnt_mapper = conf.conf_reader("../config/mnt.json")
    abfss_prefix = mnt_mapper["abfss_prefix"]
    txn_cust = get_txn_cust(spark, prjct_nm, test)
    recencies = ["360_days", "180_days", "90_days", "30_days"]
    online_all = [format_kpi(recency=recency, str_frmt_grp="all",
                             chnnl="online", txn_x_recency=txn_cust) for recency in recencies]
    offline_hde = [format_kpi(recency=recency, str_frmt_grp="hde",
                              chnnl="offline", txn_x_recency=txn_cust) for recency in recencies]
    offline_gofresh = [format_kpi(
        recency=recency, str_frmt_grp="gofresh", chnnl="offline", txn_x_recency=txn_cust) for recency in recencies]

    online = reduce(outer_join_hh, online_all)
    hde = reduce(outer_join_hh, offline_hde)
    gofresh = reduce(outer_join_hh, offline_gofresh)

    all_recency = reduce(outer_join_hh, [online, hde, gofresh])
    all_recency = all_recency.fillna(0)
    filename = "all_recency.parquet" if not test else "all_recency_test.parquet"
    conf.save(all_recency, os.path.join(abfss_prefix, prjct_nm, "features", filename
                                         ), format="parquet", mode="overwrite", overwriteSchema=True)
