from utils.logger import logger
import os
import sys
from pyspark.sql import functions as F
from pathlib import Path
import argparse
from datetime import datetime, timedelta

sys.path.append('../')


def kpi(name, recency, flag: list = None):
    '''
    kpi
        this method will map data into kpi
    :param:
        name: name of kpi
        recency: recency of kpi
        flag: list of kpi
    :return:
        kpi: list of kpi
    '''
    kpi = []
    if name is None:
        name = ""
    else:
        name = name + "_"
    if recency is None:
        recency = ""
    else:
        recency = "_" + recency
    if flag is None:
        flag = ["net_spend_amt", "pkg_weight_unit", "transaction_uid"]
    if "net_spend_amt" in flag:
        kpi.append(F.sum("net_spend_amt").alias(f"{name}sales{recency}"))
    if "pkg_weight_unit" in flag:
        kpi.append(F.sum("pkg_weight_unit").alias(f"{name}units{recency}"))
    if "transaction_uid" in flag:
        kpi.append(F.countDistinct("transaction_uid").alias(
            f"{name}visits{recency}"))
    return kpi


@logger
def tender_kpi(spark, prjct_nm, txn_cust_r360, test=False):
    '''
    tender kpi
        this method will map data into tender kpi
        Recency : 360 days
        Store_Format : allFmt
    :param:
        txn_cust_r360: dataframe from txn_cust_r360 (recency.py)
        test: test mode
    :return:
        cust_tender_kpi: dataframe with tender kpi
    :description:
    (2) KPI by tender type
    ----
    Recency : 360 days
    Store_Format : allFmt

    - % sales with cash.
    - % sales with credit card.
    - % visits with cash.
    - % visits with credit card
    '''
    from utils import files
    mnt_mapper = files.conf_reader("../config/mnt.json")
    abfss_prefix = mnt_mapper["abfss_prefix"]

    cust_tender_total = \
        (txn_cust_r360
         .groupBy("household_id")
         .agg(*kpi("total", "360_days"))
         )

    cust_tender_total.agg(F.count("household_id")).display()

    cust_by_tender_kpi_pv = \
        (txn_cust_r360
         .groupBy("household_id")
         .pivot("tender_type_group")
         .agg(*kpi(None, "360_days", ["net_spend_amt", "transaction_uid"]))
         .fillna(value=0)
         )

    lower_case_column_names = [c.lower()
                               for c in cust_by_tender_kpi_pv.columns]
    cust_by_tender_kpi_pv = cust_by_tender_kpi_pv.toDF(
        *lower_case_column_names)

    cust_tender_kpi = \
        (cust_tender_total
         .join(cust_by_tender_kpi_pv, on=["household_id"], how="left")
         .fillna(0)
         .withColumn("prop_visit_by_cash_360_days", ((F.col("cash_visits_360_days") / F.col("total_visits_360_days"))))
         .withColumn("prop_visit_by_card_360_days", ((F.col("ccard_visits_360_days") / F.col("total_visits_360_days"))))
         .withColumn("prop_sales_by_cash_360_days", ((F.col("cash_sales_360_days") / F.col("total_sales_360_days"))))
         .withColumn("prop_sales_by_card_360_days", ((F.col("ccard_sales_360_days") / F.col("total_sales_360_days"))))
         )

    filename = "cust_allfmt_r360_kpi_by_tender.parquet" if not test else "cust_allfmt_r360_kpi_by_tender_test.parquet"
    files.save(cust_tender_kpi, os.path.join(abfss_prefix, prjct_nm, "features",
               filename), format="parquet", mode="overwrite", overwriteSchema=True)


@logger
def all_kpi(spark, prjct_nm, txn_cust_r360, test=False):
    '''
        all kpi
            Recency : 360 days
            Store_Format : allFmt

            - % Visit by time of day.
            - % Visist by day of week.
        :param:
            txn_cust_r360: dataframe from txn_cust_r360 (recency.py)
        :return:
            cust_tt_kpi: dataframe with all kpi
        :description:
        (5) KPI
        ----
        Recency : 360 days
        Store_Format : allFmt

        - % Visit by time of day.  
        - % Visist by day of week.
    '''
    from utils import files
    mnt_mapper = files.conf_reader("../config/mnt.json")
    abfss_prefix = mnt_mapper["abfss_prefix"]
    cust_tt_kpi = \
        (txn_cust_r360
         .groupBy("household_id")
         .agg(*kpi("Total", "360_days"))
         )

    filename = "cust_allfmt_r360_kpi.parquet" if not test else "cust_allfmt_r360_kpi_test.parquet"
    files.save(cust_tt_kpi, os.path.join(abfss_prefix, prjct_nm, "features",
                                         filename), format="parquet", mode="overwrite", overwriteSchema=True)


@logger
def format_kpi(recency: str,
               str_frmt_grp: str,
               chnnl: str = "all",
               txn_x_recency=None):
    '''
    Compute customer KPI from snap txn x recency
    :param:
        recency: str
            "360_days", "180_days", "90_days", "30_days"
            will map to start KPI range base 
        str_frmt_grp: str
            "hde", "gofresh" or "all" for "hdet" and "gofresh"
        chnnl: str, default "all"
            Filter based on column channel
            "all" or all channel, aslo all format
            "offline" for OFFLINE only
            "online" for ONLINE only (not OFFLINE), will force `all` for str_frmt_gp
        txn_x_recency: SparkDataFrame, default None
            SparkDataFrame of snap txn x recency
    :return:
        SparkDataFrame of customer KPI
    :description:
    (1) KPI by Channel / Format
    ----
    Recency : 360 days, 180 days, 90 days, 30 days
    Channel - Format : online-all, offline-all, offline-hde, offline-gofresh
    '''
    recency_str_date_map = {"360_days": "r360_start_date",
                            "180_days": "r180_start_date",
                            "90_days": "r90_start_date",
                            "30_days": "r30_start_date"}

    # Check transaction x recency date range
    str_date_col = recency_str_date_map[recency]
    print("Recency and recency date column")
    print(f"{recency} : {str_date_col}")

    # Check recency range
    txn_filter_recency = txn_x_recency.where(F.col("date_id").between(
        F.col(str_date_col), F.col("first_contact_date")))

    # Filter channel and format
    if chnnl.lower() == "all":
        txn_to_kpi = txn_filter_recency
    elif chnnl.lower() == f"online":
        txn_to_kpi = txn_filter_recency.where(
            F.col("offline_online_other_channel").isin("ONLINE"))
        str_frmt_grp = "all"  # Overwrite
    elif chnnl.lower() == "offline":
        txn_offline = txn_filter_recency.where(
            F.col("offline_online_other_channel").isin("OFFLINE"))

        # for offline, filter format
        if str_frmt_grp.lower() == "all":
            txn_to_kpi = txn_offline
        elif str_frmt_grp.lower() == "hde":
            txn_to_kpi = txn_offline.where(
                F.col("store_format_online_subchannel_other") == "HDE")
        elif str_frmt_grp.lower() == "gofresh":
            txn_to_kpi = txn_offline.where(
                F.col("store_format_online_subchannel_other") == "GoFresh")
        else:
            pass
    else:
        pass

    print("Check Channel - Format")
    txn_to_kpi.select("offline_online_other_channel",
                      "store_format_online_subchannel_other").drop_duplicates().show()
    name = f"{chnnl}_{str_frmt_grp}"
    agg_kpi = txn_to_kpi.groupBy("household_id").agg(*kpi(name, recency))

    return agg_kpi
