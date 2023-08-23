# # system
import sys
import traceback
import os
from pathlib import Path

# basic data lib
import pandas as pd
import numpy as np

# pyspark lib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import argparse

sys.path.append('../')


def feedback(spark, insurer, abfss_prefix, dbfs_path, prjct_nm, stage, conf_mapper, fdbk_type, lot):
    from utils import etl
    insurer_nm = conf_mapper[insurer]["insurer_nm"]
    landing_abfss_prefix = os.path.join(
        abfss_prefix, prjct_nm, stage, f"Feedback_{insurer_nm}")
    if fdbk_type == "1":
        etl.data_appending(spark, dbfs_path, abfss_prefix,
                           insurer, conf_mapper, True, lot)
        print(landing_abfss_prefix)
    else:
        fdbk_df, filename = etl.get_df(
            dbfs_path, insurer, conf_mapper, fdbk_type=fdbk_type)
        cust_fdbck_hh = etl.ngc_to_household(spark,
                                             spark.createDataFrame(fdbk_df, schema=None))
        return cust_fdbck_hh


if __name__ == "__main__":
    from src.utils import conf
    spark = SparkSession.builder.appName("lmp").getOrCreate()
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--insurer", choices=["AZAY", "CIGNA"], help='''set insurer name 
                                                        AZAY : Allianz  
                                                        CIGNA : Cigna 
                                                ''')
    parser.add_argument(
        "-l", "--lot", help="set lot for appending e.g., 202208")
    parser.add_argument(
        "-f", "--feedback", default="1", choices=["1", "2", "3"], help="feedback type e.g., 202208")

    args = parser.parse_args()
    conf_mapper = conf.conf_reader("../config/etl.json")
    mnt_mapper = conf.conf_reader("../config/mnt.json")
    abfss_prefix, dbfs_path = (
        mnt_mapper["abfss_prefix"], Path(mnt_mapper["dbfs_path"]))
    prjct_nm = "feedback"
    stage = "landing"
    lot = None
    feedback = args.feedback
    if args.lot:
        lot = args.lot
    if args.insurer:
        feedback(spark, args.insurer, abfss_prefix, dbfs_path,
                 prjct_nm, stage, conf_mapper, lot)
    else:
        print("please provide insurer name: AZAY or CIGNA")
