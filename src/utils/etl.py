# system
import sys
import traceback
import os
from pathlib import Path

# basic data lib
import pandas as pd
import numpy as np

# pyspark lib
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .logger import logger


@logger
def file_copy(dt, source="AAC"):
    """
    file_copy:
      copy a single file from dbfs to abfss
    :param:
      dt:str, date in YYYYMM format
      source:str, source of file default AAC
    """
    try:
        file_lst = [
            x.name.split("_")[-1][:-1]
            for x in dbutils.fs.ls(
                f"dbfs:/mnt/tdm_seg/niti/lmp/insurance_lead/feedback/landing/Feedback_{source}/"
            )
        ]
        if dt not in file_lst:
            # case file not appear then move file to landing
            dbutils.fs.cp(
                f"dbfs:/FileStore/niti/LMP/fdbk/FDBK_CC_{source}_{dt}/",
                f"dbfs:/mnt/tdm_seg/niti/lmp/insurance_lead/feedback/landing/Feedback_{source}/FDBK_CC_{source}_{dt}/",
                True,
            )
            # file transfer success
            print(f"done: {dt} is transfer to {source}")
        else:
            # file found then pass
            print(f"skipped: {dt} is occur in {source}")
    except BaseException as ex:
        # get current system exception
        ex_type, ex_value, ex_traceback = sys.exc_info()

        # extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)

        # eormat stacktrace
        stack_trace = list()

        # get trace back message
        for trace in trace_back:
            stack_trace.append(
                f"File : {trace[0]}, Line : {trace[1]}, Func.Name : {trace[2]}, Message : {trace[3]}"
            )

        print(f"Exception type : {ex_type.__name__}")
        print(f"Exception message : {ex_value}")
        print(f"Stack trace : {stack_trace}")


@logger
def multi_copy(dts, sources):
    """
    multi_copy:
      copy file from dbfs to abfss in multiple dt and source
    :param:
      dts: list of date
      sources: list of source
    :return:

    """
    for source in sources:
        for dt in dts:
            file_copy(dt, source)


@logger
def get_df(dbfs_path, source, conf_mapper, lot_date=None, fdbk_type=1):
    """
    get_df:
      transform txt in dbfs into pandas data frame
    :param:
      dbfs_path: str, path of dbfs
      source: str, full name of source ex. ACC --> AZAY
      conf_mapper: json, config mapper for multiple sources
      lot_date: lot id month id (optional) ex. 202208
    :return:
      df: dataframe
    """
    dbfs_path = Path(dbfs_path)
    if lot_date:
        fdbck1_files = dbfs_path.glob(
            f"*/*{source}*FDBK{fdbk_type}*{lot_date}.txt")
    else:
        fdbck1_files = dbfs_path.glob(f"*/*{source}*FDBK{fdbk_type}*.txt")
    li = []
    for filename in fdbck1_files:
        print(f"{filename}")
        df = (
            pd.read_csv(
                filename,
                index_col=None,
                header=0,
                sep="|",
                encoding=conf_mapper[source]["encoding"],
                dtype=conf_mapper[source]["dtype"],
            )
            .assign(src_file=filename.name)
            .assign(insurer_nm=conf_mapper[source]["insurer_nm"])
        )
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    return df, filename.name


@logger
def data_reading(abfss_prefix, source, conf_mapper):

    abfss_prefix = abfss_prefix
    prjct_nm = "feedback"
    stage = "staging"

    staging_path = os.path.join(
        abfss_prefix,
        prjct_nm,
        stage,
        f"Feedback_{conf_mapper[source]['insurer_nm']}.delta",
    )
    staging = spark.read.load(staging_path)
    print(f"Prior append number of record : {staging.count():,d}")
    return staging


@logger
def data_appending(spark,
                   dbfs_path, abfss_prefix, source, conf_mapper, append_flag=False, lot_date=None
                   ):
    """
    data_appending:
      joind data together and append to delta file
    :param:
      dbfs_path: str, path of dbfs
      abfss_prefix: str, path of abfss
      source: str, full name of source ex. ACC --> AZAY
      conf_mapper: json, config mapper for multiple sources
    :return:
    """
    df, filename = get_df(dbfs_path, source, conf_mapper, lot_date)

    abfss_prefix = abfss_prefix
    prjct_nm = "feedback"
    stage = "staging"

    staging_path = os.path.join(
        abfss_prefix,
        prjct_nm,
        stage,
        f"Feedback_{conf_mapper[source]['insurer_nm']}.delta",
    )
    staging = spark.read.load(staging_path)
    print(f"Prior append number of record : {staging.count():,d}")
    src_file = [row.src_file for row in staging.select(
        "src_file").distinct().collect()]
    df_clean = df[df.dib_cust_code.notna()]
    df_clean = df_clean[~df_clean.src_file.isin(src_file)]
    print(set(df_clean.src_file.tolist()))

    to_update = spark.createDataFrame(df_clean, schema=None).withColumn(
        "DHB_Model_Date", F.col("DHB_Model_Date").cast(T.StringType())
    )
    print(f"New record : {to_update.count():,d}")
    if append_flag:
        if staging.filter(staging.src_file.contains(filename)).count() == 0:
            (to_update.write.format("delta").mode("append").save(staging_path))
            print("update complete")
        else:
            print(f"update incomplete: {filename} is exists")


@logger
def ngc_to_household(spark, fdbck):
    """
    ngc_to_household:
      convert n
    """
    cust_dim = spark.table("tdm.v_customer_dim").select(
        F.col("household_id").alias("household_id_ori"),
        F.col("golden_record_external_id_hash").alias(
            "golden_record_external_id_hash_ori"),
        F.lit(1).alias("cust"),
    )
    ngc_mapping = (
        spark.table("tdm.edm_customer_profile")
        .select("ngc_customer_id", "golden_record_external_id_hash")
        .drop_duplicates()
    )
    cust = (
        spark.table("tdm.v_customer_dim")
        .select("golden_record_external_id_hash", "household_id")
        .drop_duplicates()
    )
    cust_fdbck_hh = (
        fdbck.withColumnRenamed("dib_cust_code", "ngc_customer_id")
        .join(ngc_mapping, "ngc_customer_id", "left")
        .join(cust, "golden_record_external_id_hash", "left")
    )
    cust_fdbck_hh = cust_fdbck_hh.join(cust_dim, F.col("ngc_customer_id") == F.col(
        "household_id_ori"), "left").fillna(0, subset=["cust"]).dropDuplicates()
    cust_fdbck_hh = (
        cust_fdbck_hh.withColumn(
            "household_id",
            F.when(F.col("cust") == 1, F.col("household_id_ori")
                   ).otherwise("household_id"),
        )
        .withColumn(
            "golden_record_external_id_hash",
            F.when(
                F.col("cust") == 1, F.col("golden_record_external_id_hash_ori")
            ).otherwise("golden_record_external_id_hash"),
        )
        .drop("household_id_ori", "golden_record_external_id_hash_ori", "cust")
    )
    return cust_fdbck_hh


def test():
    print("test")


class Test:
    def __init__(self):
        pass

    def show_hello(self):
        print("hello")
