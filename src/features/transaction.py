from utils.logger import logger
import os
import sys
from pyspark.sql import functions as F
from pathlib import Path
import argparse
from datetime import datetime, timedelta

sys.path.append('../')


@logger
def main(spark, dbutils, prjct_nm, txnItem, test=False):
    '''
        main function for preprocessing transaction data
        :param:
            spark, spark session
            dbutils, dbutils (for dbfs)
            prjct_nm, project name e.g. edm_202210
            txnItem, transaction item class from edm_class
            test, test default False
    '''
    from utils import files, segmentation
    conf_tnx = files.conf_reader("../config/transaction.json")
    mnt_mapper = files.conf_reader("../config/mnt.json")
    abfss_prefix, dbfs_path = (
        mnt_mapper["abfss_prefix"], Path(mnt_mapper["dbfs_path"]))

    filename = "cust_details_seg.parquet" if not test else "cust_details_seg_test.parquet"
    dbutils.fs.cp(os.path.join(abfss_prefix, "feedback", "storage", filename),
                  os.path.join(abfss_prefix, prjct_nm, "feedback", filename), True)

    cust_details = spark.read.parquet(os.path.join(
        abfss_prefix, prjct_nm, "feedback", filename))

    MIN_FRST_CNTCT_DATE = cust_details.agg(
        F.min("first_contact_date")).collect()[0][0]
    MAX_FRST_CNTCT_DATE = cust_details.agg(
        F.max("first_contact_date")).collect()[0][0]
    # log transaction first contact range
    print(
        f">>> Min first contact date : {MIN_FRST_CNTCT_DATE.strftime('%Y-%m-%d')}")
    print(
        f">>> Max first contact date : {MAX_FRST_CNTCT_DATE.strftime('%Y-%m-%d')}")

    MAX_BACK_DAY = conf_tnx["MAX_BACK_DAY"]
    MIN_DATA_DATE = MIN_FRST_CNTCT_DATE - timedelta(days=MAX_BACK_DAY)
    MAX_DATA_DATE = MAX_FRST_CNTCT_DATE
    if test:
        MIN_DATA_DATE = datetime.strptime(
            conf_tnx["test"]["MIN_DATA_DATE"], '%Y-%m-%d')
        MAX_DATA_DATE = datetime.strptime(
            conf_tnx["test"]["MAX_DATA_DATE"], '%Y-%m-%d')
    # log transaction date range
    print(f">>> Min data date : {MIN_DATA_DATE.strftime('%Y-%m-%d')}")
    print(f">>> Max data date : {MAX_DATA_DATE.strftime('%Y-%m-%d')}")

    # Min, Max fis_week_id
    date_dim = spark.table("tdm.v_date_dim")
    MIN_DATA_WEEK_ID = date_dim.where(F.col("date_id") == MIN_DATA_DATE).select(
        "week_id").drop_duplicates().collect()[0][0]
    MAX_DATA_WEEK_ID = date_dim.where(F.col("date_id") == MAX_DATA_DATE).select(
        "week_id").drop_duplicates().collect()[0][0]
    print(f">>> Min data week_id : {MIN_DATA_WEEK_ID}")
    print(f">>> Max date week_id : {MAX_DATA_WEEK_ID}")

    # Total week of data pulled (week of max date ) - (week of min date) + 1
    NUM_WEEK_DATA = date_dim.where(F.col("date_id") == MAX_DATA_DATE).select("week_num_sequence").drop_duplicates().collect()[
        0][0] - date_dim.where(F.col("date_id") == MIN_DATA_DATE).select("week_num_sequence").drop_duplicates().collect()[0][0] + 1
    print(f">>> Number of data (week) : {NUM_WEEK_DATA}")
    txn_snap = txnItem(end_wk_id=MAX_DATA_WEEK_ID,
                       range_n_week=NUM_WEEK_DATA,
                       customer_data="CC",
                       item_col_select=['transaction_uid', 'store_id', 'date_id', 'upc_id', 'week_id',
                                        'net_spend_amt', 'unit', 'customer_id', 'discount_amt', 'tran_datetime'])
    txn_cc = txn_snap.get_txn()
    # rename unit to pkg_weight_unit
    txn_cc = txn_cc.withColumnRenamed("unit", "pkg_weight_unit")
    txn_cc = txn_cc.withColumn("transaction_concat", F.concat(
        F.col("transaction_uid_orig"), F.col("store_id"), F.col("date_id")))
    txn_cc = txn_cc.join(cust_details.select(
        "household_id").drop_duplicates(), "household_id", "inner").checkpoint()

    # log transaction count
    print(">>> Transaction count")
    print(txn_cc
          .agg(F.count_distinct("household_id").alias("n_hh_id"),
               F.count_distinct("transaction_uid").alias("n_bask")).show()
          )
    single_tndr_typ = segmentation.single_tender(
        spark, MIN_DATA_DATE=MIN_DATA_DATE, MAX_DATA_DATE=MAX_DATA_DATE)
    txn_single_tndr = txn_cc.join(single_tndr_typ, "transaction_uid", "inner")
    shp_ms = segmentation.shopping_mission(
        spark, MIN_DATA_WEEK_ID=MIN_DATA_WEEK_ID, MAX_DATA_WEEK_ID=MAX_DATA_WEEK_ID)
    txn_tndr_shp_ms = txn_single_tndr.join(
        shp_ms.drop("transaction_uid"), on=["transaction_concat", "week_id"], how="left")
    filename = "txn_cc_sngl_tndr.parquet" if not test else "txn_cc_sngl_tndr_test.parquet"

    files.save(txn_tndr_shp_ms, os.path.join(abfss_prefix, prjct_nm,
               filename), format="parquet", mode="overwrite", overwriteSchema=True)


if __name__ == "__main__":
    '''
        python features/ground_truth.py -d 2020-12-31 -t
    '''
    parser = argparse.ArgumentParser()
    # main() arguments parser later used in main()
