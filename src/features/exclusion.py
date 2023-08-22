from pyspark.sql import functions as F
from pathlib import Path
import os
import sys
sys.path.append('../')

from utils.logger import logger

@logger
def main(spark, df, test=False):
    '''
        main
        :param:
            df: dataframe from cust_detail_wk from quarter_recency (recency.py)
    '''
    cust_w_consent = consent(spark, df)
    cust_reg_10yr = old_account(cust_w_consent)
    cust_detail_ex_trdr = trader(spark, cust_reg_10yr, True, test=test)
    cust_detail_seg(spark,cust_detail_ex_trdr, True, test=test)
    del [cust_w_consent, cust_reg_10yr, cust_detail_ex_trdr]

@logger
def consent(spark, df):
    '''
        consent

        :param:
            df: dataframe from cust_detail_wk from quarter_recency (recency.py)
    '''
    prsn_acct = spark.table("stdm.personaccount")
    prsn_lmp_call_ok = \
        (prsn_acct
         .where(F.col("HashedCustomerId") == F.col("GoldenExternalId"))
         .where(F.col("customerstatus") == "Active")
         .where(F.col("LotusMoneyServices") == "true")
         .where(F.col("PersonDoNotCall") == "false")
         .select("GoldenExternalId")
         .withColumnRenamed("GoldenExternalId", "golden_record_external_id_hash")
         .drop_duplicates()
         )

    cust_dim = spark.table("tdm.v_customer_dim").select(
        "household_id", "golden_record_external_id_hash").drop_duplicates()

    hh_id_lmp_call_ok = cust_dim.join(
        prsn_lmp_call_ok, "golden_record_external_id_hash")

    print(hh_id_lmp_call_ok)

    cust_w_consent = \
        (df
         .join(hh_id_lmp_call_ok.select("household_id").drop_duplicates(), "household_id", "inner")
         )

    return cust_w_consent

@logger
def old_account(df, YEAR=10):
    '''
        old account
        :param:  
            df, dataframe from cust_w_consent (consent)
    '''
    cust_reg_10yr = df.where(F.col("day_regis") <= YEAR*360)
    return cust_reg_10yr

@logger
def trader(spark, df, save=True, test=False):
    '''
        trader
        :param: 
            df, dataframe from cust_reg_10yr (old_account)
    '''
    from utils import files
    trader = spark.table("tdm_seg.trader_subseg_party2_master").select(
        "household_id", "mapping_quarter_id").withColumnRenamed("mapping_quarter_id", "quarter_id")

    cust_detail_ex_trdr = df.join(
        trader, ["household_id", "quarter_id"], "leftanti")

    if save:
        mnt_mapper = files.conf_reader("../config/mnt.json")
        abfss_prefix, dbfs_path = (
            mnt_mapper["abfss_prefix"], Path(mnt_mapper["dbfs_path"]))
        storage_fdbck_abfss_prefix = os.path.join(
            abfss_prefix, "feedback", "storage")
        filename = "cust_details.parquet" if not test else "cust_details_test.parquet"
        print(f">>> filename: {filename}")
        save_path = os.path.join(
            storage_fdbck_abfss_prefix, filename)
        files.save(cust_detail_ex_trdr, save_path, "parquet",
                   "overwrite", overwriteSchema=True)
    return cust_detail_ex_trdr

@logger
def cust_detail_seg(spark, df, save=True, test=False):
    '''
        cust_detail_seg
        :param:
            df, dataframe from trader (trader) cust_detail_ex_trdr
    '''
    from utils import segmentation, files
    truprice = segmentation.truprice(spark)
    facts = segmentation.facts(spark)
    lifestage = segmentation.lifestage(spark)
    prefer_store = segmentation.prefer_store(spark)
    lifecycle = segmentation.lifecycle(spark)
    cust_detail_seg = \
        (df
         .join(truprice, on=["household_id", "period_id"], how="left")
         .fillna(value="Unclassified", subset=["truprice_seg_desc"])

         .join(facts, on=["household_id", "week_id"], how="left")
         .fillna(value="Unclassified", subset=["facts_seg_desc"])

         .join(lifestage, on=["household_id", "quarter_id"], how="left")
         .fillna(value="Unclassified", subset=["lifestage_seg_name"])

         .join(prefer_store, on=["household_id", "period_id"], how="left")
         .fillna(value="NA", subset=["pref_store_id", "pref_store_format", "pref_store_region"])
         .withColumn("pref_store_format", F.when(F.col("pref_store_format") == "missing", "NA").otherwise(F.col("pref_store_format")))
         .withColumn("perf_store_area", F.when(F.col("pref_store_region").isin(["BKK"]), "BKK")
                     .when(F.col("pref_store_region").isin(["NA"]), "NA").otherwise(F.lit("UPC")))
         )
    cust_detail_seg_all = cust_detail_seg.join(lifecycle, on=["household_id", "week_id"], how="left").fillna(
        value="Unclassified", subset=["lifecycle_name"])
    if save:
        mnt_mapper = files.conf_reader("../config/mnt.json")
        abfss_prefix, dbfs_path = (
            mnt_mapper["abfss_prefix"], Path(mnt_mapper["dbfs_path"]))
        storage_fdbck_abfss_prefix = os.path.join(
            abfss_prefix, "feedback", "storage")
        filename = "cust_details_seg.parquet" if not test else "cust_details_seg_test.parquet"
        print(f">>> filename: {filename}")
        save_path = os.path.join(
            storage_fdbck_abfss_prefix, filename)
        files.save(cust_detail_seg_all, save_path, "parquet",
                   "overwrite", overwriteSchema=True)
