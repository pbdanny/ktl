# Databricks notebook source
from utils import files
from utils import etl
from etl import staging
import os

import pandas as pd
import numpy as np
import pyspark.sql.functions as F

import unittest
from evaluation import contactable

# COMMAND ----------

# MAGIC %md 
# MAGIC # prep

# COMMAND ----------

conf_mapper = files.conf_reader("../config/etl.json")
mnt_mapper = files.conf_reader("../config/mnt.json")
abfss_prefix, dbfs_path = (mnt_mapper["abfss_prefix"], mnt_mapper["dbfs_path"])

# COMMAND ----------

model_outbound_path = {
  "lot1": "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/thanakrit/lmp/insurance_lead/edm_202207/outbound_edm202207.parquet",
  "lot2": "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/edm_202210/lead/outbound/all_outbound.parquet",
  "lot3": "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/edm_202301/lead/outbound/all_outbound.parquet"
}

# COMMAND ----------

division_ids = [1, 2, 3, 4, 9, 10, 13]
format_ids = [1, 2, 3, 4, 5]
dim_store = (
    spark.table("tdm.v_store_dim_c")
    .where(F.col("country") == "th")
    .where(F.col("format_id").isin(format_ids))
    .select("store_id", "format_id", "store_name", "region", "city_name")
)

# COMMAND ----------

# based on lot 1 data
abfss_prefix = "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/thanakrit/lmp/insurance_lead"
prjct_nm = "edm_202207"

prjct_abfss_prefix = os.path.join(abfss_prefix, prjct_nm)
# All txn 
txn = spark.read.parquet(os.path.join(prjct_abfss_prefix, "txn_cc_53wk_seg.parquet"))
# feedback details with start date, end date for each recency
cust = spark.read.parquet(os.path.join(prjct_abfss_prefix, "cust_details_seg.parquet")).drop("week_id", "period_id", "quarter_id")

# Map feedback details with txn
txn_cust = txn.join(cust, on="household_id", how="inner")

# Filter only Txn with date in recency 360
txn_cust_r360_lot1 = txn_cust.where(F.col("date_id").between(F.col("r360_start_date"), F.col("first_contact_date")))

base_lot1 = spark.read.parquet(os.path.join(prjct_abfss_prefix, "all_sec_data_with_doc_avlbl.parquet"))
print(f"Exclude last lot count : {base_lot1.count():,d}")

# COMMAND ----------

txn_cust.select("first_contact_date").display()

# COMMAND ----------

"""
Load customer segment details
Load snap txn : only MyLo, only customer in feedback 
53weeks back, filter only single tender 
"""
abfss_prefix = "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead"
prjct_nm = "edm_202301"
stage = "lead"
substage = "features_agg"

txn_cust = spark.read.parquet(os.path.join(abfss_prefix, prjct_nm, stage, "txn_cc_sngl_tndr.parquet"))
cust_prfl_col = [
    "household_id",
    #              "Bought_Status",
    "card_aging_day",
    "truprice_seg_desc",
    "facts_seg_desc",
    "lifestage_seg_name",
    "lifecycle_name",
    "pref_store_format",
    "pref_store_region",
    "pref_store_id"
]

cust = (
    spark.read.parquet(
        os.path.join(abfss_prefix, prjct_nm, stage, "txn_cc_sngl_tndr.parquet")
    )
    .select(cust_prfl_col)
    .drop_duplicates()
)
txn_cust_r360_lot3 = txn_cust.where(F.col("date_id").between(F.col("r360_start_date"), F.col("data_end_date")))
txn_cust_r360_lot3.display()

stage = "lead"
substage = "alloc"

# exc_last_lot = spark.read.parquet(os.path.join(abfss_prefix, prjct_nm, stage, substage, f"{prjct_nm}_exc_last_lot.parquet"))
base_lot3 = spark.read.parquet(os.path.join(abfss_prefix, prjct_nm, stage, substage, f"{prjct_nm}_exc_exc_lst.parquet"))
print(f"Exclude last lot count : {base_lot3.count():,d}")

# COMMAND ----------

cust = cust.withColumn(
    "card_aging_mth", F.round(F.col("card_aging_day") / 30, 0)
).withColumn(
    "card_aging_range",
    F.when(F.col("card_aging_mth") <= 4, "<=4")
    .when(F.col("card_aging_mth") <= 6, "4-6")
    .when(F.col("card_aging_mth") <= 8, "7-8")
    .when(F.col("card_aging_mth") <= 12, "9-12")
    .when(F.col("card_aging_mth") > 12, ">12")
    .otherwise("NA"),
)

# COMMAND ----------

cust_life_seg = cust.select("household_id", "lifestage_seg_name").drop_duplicates()

# COMMAND ----------

txn_cust.select("data_end_date").display()

# COMMAND ----------

lot1 = spark.read.parquet(model_outbound_path["lot1"])
lot2 = spark.read.parquet(model_outbound_path["lot2"])
lot3 = spark.read.parquet(model_outbound_path["lot3"])

# COMMAND ----------

lot1.agg(F.avg("LifeStyle3")).display()
lot3.agg(F.avg("LifeStyle3")).display()

# COMMAND ----------

cust.count()

# COMMAND ----------

cust.agg(F.countDistinct("household_id")).display()

# COMMAND ----------

lot3.agg(F.count("household_id")).display()

# COMMAND ----------

lot3.join(cust_life_seg, "household_id", "inner").groupBy("household_id").agg(F.count("household_id").alias("cnt")).filter(F.col("cnt")>1).display()

# COMMAND ----------

lot3.filter(F.col("household_id")=="102111060010187381").display()

# COMMAND ----------

cust.filter(F.col("household_id")=="102111060010187381").display()

# COMMAND ----------

lot3.filter(F.col("InsurerName") == "AAC").filter(F.col("LeadType") == "normal").join(
    cust_life_seg, "household_id", "inner"
).groupBy("lifestage_seg_name").agg(F.count("household_id").alias("cnt")).display()

# COMMAND ----------

lot1 = lot1.withColumn("high_spend", F.when(F.col("LifeStyle3")>=3000, 1).otherwise(0))
lot2 = lot2.withColumn("high_spend", F.when(F.col("LifeStyle3")>=3000, 1).otherwise(0))
lot3 = lot3.withColumn("high_spend", F.when(F.col("LifeStyle3")>=3000, 1).otherwise(0))

# COMMAND ----------

lot1.filter(F.col("InsurerName") == "AAC").groupBy("high_spend").count().display()
lot3.filter(F.col("InsurerName") == "AAC").groupBy("high_spend").count().display()

# COMMAND ----------

dbutils.fs.ls("abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/thanakrit/lmp/insurance_lead")

# COMMAND ----------

cust_life_seg = cust.select("household_id", F.col("lifestage_seg_name").alias("life_seg")).drop_duplicates()

# COMMAND ----------

abfss_prefix = "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/thanakrit/lmp/insurance_lead"
prjct_nm = "edm_202210"
stage = "lead"
substage = "features_agg"

trprc_lfstg = spark.read.parquet(os.path.join(abfss_prefix, prjct_nm, stage, substage, "all_feature.parquet")).select("household_id","truprice_seg_desc","lifestage_seg_name").drop_duplicates()

# COMMAND ----------

cust_life_seg.join(trprc_lfstg, "household_id", "inner").groupBy(
    "lifestage_seg_name", "life_seg"
).count().display()

# COMMAND ----------

trprc_lfstg.display()

# COMMAND ----------

abfss_prefix = "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/thanakrit/lmp/insurance_lead"
prjct_nm = "edm_202210"
stage = "lead"
substage = "features_agg"

trprc_lfstg = spark.read.parquet(os.path.join(abfss_prefix, prjct_nm, stage, substage, "all_feature.parquet")).select("household_id","truprice_seg_desc","lifestage_seg_name").drop_duplicates()

# COMMAND ----------

lot2.join(trprc_lfstg, "household_id", "inner").groupBy("lifestage_seg_name").agg(F.count("household_id").alias("cnt")).display()

# COMMAND ----------

txn_cust_r360_lot1.agg(
    F.sum("net_spend_amt").alias("spends"),
    F.countDistinct("household_id").alias("customers"),
).withColumn("spc", F.col("spends") / F.col("customers")).display()
txn_cust_r360_lot3.agg(c
    F.sum("net_spend_amt").alias("spends"),
    F.countDistinct("household_id").alias("customers"),
).withColumn("spc", F.col("spends") / F.col("customers")).display()

# COMMAND ----------

txn_cust_r360_lot1_joined = txn_cust_r360_lot1.join(
    lot1.select("household_id"), "household_id", "inner"
)
txn_cust_r360_lot3_joined = txn_cust_r360_lot3.join(
    lot3.select("household_id"), "household_id", "inner"
)

# COMMAND ----------

txn_cust_r360_lot1_joined.agg(
    F.sum("net_spend_amt").alias("spends"),
    F.countDistinct("household_id").alias("customers"),
).withColumn("spc", F.col("spends") / F.col("customers")).display()
txn_cust_r360_lot3_joined.agg(
    F.sum("net_spend_amt").alias("spends"),
    F.countDistinct("household_id").alias("customers"),
).withColumn("spc", F.col("spends") / F.col("customers")).display()

# COMMAND ----------

base_1_join = base_lot1.join(
    lot1.select("household_id"), "household_id", "inner"
)
base_3_join = base_lot3.join(
    lot3.select("household_id"), "household_id", "inner"
)

# COMMAND ----------

base_1_join.agg(
    F.sum("total_sales_360_days").alias("spends"),
    F.countDistinct("household_id").alias("customers"),
).withColumn("spc", F.col("spends") / F.col("customers")).display()
base_3_join.agg(
    F.sum("total_sales_360_days").alias("spends"),
    F.countDistinct("household_id").alias("customers"),
).withColumn("spc", F.col("spends") / F.col("customers")).display()

# COMMAND ----------

lot1.select("InsurerName").distinct().display()

# COMMAND ----------

lot1.filter(F.col("InsurerName") == "AAC").select("household_id")

# COMMAND ----------

txn_cust_r360_lot1_joined = txn_cust_r360_lot1.join(
    lot1.filter(F.col("InsurerName") == "AAC").select("household_id"),
    "household_id",
    "inner",
)
txn_cust_r360_lot3_joined = txn_cust_r360_lot3.join(
    lot3.filter(F.col("InsurerName") == "AAC").select("household_id"),
    "household_id",
    "inner",
)

# COMMAND ----------

cust_lot1 = txn_cust_r360_lot1_joined.select("transaction_uid", "household_id").drop_duplicates()
cust_lot3 = txn_cust_r360_lot3_joined.select("transaction_uid", "household_id").drop_duplicates()

# COMMAND ----------

txn_lot1_250 = txn_cust_r360_lot1_joined.select(
    "transaction_uid", "net_spend_amt", "household_id"
).groupBy("transaction_uid").agg(F.sum("net_spend_amt").alias("spends")).filter(
    F.col("spends") >= 250
).join(cust_lot1, "transaction_uid", "left")

txn_lot3_250 = txn_cust_r360_lot3_joined.select(
    "transaction_uid", "net_spend_amt", "household_id"
).groupBy("transaction_uid").agg(F.sum("net_spend_amt").alias("spends")).filter(
    F.col("spends") >= 250
).join(cust_lot3, "transaction_uid", "left")

# COMMAND ----------

txn_cust_r360_lot1_joined.agg(
    F.sum("net_spend_amt").alias("spends"),
    F.countDistinct("household_id").alias("customers"),
).withColumn("spc", F.col("spends") / F.col("customers")).display()
txn_cust_r360_lot3_joined.agg(
    F.sum("net_spend_amt").alias("spends"),
    F.countDistinct("household_id").alias("customers"),
).withColumn("spc", F.col("spends") / F.col("customers")).display()

# COMMAND ----------

txn_lot1_250.groupBy(
    "household_id"
).agg(
    F.avg("spends").alias("spends"),
).agg(
    F.avg("spends").alias("spends"),
    F.max("spends").alias("max_spends")
).display()

# COMMAND ----------

txn_lot3_250.groupBy(
    "household_id"
).agg(
    F.avg("spends").alias("spends"),
).agg(
    F.avg("spends").alias("spends"),
    F.max("spends").alias("max_spends")
).display()

# COMMAND ----------

txn_lot1_250.agg(
    F.sum("spends").alias("spends"),
    F.countDistinct("household_id").alias("customers"),
).withColumn("spc", F.col("spends") / F.col("customers")).display()
txn_lot3_250.agg(
    F.sum("spends").alias("spends"),
    F.countDistinct("household_id").alias("customers"),
).withColumn("spc", F.col("spends") / F.col("customers")).display()

# COMMAND ----------


