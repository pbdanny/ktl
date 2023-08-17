# Databricks notebook source
from utils import files
from utils import etl
from etl import staging

import pandas as pd
import pyspark.sql.functions as F

import unittest

import os

# COMMAND ----------

conf_mapper = files.conf_reader("../config/etl.json")
mnt_mapper = files.conf_reader("../config/mnt.json")
abfss_prefix, dbfs_path = (mnt_mapper["abfss_prefix"], mnt_mapper["dbfs_path"])

# COMMAND ----------

def gen(spark, dbfs_path, insurer, conf_mapper, fdbk_type):
    from utils import etl
    fdbk_df, filename = etl.get_df(
        dbfs_path, insurer, conf_mapper, fdbk_type=fdbk_type)
    # cust_fdbck_hh = etl.ngc_to_household(
    #     spark, spark.createDataFrame(fdbk_df, schema=None))
    return spark.createDataFrame(fdbk_df, schema=None)

# COMMAND ----------

# use latest delta instead
insurer = "AZAY"
feedback1_azay = gen(spark, dbfs_path, insurer, conf_mapper, 1)
feedback2_azay = gen(spark, dbfs_path, insurer, conf_mapper, 2)
insurer = "CIGNA"
feedback1_cigna = gen(spark, dbfs_path, insurer, conf_mapper, 1)
feedback2_cigna = gen(spark, dbfs_path, insurer, conf_mapper, 2)

# COMMAND ----------

mapper_path = "/dbfs/FileStore/niti/LMP/Lead_mapper.csv"
lead_map = pd.read_csv(mapper_path)
lead_map = spark.createDataFrame(lead_map)

# COMMAND ----------

# concat lot for azay
feedback1_azay = feedback1_azay.withColumn("Filename", F.split("src_file",".txt")[0])
feedback1_azay = feedback1_azay.withColumn("lot", F.split("Filename","_")[4])
feedback2_azay = feedback2_azay.withColumn("Filename", F.split("src_file",".txt")[0])
feedback2_azay = feedback2_azay.withColumn("lot", F.split("Filename","_")[4])

# COMMAND ----------

# concat lot for lot
feedback1_cigna = feedback1_cigna.withColumn("Filename", F.split("src_file",".txt")[0])
feedback1_cigna = feedback1_cigna.withColumn("lot", F.split("Filename","_")[4])
feedback2_cigna = feedback2_cigna.withColumn("Filename", F.split("src_file",".txt")[0])
feedback2_cigna = feedback2_cigna.withColumn("lot", F.split("Filename","_")[4])

# COMMAND ----------

feedback1_azay = feedback1_azay.filter(F.col("lot").isin(["202209","202210","202211"]))
feedback2_azay = feedback2_azay.filter(F.col("lot").isin(["202209","202210","202211"]))

# COMMAND ----------

feedback1_cigna = feedback1_cigna.filter(F.col("lot").isin(["202209","202210","202211"]))
feedback2_cigna = feedback2_cigna.filter(F.col("lot").isin(["202209","202210","202211"]))

# COMMAND ----------

def joined_feedback(feedback1, feedback2):
    feedback = feedback1.select(
        "dib_cust_code",
        "lead_type",
        "Contact_Status",
        "Bought_Status",
        "lead_shop_payment_type",
        "lot",
    ).join(
        feedback2.distinct(),
        ["dib_cust_code", "lot"],
        "left",
    )
    return feedback

# COMMAND ----------

feedback_azay = joined_feedback(feedback1_azay, feedback2_azay)
feedback_azay.filter(F.col("Payment_Type").isNull()).display()

# COMMAND ----------

feedback_cigna = joined_feedback(feedback1_cigna, feedback2_cigna)
feedback_cigna.filter(F.col("Payment_Type").isNull()).display()

# COMMAND ----------

"""
High index
Life-stage = Family xx
Card aging = 
Life Cycle = Growing 
Tru Price = Price insensitive
Preferred store format = HDE
Preferred store region = BKK
Gender = Female

"""

# COMMAND ----------

model_outbound_path = {
  "lot1": "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/thanakrit/lmp/insurance_lead/edm_202207/outbound_edm202207.parquet",
  "lot2": "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/edm_202210/lead/outbound/all_outbound.parquet",
  "lot3": "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/edm_202301/lead/outbound/all_outbound.parquet"
}

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
txn_cust_r360 = txn_cust.where(F.col("date_id").between(F.col("r360_start_date"), F.col("first_contact_date")))

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

lot1 = spark.read.parquet(model_outbound_path["lot1"]).select("household_id", F.col("InsurerName"), F.col("CashorCreditCard"), F.col("LeadType"), F.col("LeadDate"))

# COMMAND ----------

lot1.display()

# COMMAND ----------

index_lists = [
    "lifestage_seg_name",
    "card_aging_range",
    "lifecycle_name",
    "truprice_seg_desc",
    "facts_seg_desc",
    "pref_store_format",
    "pref_store_region",
]

# COMMAND ----------

def index_performance(cust, feedback, group):
    tot = cust.count()
    prior = (
        cust.groupby(group)
        .count()
        .withColumnRenamed("count", "cnt_per_group1")
        .withColumn("perc_of_count_total1", (F.col("cnt_per_group1") / tot))
    )
    cust_feedback = feedback.select(
        F.col("dib_cust_code").alias("household_id"), "bought_status"
    ).join(cust, "household_id", "inner")
    
    cust_feedback = cust_feedback.filter(F.col("bought_status") == "Bought")
    tot = cust_feedback.count()
    lead = (
        cust_feedback.groupby(group)
        .count()
        .withColumnRenamed("count", "cnt_per_group2")
        .withColumn("perc_of_count_total2", (F.col("cnt_per_group2") / tot))
    )
    prior = prior.join(lead, group, "left")
    prior = prior.withColumn("index", F.col("perc_of_count_total2")/F.col("perc_of_count_total1"))
    return prior

# COMMAND ----------

cust

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName")=="AAC").join(cust, "household_id", "inner")

# COMMAND ----------

cust_lot1.count()

# COMMAND ----------

feedback_azay.filter(F.col("lead_type")!="Pom").select(F.col("dib_cust_code").alias("household_id"))

# COMMAND ----------

cust_lot1 = cust_lot1.join(feedback_azay.filter(F.col("lead_type")!="Pom").select(F.col("dib_cust_code").alias("household_id")), "household_id", "inner")

# COMMAND ----------

cust_lot1.count()

# COMMAND ----------

for elum in index_lists:
    _index = index_performance(cust_lot1, feedback_azay, elum)
    _index.sort("index",ascending=False).display()

# COMMAND ----------

product_grpabfss_prefix = "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/thanakrit/lmp/insurance_lead/"
prd_gr_ver = "202205"
product_grp = spark.read.parquet(os.path.join(
abfss_prefix, "product_group", f"prod_grp_{prd_gr_ver}.parquet"))

# COMMAND ----------

product_grp.display()

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

cust_lot1.filter(F.col("pref_store_region") == "West").groupby(
    "pref_store_id"
).count().join(dim_store, F.col("pref_store_id")==F.col("store_id"), "left").sort("count", ascending=False).display()

# COMMAND ----------

pref_store_ids =  cust_lot1.filter(F.col("pref_store_region") == "West").select(
    "pref_store_id"
).rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

_index = index_performance(cust_lot1, feedback_azay, "pref_store_id")
_index = _index.filter(F.col("pref_store_id").isin(pref_store_ids))
_index.sort("index",ascending=False).display()

# COMMAND ----------

_index = index_performance(
    cust_lot1.filter(F.col("pref_store_region") == "West"),
    feedback_azay,
    "pref_store_id",
)
_index.sort("index", ascending=False).display()
_index.filter(F.col("index")>=1.2).sort(["cnt_per_group2", "index"], ascending=False).display()

# COMMAND ----------

dim_store.filter(F.col("store_id").isin([2589, 2795, 2152, 1765, 3125])).display()
dim_store.filter(F.col("store_id").isin([5078, 5055, 5181, 6435, 6418])).display()

# COMMAND ----------


