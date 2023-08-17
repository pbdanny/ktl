# Databricks notebook source
from utils import files
from utils import etl
from etl import staging
import os

import pandas as pd
import pyspark.sql.functions as F

import unittest
from evaluation import contactable

# COMMAND ----------

# MAGIC %md 
# MAGIC # prep feedback lot data

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

index_lists = [
    "lifestage_seg_name",
    "card_aging_range",
    "lifecycle_name",
    "truprice_seg_desc",
    "facts_seg_desc",
    "pref_store_format",
    "pref_store_region",
    "ModelScoreBucket",
]

# COMMAND ----------

def gen(spark, dbfs_path, insurer, conf_mapper, fdbk_type):
    from utils import etl

    fdbk_df, filename = etl.get_df(dbfs_path, insurer, conf_mapper, fdbk_type=fdbk_type)
    return spark.createDataFrame(fdbk_df, schema=None)


def joined_feedback(feedback1, feedback2):
    feedback = feedback1.select(
        "dib_cust_code",
        "lead_type",
        "Contact_Status",
        "Bought_Status",
        "lead_shop_payment_type",
        "model_score_bucket",
        "lot",
    ).join(
        feedback2.distinct(),
        ["dib_cust_code", "lot"],
        "left",
    )
    return feedback


def index_performance(cust, feedback, group):
    tot = cust.filter((F.col(group) != "Unclassified") & (F.col(group) != "NA")).count()
    prior = (
        cust.filter((F.col(group) != "Unclassified") & (F.col(group) != "NA"))
        .groupby(group)
        .count()
        .withColumnRenamed("count", "cnt_per_group1")
        .withColumn("perc_of_count_total1", (F.col("cnt_per_group1") / tot))
    )
    cust_feedback = feedback.select(
        F.col("dib_cust_code").alias("household_id"), "bought_status"
    ).join(cust, "household_id", "inner")

    cust_feedback = cust_feedback.filter(F.col("bought_status") == "Bought")
    tot = cust_feedback.filter((F.col(group) != "Unclassified") & (F.col(group) != "NA")).count()
    lead = (
        cust_feedback.filter((F.col(group) != "Unclassified") & (F.col(group) != "NA"))
        .groupby(group)
        .count()
        .withColumnRenamed("count", "cnt_per_group2")
        .withColumn("perc_of_count_total2", (F.col("cnt_per_group2") / tot))
    )
    prior = prior.join(lead, group, "left")
    prior = prior.withColumn(
        "index", F.col("perc_of_count_total2") / F.col("perc_of_count_total1")
    )
    return prior

# COMMAND ----------

class Evaluation:
    def __init__(self, df):
        """
        df: dataframe that include all feedback 1 and feedback 2
        """
        self.df = df.filter(F.col("lead_type").isin(["Hot", "Normal"]))
        self.total = self.df.count()
        self.lead_type = self.df.select("lead_type").distinct().rdd.flatMap(lambda x: x).collect()
        self.lead_shop_payment_type = self.df.select("lead_shop_payment_type").distinct().rdd.flatMap(lambda x: x).collect()


    def contactable(self, filtered=[], cols=[]):
        contact_rate = self.df
        for col in filtered:
            contact_rate = contact_rate.filter(F.col(col).isNotNull())
        total = contact_rate.count()
        contact_rate = contact_rate.groupby("Contact_Status", *cols).count()
        contact_rate = contact_rate.withColumn("ration", F.col("count") / total)
        return contact_rate

    def get_cust(
        self, bought=None, lead_type: str = None, lead_shop_payment_type: str = None
    ):
        tmp_df = self.df
        if bought is not None:
            if bought:
                tmp_df = tmp_df.filter(F.col("Bought_Status") == "Bought")
            else:
                tmp_df = tmp_df.filter(F.col("Bought_Status") != "Bought")
        if lead_type is not None:
            tmp_df = tmp_df.filter(F.col("lead_type") == lead_type)
        if lead_shop_payment_type is not None:
            tmp_df = tmp_df.filter(
                F.col("lead_shop_payment_type") == lead_shop_payment_type
            )
        return tmp_df

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

"""
Load customer segment details
Load snap txn : only MyLo, only customer in feedback 
53weeks back, filter only single tender 
"""
abfss_prefix = "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/thanakrit/lmp/insurance_lead"
prjct_nm = "edm_202210"
stage = "lead"
substage = "features_agg"

txn_cust = spark.read.parquet(os.path.join(abfss_prefix, prjct_nm, stage, "txn_cc_sngl_tndr.parquet"))

txn_cust_r360 = txn_cust.where(F.col("date_id").between(F.col("r360_start_date"), F.col("data_end_date")))
txn_cust_r360.display()

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

lot1 = spark.read.parquet(model_outbound_path["lot1"]).select(
    "household_id",
    F.col("InsurerName"),
    F.col("CashorCreditCard"),
    F.col("LeadType"),
    F.col("LeadDate"),
    F.col("ModelScoreBucket"),
)
lot2 = spark.read.parquet(model_outbound_path["lot2"]).select(
    "household_id",
    F.col("InsurerName"),
    F.col("CashorCreditCard"),
    F.col("LeadType"),
    F.col("LeadDate"),
    F.col("ModelScoreBucket"),
)
lot3 = spark.read.parquet(model_outbound_path["lot3"]).select(
    "household_id",
    F.col("InsurerName"),
    F.col("CashorCreditCard"),
    F.col("LeadType"),
    F.col("LeadDate"),
    F.col("ModelScoreBucket"),
)

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

# concat lot for lot
feedback1_cigna = feedback1_cigna.withColumn("Filename", F.split("src_file",".txt")[0])
feedback1_cigna = feedback1_cigna.withColumn("lot", F.split("Filename","_")[4])
feedback2_cigna = feedback2_cigna.withColumn("Filename", F.split("src_file",".txt")[0])
feedback2_cigna = feedback2_cigna.withColumn("lot", F.split("Filename","_")[4])

# COMMAND ----------

lot_list = ["202212", "202301", "202302"]
feedback1_azay = feedback1_azay.filter(
    F.col("lot").isin(lot_list)
)
feedback2_azay = feedback2_azay.filter(
    F.col("lot").isin(lot_list)
)

feedback1_cigna = feedback1_cigna.filter(
    F.col("lot").isin(lot_list)
)
feedback2_cigna = feedback2_cigna.filter(
    F.col("lot").isin(lot_list)
)

# COMMAND ----------

feedback_azay = joined_feedback(feedback1_azay, feedback2_azay)
feedback_azay.filter(F.col("Payment_Type").isNull()).display()

feedback_cigna = joined_feedback(feedback1_cigna, feedback2_cigna)
feedback_cigna.filter(F.col("Payment_Type").isNull()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluation

# COMMAND ----------

# lead requirement facts
lot_number = 1

# COMMAND ----------


