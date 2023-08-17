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
    tot = cust_feedback.filter(
        (F.col(group) != "Unclassified") & (F.col(group) != "NA")
    ).count()
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


def conversion_group(cust, feedback, group):
    prior = (
        cust.filter((F.col(group) != "Unclassified") & (F.col(group) != "NA"))
        .groupby(group)
        .count()
        .withColumnRenamed("count", "cnt_per_group1")
    )
    prior.display()
    cust_feedback = feedback.select(
        F.col("dib_cust_code").alias("household_id"),
        "contact_status",
        "bought_status",
        "payment_type",
    ).join(cust, "household_id", "inner")

    for x in [None, "contact_status", "bought_status"]:
        lead = cust_feedback.filter(
            (F.col(group) != "Unclassified") & (F.col(group) != "NA")
        )
        if x:
            gps = [group, x]
            lead = lead.filter(F.col("contact_status") == "Contactable")
            if x == "bought_status":
                lead = lead.filter(F.col("bought_status") == "Bought")
        else:
            gps = group
        lead = lead.groupby(gps).count().withColumnRenamed("count", "cnt_per_group")
        lead.display()

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

feedback1_azay = feedback1_azay.filter(
    F.col("lot").isin(["202209", "202210", "202211"])
)
feedback2_azay = feedback2_azay.filter(
    F.col("lot").isin(["202209", "202210", "202211"])
)

feedback1_cigna = feedback1_cigna.filter(
    F.col("lot").isin(["202209", "202210", "202211"])
)
feedback2_cigna = feedback2_cigna.filter(
    F.col("lot").isin(["202209", "202210", "202211"])
)

# COMMAND ----------

feedback_azay = joined_feedback(feedback1_azay, feedback2_azay)
feedback_azay.filter(F.col("Payment_Type").isNull()).display()

feedback_cigna = joined_feedback(feedback1_cigna, feedback2_cigna)
feedback_cigna.filter(F.col("Payment_Type").isNull()).display()

# COMMAND ----------



# COMMAND ----------

evaluate = Evaluation(feedback_azay)

# COMMAND ----------

print(evaluate.df.count())
evaluate.contactable(cols=["lead_type"]).display()

# COMMAND ----------

def eval_pipe(feedback):
    evaluate = Evaluation(feedback)
    result_dict = {}
    print("Total", evaluate.total)
    result_dict["Total"] = evaluate.total
    for lead_type in evaluate.lead_type:
        val = evaluate.get_cust(lead_type=lead_type).select("dib_cust_code").count()
        print(lead_type, val)
        result_dict[lead_type] = val
    for lead_type in evaluate.lead_type:
        for lead_shop_payment_type in evaluate.lead_shop_payment_type:
            val = (
                evaluate.get_cust(
                    lead_type=lead_type, lead_shop_payment_type=lead_shop_payment_type
                )
                .select("dib_cust_code")
                .count()
            )
            print(lead_type, lead_shop_payment_type)
            print(val)
            result_dict[f"{lead_type} {lead_shop_payment_type}"] = val
    lol= list(map(list, result_dict.items()))
    df = spark.createDataFrame(lol, ["key", "values"])
    df.display()
    print("contactable")
    evaluate.contactable().display()
    print("contactable x lead_type")
    evaluate.contactable(cols=["lead_type"]).display()
    print("contact x lead_type x bought")
    evaluate.contactable(cols=["Bought_Status", "lead_type"]).display()
    print("contact x lead_type x bought x payment confirm")
    evaluate.contactable(cols=["Bought_Status", "lead_type", "Payment_Type"]).display()

    print("contact x lead_type x bought x payment confirm x model_score_bucket")
    evaluate.contactable(cols=["Bought_Status", "lead_type", "Payment_Type", "model_score_bucket"]).display()
    

    print("contactable x lead_shop_payment_type")
    evaluate.contactable(cols=["lead_shop_payment_type"]).display()
    print("contact x lead_shop_payment_type x bought")
    evaluate.contactable(cols=["Bought_Status", "lead_shop_payment_type"]).display()
    print("contact x lead_shop_payment_type x bought x payment confirm")
    evaluate.contactable(cols=["Bought_Status", "lead_shop_payment_type", "Payment_Type"]).display()

    print("contact x lead_shop_payment_type x bought x payment confirm x model_score_bucket")
    evaluate.contactable(cols=["Bought_Status", "lead_shop_payment_type", "Payment_Type", "model_score_bucket"]).display()
    


    print("actual bough status")
    evaluate.contactable(
        cols=["Bought_Status", "lead_type", "lead_shop_payment_type", "Payment_Type"],
    ).display()
    evaluate.contactable(
        filtered=["Payment_Type"],
        cols=["Bought_Status", "lead_type", "lead_shop_payment_type", "Payment_Type"],
    ).display()
    evaluate.contactable(
        filtered=["Payment_Type"],
        cols=[
            "Bought_Status",
            "lead_type",
            "Payment_Type",
            "Premium_Frequency",
            "Policy_type_self",
            "Policy_plan_self",
        ],
    ).display()

    evaluate.contactable(
        filtered=["Bought_Status", "Payment_Type"],
        cols=[
            "Bought_Status",
            "lead_type",
            "Payment_Type",
            "Premium_Frequency",
        ],
    ).display()

    evaluate.contactable(
        filtered=["Bought_Status", "Payment_Type"],
        cols=[
            "Bought_Status",
            "lead_type",
            "Payment_Type",
            "Policy_type_self",
            "Policy_plan_self",
        ],
    ).display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## azay

# COMMAND ----------

eval_pipe(feedback_azay)

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName")=="AAC").join(cust, "household_id", "inner")


# COMMAND ----------

cust_lot1.groupBy("ModelScoreBucket").count().display()

# COMMAND ----------

cust_lot1.groupby("LeadType").agg(F.countDistinct("household_id")).display()

# COMMAND ----------

cust_lot1.groupby("LeadType", "CashorCreditCard").agg(
    F.countDistinct("household_id")
).display()

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName")=="AAC").join(cust, "household_id", "inner")
for elum in index_lists:
  print(elum)
  conversion_group(cust_lot1, feedback_azay, elum)

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName")=="AAC").join(cust, "household_id", "inner")
for elum in index_lists:
    _index = index_performance(cust_lot1, feedback_azay, elum)
    _index.sort("index",ascending=False).display()

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName") == "AAC").join(
    cust, "household_id", "inner"
)
_index = index_performance(
    cust_lot1.filter(F.col("pref_store_region") == "West"),
    feedback_azay,
    "pref_store_id",
)
_index.join(dim_store, F.col("pref_store_id") == F.col("store_id"), "left").sort(
    "index", ascending=False
).display()
_index.filter(F.col("index") >= 1.2).join(
    dim_store, F.col("pref_store_id") == F.col("store_id"), "left"
).sort(["cnt_per_group2", "index"], ascending=False).display()

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName") == "AAC").join(
    cust, "household_id", "inner"
)
for elum in index_lists:
    _index = index_performance(
        cust_lot1, feedback_azay.filter(F.col("Payment_Type").isNotNull()), elum
    )
    _index.sort("index", ascending=False).display()

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName") == "AAC").join(
    cust, "household_id", "inner"
)
for elum in index_lists:
    _index = index_performance(
        cust_lot1,
        feedback_azay.filter(
            (F.col("Payment_Type").isNotNull())
            & (F.col("Policy_type_self").isNotNull())
        ),
        elum,
    )
    _index.sort("index", ascending=False).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## cigna

# COMMAND ----------

evaluate = Evaluation(feedback_cigna)

# COMMAND ----------

feedback_cigna.display()

# COMMAND ----------

feedback_cigna.count()

# COMMAND ----------

evaluate.df.count()

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName")=="CIG").join(cust, "household_id", "inner")

# COMMAND ----------

cust_lot1.groupby("LeadType").agg(F.countDistinct("household_id")).display()

# COMMAND ----------

cust_lot1.groupby("LeadType", "CashorCreditCard").agg(
    F.countDistinct("household_id")
).display()

# COMMAND ----------

cust_lot1.display()

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName")=="CIG").join(cust, "household_id", "inner")
feedback_cigna_new = feedback_cigna.drop("lead_shop_payment_type").join(
    cust_lot1.select(
        F.col("household_id").alias("dib_cust_code"),
        F.col("CashorCreditCard").alias("lead_shop_payment_type"),
    ),
    "dib_cust_code",
    "inner",
)

# COMMAND ----------

feedback_cigna_new.display()

# COMMAND ----------

eval_pipe(feedback_cigna_new)

# COMMAND ----------

def conversion_group2(cust, feedback, group):
    prior = (
        cust.filter((F.col(group) != "Unclassified") & (F.col(group) != "NA"))
        .groupby(group)
        .count()
        .withColumnRenamed("count", "cnt_per_group1")
    )
    prior.display()
    cust_feedback = feedback.select(
        F.col("dib_cust_code").alias("household_id"),
        "contact_status",
        "bought_status",
        "payment_type",
    ).join(cust, "household_id", "inner")
    cust_feedback.display()
    for x in [None, "contact_status", "bought_status"]:
        lead = cust_feedback.filter(
            (F.col(group) != "Unclassified") & (F.col(group) != "NA")
        )
        if x:
            gps = [group, x]
            lead = lead.filter(F.col("contact_status") == "CONTACT")
            if x == "bought_status":
                lead = lead.filter(F.col("bought_status") == "Bought")
        else:
            gps = group
        lead = lead.groupby(gps).count().withColumnRenamed("count", "cnt_per_group")
        lead.display()

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName")=="CIG").join(cust, "household_id", "inner")
for elum in index_lists:
  print(elum)
  conversion_group2(cust_lot1, feedback_cigna_new, elum)

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName")=="CIG").join(cust, "household_id", "inner")
for elum in index_lists:
    _index = index_performance(cust_lot1, feedback_cigna_new, elum)
    _index.sort("index",ascending=False).display()

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName")=="CIG").join(cust, "household_id", "inner")
_index = index_performance(
    cust_lot1.filter(F.col("pref_store_region") == "East"),
    feedback_cigna_new,
    "pref_store_id",
)
_index.join(dim_store, F.col("pref_store_id") == F.col("store_id"), "left").sort(
    "index", ascending=False
).display()
_index.filter(F.col("index") >= 1.2).join(
    dim_store, F.col("pref_store_id") == F.col("store_id"), "left"
).sort(["cnt_per_group2", "index"], ascending=False).display()

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName") == "CIG").join(
    cust, "household_id", "inner"
)
for elum in index_lists:
    _index = index_performance(
        cust_lot1, feedback_cigna_new.filter(F.col("Payment_Type").isNotNull()), elum
    )
    _index.sort("index", ascending=False).display()

# COMMAND ----------

cust_lot1 = lot1.filter(F.col("InsurerName") == "CIG").join(
    cust, "household_id", "inner"
)
for elum in index_lists:
    _index = index_performance(
        cust_lot1,
        feedback_cigna_new.filter(
            (F.col("Payment_Type").isNotNull())
            & (F.col("Policy_type_self").isNotNull())
        ),
        elum,
    )
    _index.sort("index", ascending=False).display()

# COMMAND ----------


