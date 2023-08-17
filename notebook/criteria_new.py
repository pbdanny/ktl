# Databricks notebook source
# MAGIC %md
# MAGIC # prep

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT tender.ref_no12, SUBSTRING(tender.ref_no12,1,6), bin.ISSUER
# MAGIC FROM tdm.v_resa_group_resa_tran_tender tender
# MAGIC inner join tdm.edm_bin_ccard_master_stg bin
# MAGIC on SUBSTRING(tender.ref_no12,1,6) = bin.BIN
# MAGIC where tender.country = 'th' and tender.dp_data_dt = '2022-11-07'

# COMMAND ----------

import os
from utils import files
from utils import etl
from etl import staging
import pandas as pd
import pyspark.sql.functions as F

import numpy as np
from evaluation import contactable
import matplotlib.pyplot as plt

# COMMAND ----------

def get_cust_gr(txn_cust_r360, store_format_group=None):
    if store_format_group:
      print(store_format_group)
      agg_cust = txn_cust_r360.where(
          F.col("store_format_group").isin(store_format_group)
      )
    else:
        agg_cust = txn_cust_r360
    agg_cust = (
        agg_cust.groupBy("household_id")
        .pivot("tender_type_group")
        .agg(
            F.sum(F.col("net_spend_amt")).alias("sales_360_days"),
            F.countDistinct(F.col("transaction_uid")).alias("visits_360_days"),
        )
        .fillna(value=0)
    )
    lower_case_column_names = [c.lower() for c in agg_cust.columns]
    agg_cust = agg_cust.toDF(*lower_case_column_names)
    return agg_cust


def agg_cust_kpi(txn_cust_r360, agg_cust, feedback):
    agg_cust = (
        agg_cust.withColumn(
            "total_visits_360_days",
            (F.col("cash_visits_360_days") + F.col("ccard_visits_360_days")),
        )
        .withColumn(
            "total_sales_360_days",
            (F.col("cash_sales_360_days") + F.col("ccard_sales_360_days")),
        )
        .withColumn(
            "ratio_visit_by_card_360_days",
            ((F.col("ccard_visits_360_days") / F.col("total_visits_360_days"))),
        )
        .withColumn(
            "ratio_sales_by_card_360_days",
            ((F.col("ccard_sales_360_days") / F.col("total_sales_360_days"))),
        )
        .fillna(value=0)
        .select(
            "household_id",
            "ratio_visit_by_card_360_days",
            "ratio_sales_by_card_360_days",
        )
    )

    criteria = (
        txn_cust_r360.select("household_id")
        .drop_duplicates()
        .join(agg_cust, "household_id", "outer")
        .fillna(0, subset=["ratio_visit_by_card_360_days"])
    )

    data_join = feedback.join(
        criteria.select(
            "household_id",
            "ratio_visit_by_card_360_days",
            "ratio_sales_by_card_360_days",
        ),
        feedback.ngc_customer_id == criteria.household_id,
        "inner",
    )
    data_join_filtered = data_join.filter(F.col("Payment_Type").isNotNull())
    cust_join_df = data_join_filtered.select(
        "ngc_customer_id",
        "ratio_visit_by_card_360_days",
        "ratio_sales_by_card_360_days",
        "Payment_Type",
    ).toPandas()
    
    return agg_cust, criteria, cust_join_df


def criteria_prop(cust_join_df, lst):
    results = []
    avg_sales = []

    for i in lst:
        cust_join_df_tmp = cust_join_df.copy()
        cust_join_df_tmp["cash_card_customer_flag"] = cust_join_df_tmp[
            "ratio_visit_by_card_360_days"
        ].apply(lambda x: "credit" if x >= i else "cash")

        cust_join_df_tmp = cust_join_df_tmp[
            [
                "ngc_customer_id",
                "cash_card_customer_flag",
                "Payment_Type",
                "ratio_sales_by_card_360_days",
            ]
        ]

        avg_sale = cust_join_df_tmp[
            cust_join_df_tmp["cash_card_customer_flag"] == "credit"
        ]["ratio_sales_by_card_360_days"].mean()
        cust_join_df_tmp = (
            cust_join_df_tmp.groupby(["cash_card_customer_flag", "Payment_Type"])
            .agg("count")
            .reset_index()
        )
        tot = sum(
            cust_join_df_tmp[cust_join_df_tmp["cash_card_customer_flag"] == "credit"][
                "ngc_customer_id"
            ].to_list()
        )
        val = (
            cust_join_df_tmp[
                (cust_join_df_tmp["cash_card_customer_flag"] == "credit")
                & (cust_join_df_tmp["Payment_Type"] == "CREDIT CARD")
            ]["ngc_customer_id"].to_list()[0]
            / tot
        )
        results.append(val)
        avg_sales.append(avg_sale)
        '''
        | flag | payment_type |
        | credit| credit|
        |credit| cash|
        '''
    return results, avg_sales


def criteria_plot(results, avg_sales, lst, label=None):
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = [30, 10]
    plt.rcParams["figure.autolayout"] = True

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(lst, results, "g-")
    ax2.plot(lst, avg_sales, "b-")
    #     ax1.axvline(x=0.50, color="r", label="axvline - full height")
    #     ax1.axvline(x=0.76, color="r", label="axvline - full height")

    ax1.set_xlabel(f"PROP_{label}_CCARD_VISIT")
    ax1.set_ylabel("%actual per total ccard", color="g")
    ax2.set_ylabel("average sales over ccard flag", color="b")

    #     plt.show()
    return ax1

# COMMAND ----------

conf_mapper = files.conf_reader("../config/etl.json")
mnt_mapper = files.conf_reader("../config/mnt.json")
abfss_prefix, dbfs_path = (mnt_mapper["abfss_prefix"], mnt_mapper["dbfs_path"])

# COMMAND ----------

# use latest delta instead
feedback1_azay = contactable.gen(spark, dbfs_path, "AZAY", conf_mapper, 1)
feedback2_azay = contactable.gen(spark, dbfs_path, "AZAY", conf_mapper, 2)
feedback1_cigna = contactable.gen(spark, dbfs_path, "CIGNA", conf_mapper, 1)
feedback2_cigna = contactable.gen(spark, dbfs_path, "CIGNA", conf_mapper, 2)

# COMMAND ----------

# concat lot for azay
feedback1_azay = feedback1_azay.withColumn("Filename", F.split("src_file",".txt")[0])
feedback1_azay = feedback1_azay.withColumn("lot", F.split("Filename","_")[4])
feedback2_azay = feedback2_azay.withColumn("Filename", F.split("src_file",".txt")[0])
feedback2_azay = feedback2_azay.withColumn("lot", F.split("Filename","_")[4])

# COMMAND ----------

# for feeaddback lot 1 only 
feedback1_azay = feedback1_azay.filter(F.col("lot").isin(["202209","202210","202211"]))
feedback2_azay = feedback2_azay.filter(F.col("lot").isin(["202209","202210","202211"]))

# COMMAND ----------

# join cigna feedback 1 and feedback 2 together
feedback_azay = feedback1_azay.select(
    "ngc_customer_id", "lead_type", "Bought_Status"
).join(feedback2_azay, "ngc_customer_id", "left")
feedback_azay.filter(F.col("Payment_Type").isNull()).display()

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

txn.filter(F.col("household_id").isNotNull()).agg(F.countDistinct("household_id").alias("cnt")).collect()[0].cnt

# COMMAND ----------

txn.filter((F.col("household_id").isNotNull())&(F.col("store_format_group")=="HDE")).agg(F.countDistinct("household_id").alias("cnt")).collect()[0].cnt

# COMMAND ----------

lst = np.arange(0.01,1.0, 0.01)
lst = [float("{:.2f}".format(x)) for x in lst]

# COMMAND ----------

# MAGIC %md
# MAGIC # code

# COMMAND ----------

agg_cust_hde = get_cust_gr(txn_cust_r360, ["HDE"])
agg_cust_all_format = get_cust_gr(txn_cust_r360)

# COMMAND ----------

print(agg_cust_hde.count())
print(agg_cust_all_format.count())

# COMMAND ----------

cust_gr_hde, criteria_azay, cust_join_df = agg_cust_kpi(
    txn_cust_r360, agg_cust_hde, feedback_azay
)
cust_gr_azay_all, criteria_azay_all, cust_join_all_df = agg_cust_kpi(
    txn_cust_r360, agg_cust_all_format, feedback_azay
)

# COMMAND ----------

cust_join_df

# COMMAND ----------

results, avg_sales = criteria_prop(cust_join_df, lst)

# COMMAND ----------

display(pd.DataFrame([lst, results]).T)

# COMMAND ----------

ax1 = criteria_plot(results, avg_sales, lst, label="HDE")
print(results[19])
print(results[24])
ax1.axhline(y=results[19], color="r", label="axvline - full height")
ax1.axvline(x=0.20, color="r", label="axvline - full height")
ax1.axvline(x=0.50, color="r", label="axvline - full height")
ax1.axvline(x=0.76, color="r", label="axvline - full height")
plt.show()

# COMMAND ----------

results_all, avg_sales_all = criteria_prop(cust_join_all_df, lst)

# COMMAND ----------

display(pd.DataFrame([lst, results, results_all]).T)

# COMMAND ----------

ax1 = criteria_plot(results_all, avg_sales_all, lst, label="ALL")
print(results_all[19])
print(results_all[24])
ax1.axhline(y=results_all[19], color="r", label="axvline - full height")
ax1.axvline(x=0.20, color="r", label="axvline - full height")
ax1.axvline(x=0.60, color="r", label="axvline - full height")
ax1.axvline(x=0.76, color="r", label="axvline - full height")
plt.show()

# COMMAND ----------

'''
-> based --> feedback (1,2,3,4)
-> ccard payment conversion rate -> HDE (0.4) , ALL (0.4)
HDE
0.2 prop visits --> 0.4 conversion rate 
ALL
0.2 prop visits --> 0.41 conversion rate 
current criteria: ALL HDE --> 3.6 M (raw) , ALL Format --> 5.x M (raw), --> raw uplift x cust
new criteria: ALL format, and dont effect to conversion rate

total all conversion (HDE) vs toatal all conversion all (ALL)

''' 
