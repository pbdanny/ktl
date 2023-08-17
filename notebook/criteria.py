# Databricks notebook source
import sys
import os

# sys.path.append(os.path.abspath("/Workspace/Repos/niti.buesamae@lotuss.com/LMP"))


# COMMAND ----------

from utils import files
from utils import etl
from etl import staging

import pandas as pd
import pyspark.sql.functions as F
def gen(dbfs_path, insurer, conf_mapper, fdbk_type):
  fdbk_df, filename = etl.get_df(dbfs_path, insurer, conf_mapper, fdbk_type=fdbk_type)
  cust_fdbck_hh = etl.ngc_to_household(spark, spark.createDataFrame(fdbk_df, schema=None))
  return cust_fdbck_hh

# COMMAND ----------

conf_mapper = files.conf_reader("../config/etl.json")
mnt_mapper = files.conf_reader("../config/mnt.json")
abfss_prefix, dbfs_path = (mnt_mapper["abfss_prefix"], mnt_mapper["dbfs_path"])

# COMMAND ----------

prjct_nm = "test"
test = True
from features import get_txn_cust
txn_cust = get_txn_cust(spark, prjct_nm, test)
txn_cust_r360 = txn_cust.where(F.col("date_id").between(
    F.col("r360_start_date"), F.col("first_contact_date")))

# COMMAND ----------

txn_cust_r360.display()

# COMMAND ----------

model_outbound_path = {
  "lot1": "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/thanakrit/lmp/insurance_lead/edm_202207/outbound_edm202207.parquet",
  "lot2": "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/edm_202210/lead/outbound/all_outbound.parquet",
  "lot3": "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/edm_202301/lead/outbound/all_outbound.parquet"
}

# COMMAND ----------

lot1 = spark.read.parquet(model_outbound_path["lot1"]).select(F.col("Household_ID").alias("household_id"), F.col("InsurerName"), F.col("CashorCreditCard"), F.col("LeadType"), F.col("LeadDate"))
lot2 = spark.read.parquet(model_outbound_path["lot2"]).select(F.col("Household_ID").alias("household_id"), F.col("InsurerName"), F.col("CashorCreditCard"), F.col("LeadType"), F.col("LeadDate"))
lot3 = spark.read.parquet(model_outbound_path["lot3"]).select(F.col("Household_ID").alias("household_id"), F.col("InsurerName"), F.col("CashorCreditCard"), F.col("LeadType"), F.col("LeadDate"))

# COMMAND ----------

print(lot1.count())
print(lot2.count())
print(lot3.count())

# COMMAND ----------

lot1.groupby("InsurerName").count().display()
lot2.groupby("InsurerName").count().display()
lot3.groupby("InsurerName").count().display()

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

PROP_HDE_CCARD_VISIT = 0.2
# % visit by cash or card at HDE format 360 day from contact date >>> split cash customer/ creditcarded customer
agg_cust_hde = \
(txn_cust_r360
 .where(F.col("store_format_group")=="HDE")
 .groupBy("household_id")
 .pivot("tender_type_group")
 .agg(F.sum(F.col("net_spend_amt")).alias("sales_360_days"),
      F.countDistinct(F.col("transaction_uid")).alias("visits_360_days"))
 .fillna(value=0)
)

lower_case_column_names = [c.lower() for c in agg_cust_hde.columns]
agg_cust_hde = agg_cust_hde.toDF(*lower_case_column_names)

# COMMAND ----------

def get_cust(agg_cust_hde, PROP_HDE_CCARD_VISIT = 0.2):
  
  cust_gr_hde = \
  (agg_cust_hde
   .withColumn("total_visits_360_days", (F.col("cash_visits_360_days") + F.col("ccard_visits_360_days")) )
   .withColumn("total_sales_360_days",  (F.col("cash_sales_360_days") + F.col("ccard_sales_360_days")) )
   .withColumn("ratio_visit_by_card_360_days", ((F.col("ccard_visits_360_days") /  F.col("total_visits_360_days"))) )
   .fillna(value=0)
   .withColumn("cash_card_customer_flag", F.when(F.col("ratio_visit_by_card_360_days") >= PROP_HDE_CCARD_VISIT , "credit").otherwise("cash"))
   .select("household_id", "cash_card_customer_flag")
  )

  # Assign "cash" to the rest customer , that no transaction in HDE
  cust_gr = \
  (txn_cust_r360.select("household_id").drop_duplicates()
   .join(cust_gr_hde, "household_id", "outer")
   .fillna("cash", subset=["cash_card_customer_flag"])
  )
  return cust_gr

# COMMAND ----------

cust_gr = get_cust(agg_cust_hde, 0.2)

# COMMAND ----------

# use latest delta instead
feedback1_azay = gen(dbfs_path, "AZAY", conf_mapper, 1)
feedback2_azay = gen(dbfs_path, "AZAY", conf_mapper, 2)
feedback1_cigna = gen(dbfs_path, "CIGNA", conf_mapper, 1)
feedback2_cigna = gen(dbfs_path, "CIGNA", conf_mapper, 2)

# COMMAND ----------

feedback1_cigna.display()

# COMMAND ----------

feedback2_cigna.display()

# COMMAND ----------

feedback2_cigna.groupby("Payment_Type").count().display()

# COMMAND ----------

feedback1_cigna.select("DHB_Model_Date").distinct().display()


# COMMAND ----------

# quick prep for cigna
feedback1_cigna = (
    feedback1_cigna.withColumn(
        "DHB_Model_Date_new", F.regexp_replace(F.col("DHB_Model_Date"), "([a-zA-Z ]|[\u0E00-\u0E7F]+)", "")
    )
    .withColumn("DHB_Model_Date_new",F.col("DHB_Model_Date_new").substr(1,7)).filter(F.col("DHB_Model_Date_new") == "2022/07")
)
feedback1_cigna = feedback1_cigna.drop("2022/07")

# COMMAND ----------

feedback1_cigna.select("DHB_Model_Date").distinct().display()

# COMMAND ----------

# stamp new model date
feedback1_cigna = feedback1_cigna.withColumn("DHB_Model_Date", F.lit("2022/07"))

# COMMAND ----------

feedback1_cigna.groupby("src_file").count().display()

# COMMAND ----------

used_src_file_cig = feedback1_cigna.select("src_file").distinct().collect()
used_src_file_cig = [x.src_file for x in used_src_file_cig]
used_src_file_cig = [x.replace("FDBK1","FDBK2") for x in used_src_file_cig]

# COMMAND ----------

feedback1_cigna.filter(F.col("ngc_customer_id").isNull()).count()

# COMMAND ----------

feedback1_cigna.filter(F.col("household_id").isNull()).count()

# COMMAND ----------

feedback1_cigna.filter(F.col("household_id").isNotNull()).count()

# COMMAND ----------

# cigna feedback 1 --> household id
feedback1_cigna = feedback1_cigna.drop("household_id")

# COMMAND ----------

feedback2_cigna = feedback2_cigna.filter(F.col("src_file").isin(used_src_file_cig))

# COMMAND ----------

feedback2_cigna.groupby("Payment_Type").count().display()

# COMMAND ----------

feedback2_cigna.groupby("Policy_type_self").count().display()

# COMMAND ----------

feedback2_cigna.groupby("Premium_Frequency").count().display()

# COMMAND ----------

feedback2_cigna.groupby("In_Force_Policy_Description").count().display()

# COMMAND ----------

feedback2_cigna.groupby("Policy_type_self").count().display()

# COMMAND ----------

print(feedback1_cigna.count())
print(feedback2_cigna.count())

# COMMAND ----------

feedback2_cigna.display()

# COMMAND ----------

# join cigna feedback 1 and feedback 2 together
feedback_cigna = feedback1_cigna.select(
    "ngc_customer_id", "lead_type", "Bought_Status"
).join(feedback2_cigna, "ngc_customer_id", "left")
feedback_cigna.count()

# COMMAND ----------

feedback_cigna.groupby("Payment_Type").count().display()

# COMMAND ----------

feedback_cigna.groupby("Payment_Type").count().display()

# COMMAND ----------

feedback_cigna = feedback_cigna.drop_duplicates()

# COMMAND ----------

feedback_cigna.groupby("ngc_customer_id").count().sort(F.desc("count")).display()

# COMMAND ----------

feedback_cigna.filter(F.col("ngc_customer_id")=="102111060004977486").display()

# COMMAND ----------

feedback_cigna.filter(F.col("ngc_customer_id")=="102111060008403503").display()

# COMMAND ----------

lot1 = spark.read.parquet(model_outbound_path["lot1"]).select(
    F.col("Household_ID").alias("household_id"),
    F.col("InsurerName"),
    F.col("CashorCreditCard"),
    F.col("LeadType"),
    F.col("LeadDate"),
    F.col("ModelScoreBucket"),
)

# COMMAND ----------

lot1.display()

# COMMAND ----------

lot1.groupby("InsurerName").count().display()

# COMMAND ----------

lot1.filter(F.col("household_id")=="102111060008403503").display()

# COMMAND ----------

# concat lot for azay
feedback1_azay = feedback1_azay.withColumn("Filename", F.split("src_file",".txt")[0])
feedback1_azay = feedback1_azay.withColumn("lot", F.split("Filename","_")[4])
feedback2_azay = feedback2_azay.withColumn("Filename", F.split("src_file",".txt")[0])
feedback2_azay = feedback2_azay.withColumn("lot", F.split("Filename","_")[4])

# COMMAND ----------

feedback1_azay = feedback1_azay.filter(F.col("lot").isin(["202209","202210","202211"]))
feedback2_azay = feedback2_azay.filter(F.col("lot").isin(["202209","202210","202211"]))

# COMMAND ----------

feedback1_azay.display()

# COMMAND ----------

feedback2_azay.display()

# COMMAND ----------

feedback2_azay.groupby("Payment_Type").count().display()

# COMMAND ----------

# join cigna feedback 1 and feedback 2 together
feedback_azay = feedback1_azay.select(
    "ngc_customer_id", "lead_type", "Bought_Status"
).join(feedback2_azay, "ngc_customer_id", "left")
feedback_azay.filter(F.col("Payment_Type").isNull()).display()

# COMMAND ----------

feedback_azay.groupby("Payment_Type").count().display()

# COMMAND ----------

lot1_azay = lot1.filter(F.col("InsurerName")=="AAC")

# COMMAND ----------

lot1_azay.count()

# COMMAND ----------

feedback_azay.display()

# COMMAND ----------

agg_cust_hde.count()

# COMMAND ----------

agg_cust_hde.drop_duplicates().count()

# COMMAND ----------

cust_gr_azay = get_cust(agg_cust_hde, 0.2)

# COMMAND ----------

cust_gr_azay.display()

# COMMAND ----------

cust_gr_azay = get_cust(agg_cust_hde, 0.2)
cust_gr_azay = cust_gr_azay.join(lot1_azay.select("household_id"), "household_id", "inner")

# COMMAND ----------

print(cust_gr_azay.count())
print(lot1_azay.count())

# COMMAND ----------

azay_join = feedback_azay.join(cust_gr_azay, feedback_azay.ngc_customer_id == cust_gr_azay.household_id, "inner")

# COMMAND ----------

azay_join.count()

# COMMAND ----------

azay_join.display()

# COMMAND ----------

azay_join.groupby("Payment_Type").count().display()

# COMMAND ----------

azay_join.groupby("cash_card_customer_flag").count().display()

# COMMAND ----------

azay_join_filtered = azay_join.filter(F.col("Payment_Type").isNotNull())

# COMMAND ----------

azay_join_filtered.groupby("Payment_Type").count().display()
azay_join_filtered.groupby("cash_card_customer_flag").count().display()
azay_join_filtered.groupby("Payment_Type", "cash_card_customer_flag").count().display()

# COMMAND ----------

tot = azay_join_filtered.count()

# COMMAND ----------

azay_join_filtered.groupBy("Payment_Type") \
  .count() \
  .withColumnRenamed('count', 'cnt_per_group') \
  .withColumn('perc_of_count_total', (F.col('cnt_per_group') / tot) ) \
  .display()

azay_join_filtered.groupBy("cash_card_customer_flag") \
  .count() \
  .withColumnRenamed('count', 'cnt_per_group') \
  .withColumn('perc_of_count_total', (F.col('cnt_per_group') / tot) ) \
  .display()

# COMMAND ----------

actual = (
    azay_join_filtered.groupBy("Payment_Type")
    .count()
    .withColumnRenamed("count", "cnt_per_group")
    .withColumn("perc_of_count_total", (F.col("cnt_per_group") / tot))
    .filter(F.col("Payment_Type") == "CREDIT CARD")
    .collect()[0].perc_of_count_total
)

# COMMAND ----------

actual

# COMMAND ----------

criteria

# COMMAND ----------

azay_join.count()

# COMMAND ----------

cust_join.display()

# COMMAND ----------

cust_join = feedback_azay.join(agg_cust_hde, feedback_azay.ngc_customer_id == agg_cust_hde.household_id, "inner")

# COMMAND ----------

cust_join.count()

# COMMAND ----------

def get_cust_prop(agg_cust_hde, PROP_HDE_CCARD_VISIT = 0.2):
  
  cust_gr_hde = \
  (agg_cust_hde
   .withColumn("total_visits_360_days", (F.col("cash_visits_360_days") + F.col("ccard_visits_360_days")) )
   .withColumn("total_sales_360_days",  (F.col("cash_sales_360_days") + F.col("ccard_sales_360_days")) )
   .withColumn("ratio_visit_by_card_360_days", ((F.col("ccard_visits_360_days") /  F.col("total_visits_360_days"))) )
   .fillna(value=0)
#    .withColumn("cash_card_customer_flag", F.when(F.col("ratio_visit_by_card_360_days") >= PROP_HDE_CCARD_VISIT , "credit").otherwise("cash"))
   .select("household_id","ratio_visit_by_card_360_days")
  )

  # Assign "cash" to the rest customer , that no transaction in HDE
  cust_gr = \
  (txn_cust_r360.select("household_id").drop_duplicates()
   .join(cust_gr_hde, "household_id", "outer")
   .fillna(0, subset=["ratio_visit_by_card_360_days"])
  )
  return cust_gr

# COMMAND ----------

criteria_azay = get_cust_prop(agg_cust_hde, PROP_HDE_CCARD_VISIT=0.2)

# COMMAND ----------

criteria_azay.count()

# COMMAND ----------

cust_join = azay_join_filtered.join(criteria_azay, azay_join_filtered.ngc_customer_id == criteria_azay.household_id, "inner")

# COMMAND ----------

cust_join.count()

# COMMAND ----------

import numpy as np
from tqdm.notebook import tqdm

# COMMAND ----------

lst = np.arange(0.01,1.0, 0.01) 

# COMMAND ----------

results = []
for i in lst:
  print(i)
  cust_join_tmp = cust_join.withColumn("cash_card_customer_flag", F.when(F.col("ratio_visit_by_card_360_days") >= i , "credit").otherwise("cash"))
  cust_join_tmp = cust_join_tmp.select("ngc_customer_id", "cash_card_customer_flag")
  criteria = (
    cust_join_tmp.groupBy("cash_card_customer_flag")
    .count()
    .withColumnRenamed("count", "cnt_per_group")
    .withColumn("perc_of_count_total", (F.col("cnt_per_group") / tot))
    .filter(F.col("cash_card_customer_flag") == "credit")
    .collect()[0].perc_of_count_total
   )
  results.append(criteria)

# COMMAND ----------



# COMMAND ----------

cust_join_df = cust_join.toPandas()

# COMMAND ----------

tot

# COMMAND ----------

azay_join_filtered_df = azay_join_filtered.toPandas()

# COMMAND ----------

results = []
for i in tqdm(lst):
  cust_join_df_tmp = cust_join_df.copy()
  cust_join_df_tmp["cash_card_customer_flag"] = cust_join_df_tmp["ratio_visit_by_card_360_days"].apply(lambda x: "credit" if x >= i else "cash")
  cust_join_df_tmp = cust_join_df_tmp[["ngc_customer_id", "cash_card_customer_flag"]]
  cust_join_df_tmp = cust_join_df_tmp.groupby("cash_card_customer_flag").agg("count").reset_index()
  val = cust_join_df_tmp[cust_join_df_tmp["cash_card_customer_flag"]=="credit"]["ngc_customer_id"].to_list()[0]/tot
  results.append(val)

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

actuals = [actual]*len(results)

# COMMAND ----------

lst = np.arange(0.01,1.0, 0.01)

# COMMAND ----------

lst = [float("{:.2f}".format(x)) for x in lst]

# COMMAND ----------



# COMMAND ----------

plt.rcParams["figure.figsize"] = [30, 10]
plt.rcParams["figure.autolayout"] = True
plt.plot(results)
plt.plot(actuals)
plt.xticks(range(len(results)), lst, rotation=90)
plt.xlabel('PROP_HDE_CCARD_VISIT')
plt.ylabel('ylabel')
plt.show()

# COMMAND ----------

agg_cust_all_format = \
(txn_cust_r360
 .groupBy("household_id")
 .pivot("tender_type_group")
 .agg(F.sum(F.col("net_spend_amt")).alias("sales_360_days"),
      F.countDistinct(F.col("transaction_uid")).alias("visits_360_days"))
 .fillna(value=0)
)

lower_case_column_names = [c.lower() for c in agg_cust_all_format.columns]
agg_cust_all_format = agg_cust_all_format.toDF(*lower_case_column_names)

# COMMAND ----------

cust_gr_azay_all = get_cust(agg_cust_all_format, 0.2)
cust_gr_azay_all = cust_gr_azay_all.join(lot1_azay.select("household_id"), "household_id", "inner")
azay_join_all = feedback_azay.join(cust_gr_azay_all, feedback_azay.ngc_customer_id == cust_gr_azay_all.household_id, "inner")
azay_join_filtered_all = azay_join_all.filter(F.col("Payment_Type").isNotNull())
criteria_azay_all = get_cust_prop(agg_cust_all_format, PROP_HDE_CCARD_VISIT=0.2)
cust_join_all = azay_join_filtered_all.join(criteria_azay_all, azay_join_filtered_all.ngc_customer_id == criteria_azay_all.household_id, "inner")
cust_join_all_df = cust_join_all.toPandas()

# COMMAND ----------

cust_join_df_tmp = cust_join_all_df[["Payment_Type","ngc_customer_id"]].groupby("Payment_Type").agg("count").reset_index()
actual = cust_join_df_tmp[cust_join_df_tmp["Payment_Type"]=="CREDIT CARD"]["ngc_customer_id"].to_list()[0]/tot

# COMMAND ----------

actual

# COMMAND ----------

results = []
for i in tqdm(lst):
  cust_join_df_tmp = cust_join_all_df.copy()
  cust_join_df_tmp["cash_card_customer_flag"] = cust_join_df_tmp["ratio_visit_by_card_360_days"].apply(lambda x: "credit" if x >= i else "cash")
  cust_join_df_tmp = cust_join_df_tmp[["ngc_customer_id", "cash_card_customer_flag"]]
  cust_join_df_tmp = cust_join_df_tmp.groupby("cash_card_customer_flag").agg("count").reset_index()
  val = cust_join_df_tmp[cust_join_df_tmp["cash_card_customer_flag"]=="credit"]["ngc_customer_id"].to_list()[0]/tot
  results.append(val)

# COMMAND ----------

actuals = [actual]*len(results)
plt.rcParams["figure.figsize"] = [30, 10]
plt.rcParams["figure.autolayout"] = True
plt.plot(results)
plt.plot(actuals)
plt.xticks(range(len(results)), lst, rotation=90)
plt.xlabel('card flag propotion [ALL]')
plt.ylabel('feedback payment propotaion')
plt.show()

# COMMAND ----------


