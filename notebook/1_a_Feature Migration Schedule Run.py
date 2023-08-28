# Databricks notebook source
# MAGIC %md ##Original Source Code
# MAGIC /Users/kritawats.kraiwitchaicharoen@lotuss.com/Project/(Clone) KTL

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialization + Load config
# MAGIC

# COMMAND ----------

import sys
import os
from pathlib import Path

import time
import calendar
from datetime import *
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from functools import reduce

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("lmp").getOrCreate()

# COMMAND ----------

# DBTITLE 1,Load ETL config
from utils import files

conf_path = "../config/snap_txn.json"

conf_mapper = files.conf_reader("../config/snap_txn.json")
decision_date = conf_mapper["decision_date"]
print(f"{decision_date}")

# COMMAND ----------

# DBTITLE 1,Decision date -> define start , end timeframe for txn -> conf_mapper
# decision date = latest data end date
decision_date =  datetime.strptime(conf_mapper["decision_date"], '%Y-%m-%d').date()

#--- For Code testing 
# decision_date =  datetime.strptime("2023-08-13", '%Y-%m-%d').date() + timedelta(days=365)
#----

# timeframe_end = 1 month back from decision date
# timeframe_start = 1 year from timeframe_end
timeframe_end = date(decision_date.year, decision_date.month - 1, 1) - timedelta(days=1)
timeframe_start = (timeframe_end - relativedelta(months=11)).replace(day=1)

print(f"decision date : {decision_date}\ntxn start date : {timeframe_start}\ntxn end date : {timeframe_end}")
print(f"gap days from decision - txn end : {(decision_date - timeframe_end).days}")
print(f"gap days from txn start - txn end : {(timeframe_end - timeframe_start).days}")

# Get week_id of time frame
date_dim = spark.table('tdm.v_date_dim').select('week_id', 'date_id', 'period_id', 'quarter_id', 'promoweek_id')

start_week = date_dim.filter(F.col('date_id').between(timeframe_start, timeframe_end)).agg(F.min('week_id')).collect()[0][0]
end_week = date_dim.filter(F.col('date_id').between(timeframe_start, timeframe_end)).agg(F.max('week_id')).collect()[0][0]

print(f"start_week : {start_week}, end_week : {end_week}")

conf_mapper["timeframe_end"] = timeframe_end.strftime("%Y-%m-%d")
conf_mapper["timeframe_start"] = timeframe_start.strftime("%Y-%m-%d")
conf_mapper["start_week"] = start_week
conf_mapper["end_week"] = end_week

files.conf_writer(conf_mapper, conf_path)

# COMMAND ----------

# DBTITLE 1,Skip, use 118wk Get Raw Data
product_df = (
    spark.table("tdm.v_prod_dim_c")
    .select(
        [
            "upc_id",
            "brand_name",
            "division_id",
            "division_name",
            "department_id",
            "department_name",
            "department_code",
            "section_id",
            "section_name",
            "section_code",
            "class_id",
            "class_name",
            "class_code",
            "subclass_id",
            "subclass_name",
            "subclass_code",
        ]
    )
    .filter(F.col("division_id").isin(product_division))
    .filter(F.col("country") == country)
)

header_df = (
    spark.table("tdm.v_transaction_head")
    .select(["transaction_uid", "store_id", "date_id", "channel"])
    .filter(F.col("week_id").between(start_week, end_week))
    .filter(F.col("date_id").between(timeframe_start, timeframe_end))
    .filter(F.col("country") == country)
)

item_df = (
    spark.table("tdm.v_transaction_item")
    .select(
        [
            "transaction_uid",
            "store_id",
            "date_id",
            "week_id",
            "tran_datetime",
            "upc_id",
            "customer_id",
            "net_spend_amt",
            "cc_flag",
            "count_qty",
            "product_qty",
            "measured_qty",
            "discount_amt",
            "source",
        ]
    )
    .filter(F.col("week_id").between(start_week, end_week))
    .filter(F.col("date_id").between(timeframe_start, timeframe_end))
    .filter(F.col("country") == country)
    .where(
        (F.col("net_spend_amt") > 0)
        & (F.col("product_qty") > 0)
        & (F.col("date_id").isNotNull())
    )
    .filter(F.col("cc_flag") == "cc")
)

date_df = (
    spark.table("tdm.v_date_dim")
    .select(
        [
            "date_id",
            "period_id",
            "quarter_id",
            "year_id",
            "month_id",
            "weekday_nbr",
            "day_in_month_nbr",
            "day_in_year_nbr",
            "day_num_sequence",
            "week_num_sequence",
            "promoweek_id",
            "dp_data_dt",
        ]
    )
    .filter(F.col("week_id").between(start_week, end_week))
    .filter(F.col("date_id").between(timeframe_start, timeframe_end))
    .dropDuplicates()
)

# COMMAND ----------

# DBTITLE 1,Skip, use 118wk Get Raw data
customer_df = (
    spark.table("tdm.v_customer_dim")
    .select(
        [
            "customer_id",
            "household_id",
            "card_issue_date",
            "golden_record_external_id_hash",
        ]
    )
    .filter(F.col("country") == country)
)

# COMMAND ----------

# DBTITLE 1,Skip, use 118wk Get Raw Data
store_df = (
    spark.table("tdm.v_store_dim_c")
    .select("store_id", "format_id", "region")
    .filter(F.col("format_id").isin(store_format))
    .filter(F.col("country") == country)
    .withColumn(
        "format_name",
        F.when(F.col("format_id").isin([1, 2, 3]), "Hypermarket")
        .when(F.col("format_id") == 4, "Supermarket")
        .when(F.col("format_id") == 5, "Mini Supermarket")
        .otherwise(F.col("format_id")),
    )
)

data_df = (
    item_df.join(header_df, on=["transaction_uid", "store_id", "date_id"], how="inner")
    .join(product_df, on="upc_id", how="inner")
    .join(store_df, on="store_id", how="inner")
    .join(date_df, on="date_id", how="left")
    .join(customer_df, on="customer_id", how="left")
    .withColumn(
        "unique_transaction_uid",
        F.concat_ws("_", F.col("transaction_uid"), F.col("store_id"), F.col("date_id")),
    )
    .withColumn(
        "unit",
        F.when(F.col("count_qty").isNotNull(), F.col("product_qty")).otherwise(
            F.col("measured_qty")
        ),
    )
)

data_df = data_df.withColumn(
    "channel_group_forlotuss",
    F.when(F.col("channel") == "OFFLINE", "OFFLINE")
    .when(F.col("channel").isin("Click and Collect", "Scheduled CC"), "Click & Collect")
    .when(F.col("channel").isin("HATO"), "Line HATO")
    .when(
        F.col("channel").isin("GHS 1", "GHS 2", "GHS APP", "Scheduled HD"),
        "Scheduled HD",
    )
    .when(F.col("channel").isin("HLE", "Scheduled HLE"), "Electronic Mall")
    .when(
        (F.col("channel").isin("OnDemand HD"))
        & (F.col("format_name").isin("Hypermarket"))
        & (F.col("store_id") != 5185),
        "OnDemand Hypermarket",
    )
    .when(
        (F.col("channel").isin("OnDemand HD"))
        & (F.col("format_name").isin("Supermarket", "Mini Super")),
        "OnDemand",
    )
    .when(F.col("channel").isin("Light Delivery"), "OnDemand")
    .when(
        (F.col("channel").isin("OnDemand HD"))
        & (F.col("store_id") == 5185)
        & (F.col("date_id") <= "2023-04-06"),
        "OnDemand",
    )
    .when(
        (F.col("channel").isin("OnDemand HD"))
        & (F.col("store_id") == 5185)
        & (F.col("date_id") >= "2023-04-07"),
        "OnDemand Hypermarket",
    )
    .when(
        F.col("channel").isin("Shopee", "Lazada", "O2O Lazada", "O2O Shopee"),
        "Marketplace",
    )
    .when(
        F.col("channel").isin(
            "Ant Delivery ",
            "Food Panda",
            "Grabmart",
            "Happy Fresh",
            "Robinhood",
            "We Fresh",
            "7 MARKET",
        ),
        "Aggregator",
    )
    .when(F.col("channel").isin("TRUE SMART QR"), "Others")
    .otherwise(F.lit("OFFLINE")),
)

data_df = data_df.withColumn(
    "format_name",
    F.when(
        F.col("channel_group_forlotuss").isin(
            "Scheduled HD",
            "Electronic Mall",
            "OnDemand",
            "OnDemand Hypermarket",
            "Click & Collect",
            "Line HATO",
            "Marketplace",
            "Aggregator",
            "Others",
        ),
        F.lit("ONLINE"),
    ).otherwise(F.col("format_name")),
)

data_df = data_df.withColumn(
    "channel_flag",
    F.when(
        F.col("format_name").isin("Hypermarket", "Supermarket", "Mini Super"),
        F.lit("OFFLINE"),
    ).otherwise(F.lit("ONLINE")),
)


# data_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_full_data_tmp")
data_df.write.mode("overwrite").saveAsTable(
    "tdm_seg.th_lotuss_ktl_txn_year_full_data_tmp"
)

# COMMAND ----------

# DBTITLE 1,Skip, use 118wk Load Back Data
# data full year
# data_df = spark.table("tdm_seg.kritawatkrai_th_year_full_data_tmp")

# data_df.display()

# data_df = spark.table("tdm_seg.th_lotuss_ktl_txn_year_full_data_tmp")

# COMMAND ----------

# DBTITLE 1,Get Txn data from 118wk , remove trader
conf_mapper = files.conf_reader(conf_path)

txn_cc = (spark.table("tdm_seg.v_latest_txn118wk")
           .where(F.col("week_id").between(start_week, end_week))
           .where(F.col("date_id").between(conf_mapper["timeframe_start"], conf_mapper["timeframe_end"]))
           .where(F.col("cc_flag").isin(["cc"]))
           .withColumn("store_region", F.when(F.col("store_region").isNull(), "Unidentified").otherwise(F.col("store_region")))
)

# Remove trader 
trader_df = spark.table('tdm_seg.trader2023_subseg_master')
max_quarter_id = 0
max_quarter_id_trader = trader_df.agg(F.max('quarter_id')).collect()[0][0]
max_txn_date_id = txn_cc.agg(F.max('date_id')).collect()[0][0]
max_quarter_id_txn = spark.table("tdm.v_date_dim").where(F.col("date_id")==timeframe_end).select("quarter_id").collect()[0][0]

if (max_quarter_id_trader >= max_quarter_id_txn):
    max_quarter_id = max_quarter_id_txn
else:
    max_quarter_id = max_quarter_id_trader

trader_df = spark.table('tdm_seg.trader2023_subseg_master').filter(F.col('quarter_id') == max_quarter_id)

data_df = txn_cc.join(trader_df, on='household_id', how='leftanti')

# COMMAND ----------

from utils import files

conf_path = "../config/snap_txn.json"

from features import snap_txn

txn_cc = snap_txn.get_txn_cc_exc_trdr(spark, conf_path)
txn_time = snap_txn.map_txn_time(spark, conf_path, txn_cc)

# COMMAND ----------

files.conf_reader(conf_path)

# COMMAND ----------

# DBTITLE 1,Flag txn - time : time of day, week of month, week end flag
# time of day / week of month flag 
    
date_dim = (spark
            .table('tdm.v_date_dim')
            .select(['date_id','period_id','quarter_id','year_id','month_id','weekday_nbr','week_id',
                     'day_in_month_nbr','day_in_year_nbr','day_num_sequence','week_num_sequence'])
            .where(F.col("week_id").between(start_week, end_week))
            .where(F.col("date_id").between(timeframe_start, timeframe_end))
                     .dropDuplicates()
            )

time_of_day_df = (data_df
                  .join(date_dim.drop("week_id"), "date_id", "inner")
                  .withColumn('decision_date', F.lit(decision_date))
                  .withColumn('tran_hour', F.hour(F.col('tran_datetime')))
)

time_of_day_df = (time_of_day_df
                  .withColumn('time_of_day', F.when((F.col('tran_hour') >= 5) & (F.col('tran_hour') <= 8), 'prework')
                                              .when((F.col('tran_hour') >= 9) & (F.col('tran_hour') <= 11), 'morning')
                                              .when(F.col('tran_hour') == 12, 'lunch')
                                              .when((F.col('tran_hour') >= 13) & (F.col('tran_hour') <= 17), 'afternoon')
                                              .when((F.col('tran_hour') >= 18) & (F.col('tran_hour') <= 20), 'evening')
                                              .when(F.col('tran_hour') >= 21, 'late')
                                              .when(F.col('tran_hour') <= 4, 'night')
                                              .otherwise('def'))
                  .withColumn('week_of_month', 
                              F.when(F.col('day_in_month_nbr') <= 7, 1)
                               .when((F.col('day_in_month_nbr') > 7) & (F.col('day_in_month_nbr') <= 14), 2)
                               .when((F.col('day_in_month_nbr') > 14) & (F.col('day_in_month_nbr') <= 21), 3)
                               .when(F.col('day_in_month_nbr') > 21, 4))
                  .withColumn('weekend_flag', 
                              F.when(F.col('weekday_nbr').isin(6,7), F.lit('Y'))
                              .when((F.col('weekday_nbr') == 5) & (F.col('time_of_day').isin('evening', 'late')), 'Y')
                              .otherwise('N'))
)
                        

# COMMAND ----------

# DBTITLE 1,Flag txn-time : festive week (xmax , songkran)
# festival flag (+- 1 from last week in december) = xmas
# month_id ends with 4 = april

date_dim.agg(F.min("week_id"), F.max("week_id")).display()

max_week_december = (date_dim
                     .filter((F.col("month_id") % 100) == 12)
                     .filter(F.col("week_id").startswith(F.col("month_id").substr(1, 4))) 
                     .agg(F.max(F.col("week_id")).alias("max_week_december")).collect()[0]["max_week_december"]
)

print(max_week_december)
d = date_dim.select('week_id').distinct()

df_with_lag_lead = d.withColumn("lag_week_id", F.lag("week_id").over(Window.orderBy("week_id"))) \
                    .withColumn("lead_week_id", F.lead("week_id").over(Window.orderBy("week_id")))

week_before = df_with_lag_lead.filter(F.col("week_id") == max_week_december).select("lag_week_id").first()[0]
week_after = df_with_lag_lead.filter(F.col("week_id") == max_week_december).select("lead_week_id").first()[0]

xmas_week_id = [week_before, max_week_december, week_after]

print(xmas_week_id)
time_of_day_df = time_of_day_df.withColumn('fest_flag',F.when(F.col('week_id').isin(xmas_week_id), 'XMAS')\
                                                       .when(F.col('month_id').cast('string').endswith('04'), 'APRIL')\
                                                       .otherwise('NONE'))

# print('Initial Count: ', time_of_day_df.count())

# COMMAND ----------

# DBTITLE 1,Flag txn-time : Last week of month
# weekday_nbr -> monday = 1, sunday = 7
# date_df = spark.table('tdm.v_date_dim').select(['date_id','period_id','quarter_id','year_id','month_id','weekday_nbr','day_in_month_nbr','day_in_year_nbr','day_num_sequence','week_num_sequence','promoweek_id','dp_data_dt'])\
#                                             .filter(F.col('week_id').between(start_week, end_week))\
#                                             .dropDuplicates()

date_dim

last_sat = date_dim.filter(F.col('weekday_nbr') == 6).groupBy('month_id').agg(F.max('day_in_month_nbr').alias('day_in_month_nbr'))\
                                                  .withColumn('last_weekend_flag',F.lit('Y'))

last_sat_df = date_dim.select('date_id', 'month_id', 'day_in_month_nbr')\
                     .join(last_sat, on=['month_id','day_in_month_nbr'],how='inner')

last_weekend_df = last_sat_df.select(F.col('month_id'),F.col('day_in_month_nbr'),F.col('date_id'),F.col('last_weekend_flag')) \
                 .unionAll(last_sat_df.select(F.col('month_id'),F.col('day_in_month_nbr'), F.date_add(F.col('date_id'), 1).alias('date_id'),F.col('last_weekend_flag'))) \
                 .unionAll(last_sat_df.select(F.col('month_id'),F.col('day_in_month_nbr'), F.date_sub(F.col('date_id'), 1).alias('date_id'),F.col('last_weekend_flag')))

last_weekend_df = last_weekend_df.select('date_id', 'last_weekend_flag')

flagged_df = (time_of_day_df
              .join(last_weekend_df, on='date_id',how='left')
              .fillna('N', subset=['last_weekend_flag'])
)

# COMMAND ----------

flagged_df.select("date_id", "tran_datetime", "weekday_nbr", "day_in_month_nbr", "day_in_year_nbr",
                  "tran_hour", "time_of_day", "week_of_month", "weekend_flag", "fest_flag", "last_weekend_flag").display(10)

# COMMAND ----------

flagged_df.where(F.col("last_weekend_flag").isin(["Y"])).select(F.dayofmonth("date_id").alias("day")).drop_duplicates().display()

# COMMAND ----------

# DBTITLE 1,Flag txn-time : Recency
# Recency and Tenure Flag

r = flagged_df.withColumn('end_date',F.lit(timeframe_end))\
          .withColumn('start_date',F.lit(timeframe_start))\
          .withColumn('start_month_date', F.trunc(F.col('date_id'), 'month'))\
          .withColumn('end_month_date', F.last_day(F.col('start_month_date')))\
          .withColumn('months_from_end_date', F.months_between(F.col('end_date'),F.col('end_month_date')) + 1)\
          .withColumn('last_3_flag',F.when(F.col('months_from_end_date') <= 3 , 'Y')\
                                    .otherwise('N'))\
          .withColumn('last_6_flag',F.when(F.col('months_from_end_date') <= 6 , 'Y')\
                                              .otherwise('N'))\
          .withColumn('last_9_flag',F.when(F.col('months_from_end_date') <= 9 , 'Y')\
                                    .otherwise('N'))\
          .withColumn('q1_flag',F.when(F.col('months_from_end_date') <= 3 , 'Y')\
                                              .otherwise('N'))\
          .withColumn('q2_flag',F.when((F.col('months_from_end_date') > 3) & (F.col('months_from_end_date') <= 6) , 'Y')\
                                    .otherwise('N'))\
          .withColumn('q3_flag',F.when((F.col('months_from_end_date') > 6) & (F.col('months_from_end_date') <= 9) , 'Y')\
                                              .otherwise('N'))\
          .withColumn('q4_flag',F.when(F.col('months_from_end_date') > 9 , 'Y')\
                                              .otherwise('N'))\
          .withColumn('app_year_qtr',F.when(F.col('q1_flag') == 'Y', 'Q1')\
                                     .when(F.col('q2_flag') == 'Y', 'Q2')\
                                     .when(F.col('q3_flag') == 'Y', 'Q3')\
                                     .when(F.col('q4_flag') == 'Y', 'Q4')\
                                     .otherwise('NA'))
          
# print('Recency Count: ', r.count())

# COMMAND ----------

# DBTITLE 1,Flag txn-product : premium & budget
# Premium / Budget Flag
product_df = (spark.table('tdm.v_prod_dim_c')
              .select(['upc_id','brand_name','division_id','division_name','department_id','department_name','department_code','section_id','section_name','section_code','class_id','class_name','class_code','subclass_id','subclass_name','subclass_code'])
              .filter(F.col('division_id').isin([1,2,3,4,9,10,13]))
              .filter(F.col('country').isin("th"))
)

temp_prod_df = product_df.select('upc_id', 'subclass_code', 'subclass_name')

premium_prod_df = (temp_prod_df
                   .filter(F.col('subclass_name').ilike('%PREMIUM%'))
                   .filter(~F.col('subclass_name').ilike('%COUPON%'))
                   .withColumn('price_level',F.lit('PREMIUM'))
                   ).distinct()

budget_prod_df = (temp_prod_df
                  .filter(F.col('subclass_name').rlike('(?i)(budget|basic|value)'))
                  .withColumn('price_level',F.lit('BUDGET'))
                  ).distinct()

price_level_df = premium_prod_df.unionByName(budget_prod_df)

more_flagged_df = r.join(price_level_df.select('upc_id','price_level'), on='upc_id', how='left')\
                   .fillna('NONE', subset=['price_level'])\
                   .dropna(subset=['household_id'])

# print('Price Level Count: ', more_flagged_df.count())

# COMMAND ----------

# DBTITLE 1,Flag txn-tender : single tender
# Payment Method Flag

# RESA tender
resa_tender = spark.table("tdm.v_resa_group_resa_tran_tender")
resa_tender = (
    resa_tender.withColumn("tender_type_group", F.trim(F.col("tender_type_group")))
    .withColumn(
        "set_tndr_type",
        F.array_distinct(
            F.collect_list(F.col("tender_type_group")).over(
                Window.partitionBy(["tran_seq_no", "store", "day"])
            )
        ),
    )
    # .withColumn("set_tndr_type", F.collect_set(F.col("tender_type_group")).over(Window.partitionBy("tran_seq_no")))
    .withColumn("n_tndr_type", F.size(F.col("set_tndr_type")))
    .select(
        "tran_seq_no", "store", "day", "dp_data_dt", "n_tndr_type", "tender_type_group"
    )
    .withColumn(
        "sngl_tndr_type",
       F.when(F.col("n_tndr_type") == 1,F.col("tender_type_group")).otherwise(
           F.lit("MULTI")
        ),
    )
    # Adjust to support new unique txn_uid from surrogate key
    .withColumnRenamed("tran_seq_no", "transaction_uid_orig")
    .withColumnRenamed("store", "store_id")
    .withColumnRenamed("dp_data_dt", "date_id")
    .select("transaction_uid_orig", "store_id", "day", "date_id", "sngl_tndr_type")
    .drop_duplicates()
)

# OSM (Online) Tender
oms_tender = spark.table("tdm_seg.v_oms_group_payment").filter(F.col("Country") == "th")

oms_tender = (
    oms_tender.withColumn("PaymentMethod", F.trim(F.col("PaymentMethod")))
    .withColumn(
        "set_tndr_type",
        F.array_distinct(
            F.collect_list(F.col("PaymentMethod")).over(
                Window.partitionBy(["transaction_uid"])
            )
        ),
    )
    # .withColumn("set_tndr_type", F.collect_set(F.col("tender_type_group")).over(Window.partitionBy("tran_seq_no")))
    .withColumn("n_tndr_type", F.size(F.col("set_tndr_type")))
    .select(
        "transaction_uid", "dp_data_dt", "n_tndr_type", "PaymentMethod"
    )
    .withColumn(
        "sngl_tndr_type",
       F.when(F.col("n_tndr_type") == 1,F.col("PaymentMethod")).otherwise(
           F.lit("MULTI")
        ),
    )
    # Adjust to support new unique txn_uid from surrogate key
    .withColumnRenamed("transaction_uid", "transaction_uid_orig")
    .select("transaction_uid_orig", "sngl_tndr_type")
    .drop_duplicates()
)

resa_tender = resa_tender.withColumn("oms",F.lit(False))
oms_tender = oms_tender.withColumn("oms",F.lit(True))

filter_resa_tender = resa_tender.filter(F.col('date_id').between(timeframe_start, timeframe_end))

filter_resa_tender = filter_resa_tender.withColumnRenamed('transaction_uid_orig', 'transaction_uid')\
                                       .withColumnRenamed('sngl_tndr_type', 'resa_payment_method')\
                                       .dropDuplicates()

filter_oms_tender = oms_tender.withColumnRenamed('sngl_tndr_type', 'oms_payment_method')\
                              .withColumnRenamed('transaction_uid_orig', 'transaction_uid')\
                              .dropDuplicates()

# Add transaction type to txn
flag_df = more_flagged_df.join(filter_resa_tender.select('transaction_uid','store_id','date_id','resa_payment_method'), 
                                    on=['transaction_uid', 'store_id', 'date_id'], how='left')\
                              .join(filter_oms_tender.select('transaction_uid','oms_payment_method'), on='transaction_uid', how='left')

flag_df = (flag_df
           .withColumn('resa_payment_method',
                       F.when(F.col('resa_payment_method').isNull(), F.lit('Unidentified'))
                        .otherwise(F.col('resa_payment_method')))
           .withColumn('payment_flag',
                       F.when((F.col('resa_payment_method') == 'CASH') | (F.col('oms_payment_method') == 'CASH'), 'CASH')
                        .when((F.col('resa_payment_method') == 'CCARD') | (F.col('oms_payment_method') == 'CreditCard'), 'CARD')
                            .when((F.col('resa_payment_method') == 'COUPON'), 'COUPON')
                            .when((F.col('resa_payment_method') == 'VOUCH'), 'VOUCHER')
                            .otherwise('OTHER'))
)

# COMMAND ----------

# DBTITLE 1,Customer dim : card issue , first txn
#--- get card_issue_date for nulls (use first transaction instead)

max_card_issue = (spark
                  .table('tdm.v_customer_dim')
                  .groupBy("household_id", "golden_record_external_id_hash")
                  .agg(F.max("card_issue_date").alias("card_issue_date"))
                  .drop_duplicates()
)

first_tran = (spark.table('tdm_seg.mylotuss_customer_1st_txn_V1')
              .select('golden_record_external_id_hash', 'tran_datetime')
              .withColumnRenamed('tran_datetime', 'first_tran_datetime')
              )
               
flag_df = (
    flag_df
    .join(max_card_issue, on='household_id', how='left')
    .join(first_tran, on='golden_record_external_id_hash', how='left')
    .withColumn('card_issue_date',
                F.when(F.col('card_issue_date').isNull(),
                       F.col('first_tran_datetime')).otherwise(F.col('card_issue_date')))
    .withColumn('CC_TENURE', F.round((F.datediff(F.col('end_date'),F.col('card_issue_date'))) / 365,1))
    .withColumn('one_year_history',F.when(F.col('first_tran_datetime') <=F.col('date_id'), 1).otherwise(0))
    )

# COMMAND ----------

# DBTITLE 1,Save Table Midway
flag_df.write.mode("overwrite").saveAsTable("tdm_seg.thanaritboo_th_lotuss_ktl_full_data_flag_tmp")

# COMMAND ----------

# DBTITLE 1,Load Table Back
flag_df = spark.table("tdm_seg.thanaritboo_th_lotuss_ktl_full_data_flag_tmp")

# COMMAND ----------

# MAGIC %md ## Promo features

# COMMAND ----------

# DBTITLE 1,Adding Promo
start_promoweek = date_dim.filter(F.col('date_id').between(timeframe_start, timeframe_end)).agg(F.min('promoweek_id')).collect()[0][0]
end_promoweek = date_dim.filter(F.col('date_id').between(timeframe_start, timeframe_end)).agg(F.max('promoweek_id')).collect()[0][0]

# print(start_promoweek, end_promoweek)

# COMMAND ----------

df_date_filtered = date_dim.filter(F.col('promoweek_id').between(start_promoweek, end_promoweek))

df_promo = spark.table('tdm.v_th_rpm_promo').filter(F.col('active_flag') == 'Y')\
                                                     .filter((F.col('change_type').isNotNull()))

df_promozone = spark.table('tdm.v_th_promo_zone')


df_prod = spark.table('tdm.v_prod_dim_c').filter(F.col('division_id').isin(product_division))\
                                                 .filter(F.col('country') == country)

df_trans_item = spark.table('tdm.v_transaction_item').filter(F.col('week_id').between(start_week, end_week))\
                                                    .filter(F.col('date_id').between(timeframe_start,timeframe_end))\
                                                    .filter(F.col('country') == country)\
                                                    .where((F.col('net_spend_amt')>0)&(F.col('product_qty')>0)&(F.col('date_id').isNotNull()))\
                                                    .filter(F.col('cc_flag') == 'cc')\
                                                    .dropDuplicates()

df_store = spark.table('tdm.v_store_dim_c').filter(F.col('format_id').isin(store_format))\
                                                .filter(F.col('country') == country)\
                                                .withColumn('format_name',F.when(F.col('format_id').isin([1,2,3]), 'Hypermarket')\
                                                                             .when(F.col('format_id') == 4, 'Supermarket')\
                                                                             .when(F.col('format_id') == 5, 'Mini Supermarket')\
                                                                             .otherwise(F.col('format_id')))\
                                                                             .dropDuplicates()

# COMMAND ----------

# Join tables into main table

df_promo_join = df_promo.select('promo_id', 'promo_offer_id', 'change_type', 'promo_start_date', 'promo_end_date', 
                                'promo_storegroup_id', 'upc_id', 'source').drop_duplicates()

df_promozone_join = df_promozone.select('zone_id', 'location') \
                                .withColumnRenamed('zone_id', 'promo_storegroup_id') \
                                .withColumnRenamed('location', 'store_id').drop_duplicates()

df_date_join = df_date_filtered.select('promoweek_id', 'date_id')

# "Explode" Promo table with one row per each date in each Promo ID & UPC ID combination
# can join on date_id now
df_promo_exploded = df_promo_join.join(df_date_join, 
                                       on=((df_promo_join['promo_start_date'] <= df_date_join['date_id']) &
                                          (df_promo_join['promo_end_date'] >= df_date_join['date_id'])),
                                       how='inner')

df_store_join = df_store.withColumn('store_format_name',F.when(F.col('format_id') == 5, 'GOFRESH') \
                                                                   .when(F.col('format_id') == 4, 'TALAD')
                                                                   .when(F.col('format_id').isin([1, 2, 3]), 'HDE')) \
                        .select('store_id', 'store_format_name').drop_duplicates()

# Explode the table further so each store ID is joined to each Promo ID & UPC ID & date combination
# Filter only for store formats 1-5
df_promo_with_stores = df_promo_exploded.join(df_promozone_join, on='promo_storegroup_id', how='left') \
                                        .join(df_store_join, on='store_id', how='inner')

# COMMAND ----------

# get txn from flag_df that appears in promo but only keep rows from flag_df 
promo_df = flag_df.join(df_promo_with_stores, on=['date_id','upc_id','store_id'], how='leftsemi')

non_promo_df = flag_df.join(promo_df, on=['household_id', 'transaction_uid', 'store_id', 'date_id', 'upc_id'], how='leftanti')

# COMMAND ----------

# DBTITLE 1,Flagging Promo
promo_df = promo_df.withColumn('discount_reason_code',F.lit('P'))\
                   .withColumn('promo_flag',F.lit('PROMO'))

non_promo_df = non_promo_df.withColumn('discount_reason_code',F.when((F.col('discount_amt') > 0), 'M')\
                                                              .otherwise('NONE'))\
                           .withColumn('promo_flag',F.when(F.col('discount_reason_code') == 'M', 'MARK_DOWN')\
                                                    .otherwise(F.col('discount_reason_code')))
                           
full_promo_df = promo_df.unionByName(non_promo_df)

# COMMAND ----------

# DBTITLE 1,Save Table 
full_promo_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_full_data_w_promo_tmp")

# COMMAND ----------

# DBTITLE 1,Load Table Back
flag_promo_df = spark.table("tdm_seg.kritawatkrai_th_year_full_data_w_promo_tmp")


# COMMAND ----------

# MAGIC %md ##Add dummy customer hh_id = -1

# COMMAND ----------

#add dummy customer
product_df = spark.table('tdm.v_prod_dim_c').select(['upc_id','brand_name','division_id','division_name','department_id','department_name','department_code','section_id','section_name','section_code','class_id','class_name','class_code','subclass_id','subclass_name','subclass_code'])\
                                                 .filter(F.col('division_id').isin(product_division))\
                                                 .filter(F.col('country') == country)

dep_exclude = ['1_36','1_92','13_25','13_32']
sec_exclude = ['3_7_130', '3_7_131', '3_8_132', '3_9_81', '10_43_34', '3_14_78', '13_6_205', '13_67_364',
    '1_36_708', '1_45_550', '1_92_992', '2_3_245', '2_4_253', '2_66_350', '13_25_249', '2_4_253',
    '13_25_250', '13_25_251', '13_67_359', '2_66_350', '4_10_84', '4_56_111', '10_46_549',
    '13_6_316', '13_25_249', '13_25_250', '13_25_251', '13_67_359', '13_32_617', '13_67_360', '2_4_719']
10
div_cust = product_df.select('division_id').distinct()\
                     .withColumn('household_id',F.lit(-1))
1
dep_schema = StructType([
    StructField("department_code", StringType(), nullable=False),
    StructField("household_id", IntegerType(), nullable=False)
])

missing_dep = [("2_33", -1)]

missing_dep_df = spark.createDataFrame(missing_dep, dep_schema)

dep_cust = product_df.select('department_code').filter(~(F.col('department_code').isin(dep_exclude))).distinct()\
                          .withColumn('household_id',F.lit(-1))\
                          .unionByName(missing_dep_df)

sec_schema = StructType([
    StructField("section_code", StringType(), nullable=False),
    StructField("household_id", IntegerType(), nullable=False)
])

missing_sec = [("3_14_80", -1),
               ("10_46_20", -1),
               ("2_33_704", -1)]

missing_sec_df = spark.createDataFrame(missing_sec, sec_schema)

sec_cust = product_df.select('section_code').filter(~(F.col('section_code').isin(sec_exclude))).distinct()\
                          .withColumn('household_id',F.lit(-1))\
                          .unionByName(missing_sec_df)

flag_promo_df = flag_promo_df.unionByName(div_cust, allowMissingColumns=True)\
                           .unionByName(dep_cust, allowMissingColumns=True)\
                           .unionByName(sec_cust, allowMissingColumns=True)\
                           .withColumn('grouped_department_code',F.when(F.col('department_code').isin('13_79', '13_77', '13_78'), '13_77&13_78&13_79')\
                                                                  .otherwise(F.col('department_code')))\
                           .withColumn('grouped_section_code',F.when(F.col('section_code').isin('1_2_187', '1_2_86'), '1_2_187&1_2_86')\
                                                                .when(F.col('section_code').isin('3_14_50', '3_51_416'), '3_14_50&3_51_416')\
                                                                .when(F.col('section_code').isin('2_3_195', '2_3_52'), '2_3_195&2_3_52')\
                                                                .otherwise(F.col('section_code')))
                           

# COMMAND ----------

# DBTITLE 1,Save Table 
flag_promo_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_full_data_w_promo_w_dummy_tmp")

# COMMAND ----------

# MAGIC %md #Aggregation

# COMMAND ----------

# DBTITLE 1,Load Table
full_flag_df = spark.table('tdm_seg.kritawatkrai_th_year_full_data_w_promo_w_dummy_tmp')

# COMMAND ----------

# MAGIC %md ##Aggregation - Total store

# COMMAND ----------

# DBTITLE 1,Total Spend Unit Visit
total_df = full_flag_df.groupBy('household_id')\
                       .agg(F.sum('net_spend_amt').alias('Total_Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Total_Visits'), \
                        F.sum('unit').alias('Total_Units'))
                       
# total_df.display()

# COMMAND ----------

# DBTITLE 1,Save Total
total_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_total_agg_data_tmp")

# COMMAND ----------

# DBTITLE 0,Untitledt
# MAGIC %md ## Aggregation - By product hierarchy + recency

# COMMAND ----------

# DBTITLE 1,Division Total
# filter out dummy of dep and sec
div_df = full_flag_df.filter((F.col('division_id').isNotNull()))\
                       .groupBy('household_id','division_id')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
div_df = div_df.join(total_df, on='household_id', how='inner')

div_df = div_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))
            
# div_df.display()


pivot_columns = div_df.select("division_id").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_div_df = div_df.groupBy("household_id").pivot("division_id", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    c = str(c)
    pivoted_div_df = pivoted_div_df.withColumnRenamed(c +"_Spend", "CAT_DIV_%" + c + "%_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "CAT_DIV_%" + c + "%_VISITS")\
                                   .withColumnRenamed(c +"_Units", "CAT_DIV_%" + c + "%_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "CAT_DIV_%" + c + "%_SPV")\
                                   .withColumnRenamed(c +"_UPV", "CAT_DIV_%" + c + "%_UPV")\
                                   .withColumnRenamed(c +"_SPU", "CAT_DIV_%" + c + "%_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_CAT_DIV_%" + c + "%_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_CAT_DIV_%" + c + "%_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_CAT_DIV_%" + c + "%_UNITS")

#exclude the dummy customer
pivoted_div_df = pivoted_div_df.filter(~(F.col('household_id') == -1))

# pivoted_div_df.display()
# print(pivot_columns)

# COMMAND ----------

# DBTITLE 1,Save Division
pivoted_div_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_div_agg_data_tmp")


# COMMAND ----------

# DBTITLE 1,Division 3/6/9 Recency
#LAST 3
l3_div_df = full_flag_df.filter(F.col('last_3_flag') == 'Y')\
                       .groupBy('household_id','division_id')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
l3_div_df = l3_div_df.join(total_df, on='household_id', how='inner')

l3_div_df = l3_div_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# div_df.display()

pivot_columns = l3_div_df.select("division_id").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l3_div_df = l3_div_df.groupBy("household_id").pivot("division_id", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    c = str(c)
    pivoted_l3_div_df = pivoted_l3_div_df.withColumnRenamed(c +"_Spend", "CAT_DIV_%" + c + "%_L3_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "CAT_DIV_%" + c + "%_L3_VISITS")\
                                   .withColumnRenamed(c +"_Units", "CAT_DIV_%" + c + "%_L3_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "CAT_DIV_%" + c + "%_L3_SPV")\
                                   .withColumnRenamed(c +"_UPV", "CAT_DIV_%" + c + "%_L3_UPV")\
                                   .withColumnRenamed(c +"_SPU", "CAT_DIV_%" + c + "%_L3_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_CAT_DIV_%" + c + "%_L3_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_CAT_DIV_%" + c + "%_L3_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_CAT_DIV_%" + c + "%_L3_UNITS")

pivoted_l3_div_df = pivoted_l3_div_df.filter(~(F.col('household_id') == -1))

# pivoted_l3_div_df.display()
# pivoted_div_df.count()
# -----------------------------------------------------------------------------------------------------------------------------------
#LAST 6

l6_div_df = full_flag_df.filter(F.col('last_6_flag') == 'Y')\
                       .groupBy('household_id','division_id')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
l6_div_df = l6_div_df.join(total_df, on='household_id', how='inner')

l6_div_df = l6_div_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# div_df.display()

pivot_columns = l6_div_df.select("division_id").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l6_div_df = l6_div_df.groupBy("household_id").pivot("division_id", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    c = str(c)
    pivoted_l6_div_df = pivoted_l6_div_df.withColumnRenamed(c +"_Spend", "CAT_DIV_%" + c + "%_L6_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "CAT_DIV_%" + c + "%_L6_VISITS")\
                                   .withColumnRenamed(c +"_Units", "CAT_DIV_%" + c + "%_L6_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "CAT_DIV_%" + c + "%_L6_SPV")\
                                   .withColumnRenamed(c +"_UPV", "CAT_DIV_%" + c + "%_L6_UPV")\
                                   .withColumnRenamed(c +"_SPU", "CAT_DIV_%" + c + "%_L6_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_CAT_DIV_%" + c + "%_L6_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_CAT_DIV_%" + c + "%_L6_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_CAT_DIV_%" + c + "%_L6_UNITS")


pivoted_l6_div_df = pivoted_l6_div_df.filter(~(F.col('household_id') == -1))

# pivoted_l6_div_df.display()
# pivoted_div_df.count()

# -----------------------------------------------------------------------------------------------------------------------------------
# LAST 9

l9_div_df = full_flag_df.filter(F.col('last_9_flag') == 'Y')\
                       .groupBy('household_id','division_id')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
l9_div_df = l9_div_df.join(total_df, on='household_id', how='inner')

l9_div_df = l9_div_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# div_df.display()

pivot_columns = l9_div_df.select("division_id").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l9_div_df = l9_div_df.groupBy("household_id").pivot("division_id", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    c = str(c)
    pivoted_l9_div_df = pivoted_l9_div_df.withColumnRenamed(c +"_Spend", "CAT_DIV_%" + c + "%_L9_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "CAT_DIV_%" + c + "%_L9_VISITS")\
                                   .withColumnRenamed(c +"_Units", "CAT_DIV_%" + c + "%_L9_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "CAT_DIV_%" + c + "%_L9_SPV")\
                                   .withColumnRenamed(c +"_UPV", "CAT_DIV_%" + c + "%_L9_UPV")\
                                   .withColumnRenamed(c +"_SPU", "CAT_DIV_%" + c + "%_L9_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_CAT_DIV_%" + c + "%_L9_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_CAT_DIV_%" + c + "%_L9_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_CAT_DIV_%" + c + "%_L9_UNITS")


pivoted_l9_div_df = pivoted_l9_div_df.filter(~(F.col('household_id') == -1))



# pivoted_l9_div_df.display()
# pivoted_div_df.count()


# COMMAND ----------

# DBTITLE 1,Save Division
pivoted_l3_div_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l3_div_agg_data_tmp")
pivoted_l6_div_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l6_div_agg_data_tmp")
pivoted_l9_div_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l9_div_agg_data_tmp")


# COMMAND ----------

# DBTITLE 1,Department Total
dep_df = full_flag_df.filter((F.col('grouped_department_code').isNotNull()))\
                        .filter(~(F.col('grouped_department_code').isin(dep_exclude)))\
                       .groupBy('household_id','grouped_department_code')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
dep_df = dep_df.join(total_df, on='household_id', how='inner')

dep_df = dep_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# dep_df.display()

pivot_columns = dep_df.select("grouped_department_code").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_dep_df = dep_df.groupBy("household_id").pivot("grouped_department_code", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    pivoted_dep_df = pivoted_dep_df.withColumnRenamed(c +"_Spend", "CAT_DEP_%" + c + "%_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "CAT_DEP_%" + c + "%_VISITS")\
                                   .withColumnRenamed(c +"_Units", "CAT_DEP_%" + c + "%_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "CAT_DEP_%" + c + "%_SPV")\
                                   .withColumnRenamed(c +"_UPV", "CAT_DEP_%" + c + "%_UPV")\
                                   .withColumnRenamed(c +"_SPU", "CAT_DEP_%" + c + "%_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_CAT_DEP_%" + c + "%_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_CAT_DEP_%" + c + "%_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_CAT_DEP_%" + c + "%_UNITS")

pivoted_dep_df = pivoted_dep_df.filter(~(F.col('household_id') == -1))

# pivoted_dep_df.display()
# pivoted_dep_df.count()


# COMMAND ----------

# DBTITLE 1,Save Dep
pivoted_dep_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_dep_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Department 3/6/9 Recency
#LAST 3
l3_dep_df = full_flag_df.filter(~(F.col('grouped_department_code').isin(dep_exclude)))\
                       .filter(F.col('last_3_flag') == 'Y')\
                       .groupBy('household_id','grouped_department_code')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
l3_dep_df = l3_dep_df.join(total_df, on='household_id', how='inner')

l3_dep_df = l3_dep_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# dep_df.display()

pivot_columns = l3_dep_df.select("grouped_department_code").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l3_dep_df = l3_dep_df.groupBy("household_id").pivot("grouped_department_code", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    pivoted_l3_dep_df = pivoted_l3_dep_df.withColumnRenamed(c +"_Spend", "CAT_DEP_%" + c + "%_L3_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "CAT_DEP_%" + c + "%_L3_VISITS")\
                                   .withColumnRenamed(c +"_Units", "CAT_DEP_%" + c + "%_L3_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "CAT_DEP_%" + c + "%_L3_SPV")\
                                   .withColumnRenamed(c +"_UPV", "CAT_DEP_%" + c + "%_L3_UPV")\
                                   .withColumnRenamed(c +"_SPU", "CAT_DEP_%" + c + "%_L3_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_CAT_DEP_%" + c + "%_L3_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_CAT_DEP_%" + c + "%_L3_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_CAT_DEP_%" + c + "%_L3_UNITS")


pivoted_l3_dep_df = pivoted_l3_dep_df.filter(~(F.col('household_id') == -1))

# pivoted_l3_dep_df.display()
# pivoted_dep_df.count()
# -----------------------------------------------------------------------------------------------------------------------------------
#LAST 6

l6_dep_df = full_flag_df.filter(~(F.col('grouped_department_code').isin(dep_exclude)))\
                       .filter(F.col('last_6_flag') == 'Y')\
                       .groupBy('household_id','grouped_department_code')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
l6_dep_df = l6_dep_df.join(total_df, on='household_id', how='inner')

l6_dep_df = l6_dep_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# dep_df.display()

pivot_columns = l6_dep_df.select("grouped_department_code").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l6_dep_df = l6_dep_df.groupBy("household_id").pivot("grouped_department_code", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    pivoted_l6_dep_df = pivoted_l6_dep_df.withColumnRenamed(c +"_Spend", "CAT_DEP_%" + c + "%_L6_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "CAT_DEP_%" + c + "%_L6_VISITS")\
                                   .withColumnRenamed(c +"_Units", "CAT_DEP_%" + c + "%_L6_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "CAT_DEP_%" + c + "%_L6_SPV")\
                                   .withColumnRenamed(c +"_UPV", "CAT_DEP_%" + c + "%_L6_UPV")\
                                   .withColumnRenamed(c +"_SPU", "CAT_DEP_%" + c + "%_L6_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_CAT_DEP_%" + c + "%_L6_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_CAT_DEP_%" + c + "%_L6_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_CAT_DEP_%" + c + "%_L6_UNITS")


pivoted_l6_dep_df = pivoted_l6_dep_df.filter(~(F.col('household_id') == -1))

# pivoted_l6_dep_df.display()
# pivoted_dep_df.count()

# -----------------------------------------------------------------------------------------------------------------------------------
# LAST 9

l9_dep_df = full_flag_df.filter(~(F.col('grouped_department_code').isin(dep_exclude)))\
                       .filter(F.col('last_9_flag') == 'Y')\
                       .groupBy('household_id','grouped_department_code')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
l9_dep_df = l9_dep_df.join(total_df, on='household_id', how='inner')

l9_dep_df = l9_dep_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# dep_df.display()

pivot_columns = l9_dep_df.select("grouped_department_code").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l9_dep_df = l9_dep_df.groupBy("household_id").pivot("grouped_department_code", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    pivoted_l9_dep_df = pivoted_l9_dep_df.withColumnRenamed(c +"_Spend", "CAT_DEP_%" + c + "%_L9_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "CAT_DEP_%" + c + "%_L9_VISITS")\
                                   .withColumnRenamed(c +"_Units", "CAT_DEP_%" + c + "%_L9_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "CAT_DEP_%" + c + "%_L9_SPV")\
                                   .withColumnRenamed(c +"_UPV", "CAT_DEP_%" + c + "%_L9_UPV")\
                                   .withColumnRenamed(c +"_SPU", "CAT_DEP_%" + c + "%_L9_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_CAT_DEP_%" + c + "%_L9_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_CAT_DEP_%" + c + "%_L9_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_CAT_DEP_%" + c + "%_L9_UNITS")


pivoted_l9_dep_df = pivoted_l9_dep_df.filter(~(F.col('household_id') == -1))

# pivoted_l9_dep_df.display()
# pivoted_dep_df.count()

# COMMAND ----------

pivoted_l3_dep_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l3_dep_agg_data_tmp")
pivoted_l6_dep_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l6_dep_agg_data_tmp")
pivoted_l9_dep_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l9_dep_agg_data_tmp")


# COMMAND ----------

# DBTITLE 1,Section
sec_df = full_flag_df.filter((F.col('grouped_section_code').isNotNull()))\
                       .filter(~(F.col('grouped_section_code').isin(sec_exclude)))\
                       .groupBy('household_id','grouped_section_code')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                                               
sec_df = sec_df.join(total_df, on='household_id', how='inner')

sec_df = sec_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))
      
# sec_df.display()

pivot_columns = sec_df.select("grouped_section_code").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_sec_df = sec_df.groupBy("household_id").pivot("grouped_section_code", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    pivoted_sec_df = pivoted_sec_df.withColumnRenamed(c +"_Spend", "CAT_SEC_%" + c + "%_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "CAT_SEC_%" + c + "%_VISITS")\
                                   .withColumnRenamed(c +"_Units", "CAT_SEC_%" + c + "%_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "CAT_SEC_%" + c + "%_SPV")\
                                   .withColumnRenamed(c +"_UPV", "CAT_SEC_%" + c + "%_UPV")\
                                   .withColumnRenamed(c +"_SPU", "CAT_SEC_%" + c + "%_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_CAT_SEC_%" + c + "%_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_CAT_SEC_%" + c + "%_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_CAT_SEC_%" + c + "%_UNITS")

pivoted_sec_df = pivoted_sec_df.filter(~(F.col('household_id') == -1))

# pivoted_sec_df.display()  
# pivoted_sec_df.count()


# COMMAND ----------

# DBTITLE 1,Save Section
pivoted_sec_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_sec_agg_data_tmp")

full_flag_df = full_flag_df.filter(~(F.col('household_id') == -1))


# COMMAND ----------

# DBTITLE 1,Monthly
monthly_df = full_flag_df.groupBy('household_id', 'end_month_date')\
                                .agg(F.sum('net_spend_amt').alias('Spend'), \
                               F.count_distinct('unique_transaction_uid').alias('Visits'), \
                                F.sum('unit').alias('Units'))\
                                .fillna(0)

monthly_df = monthly_df.groupBy('household_id').agg(
        round(F.avg(F.coalesce(F.col("Spend"),F.lit(0))), 2).alias("AVG_SPEND_MNTH"),
        round(F.stddev(F.coalesce(F.col("Spend"),F.lit(0))), 2).alias("SD_SPEND_MNTH"),
        round(F.min(F.coalesce(F.col("Spend"),F.lit(0))), 2).alias("MIN_SPEND_MNTH"),
        round(F.max(F.coalesce(F.col("Spend"),F.lit(0))), 2).alias("MAX_SPEND_MNTH"),
        round(F.avg(F.coalesce(F.col("Visits"),F.lit(0))), 2).alias("AVG_VISITS_MNTH"),
        round(F.stddev(F.coalesce(F.col("Visits"),F.lit(0))), 2).alias("SD_VISITS_MNTH"),
        round(F.min(F.coalesce(F.col("Visits"),F.lit(0))), 2).alias("MIN_VISITS_MNTH"),
        round(F.max(F.coalesce(F.col("Visits"),F.lit(0))), 2).alias("MAX_VISITS_MNTH"),
        round(F.avg(F.coalesce(F.col("Units"),F.lit(0))), 2).alias("AVG_UNITS_MNTH"),
        round(F.stddev(F.coalesce(F.col("Units"),F.lit(0))), 2).alias("SD_UNITS_MNTH"),
        round(F.min(F.coalesce(F.col("Units"),F.lit(0))), 2).alias("MIN_UNITS_MNTH"),
        round(F.max(F.coalesce(F.col("Units"),F.lit(0))), 2).alias("MAX_UNITS_MNTH")
).fillna(0)

    
# monthly_df.display()

# COMMAND ----------

# DBTITLE 1,Save Monthly
monthly_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_monthly_agg_data_tmp")


# COMMAND ----------

# DBTITLE 1,Quarterly 1/2/3/4
#------------------------------------------------------------------------------------------------------------------------------------
#Q1
qtr1_df = full_flag_df.filter(F.col('q1_flag') == 'Y')\
                       .groupBy('household_id','app_year_qtr')\
                       .agg(F.sum('net_spend_amt').alias('Q1_SPEND'), \
                       F.count_distinct('unique_transaction_uid').alias('Q1_VISITS'), \
                        F.sum('unit').alias('Q1_UNITS'))
                                               
qtr1_df = qtr1_df.join(total_df, on='household_id', how='inner')

qtr1_df = qtr1_df.withColumn('Q1_SPV',F.when((F.col('Q1_VISITS').isNull()) | (F.col('Q1_VISITS') == 0), 0)\
                                 .otherwise(F.col('Q1_SPEND') /F.col('Q1_VISITS')))\
               .withColumn('Q1_UPV',F.when((F.col('Q1_VISITS').isNull()) | (F.col('Q1_VISITS') == 0), 0)\
                                 .otherwise(F.col('Q1_UNITS') /F.col('Q1_VISITS')))\
               .withColumn('Q1_SPU',F.when((F.col('Q1_UNITS').isNull()) | (F.col('Q1_UNITS') == 0), 0)\
                                 .otherwise(F.col('Q1_SPEND') /F.col('Q1_UNITS')))\
               .withColumn('PCT_Q1_SPEND',F.col('Q1_SPEND') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Q1_VISITS',F.col('Q1_VISITS') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Q1_UNITS',F.col('Q1_UNITS') * 100 /F.col('Total_Units'))\
               .drop('Total_Spend', 'Total_Visits', 'Total_Units', 'app_year_qtr')

# qtr1_df.display()

#------------------------------------------------------------------------------------------------------------------------------------
#Q2
# pivoted_qtr1_df.display()

qtr2_df = full_flag_df.filter(F.col('q2_flag') == 'Y')\
                       .groupBy('household_id','app_year_qtr')\
                       .agg(F.sum('net_spend_amt').alias('Q2_SPEND'), \
                       F.count_distinct('unique_transaction_uid').alias('Q2_VISITS'), \
                        F.sum('unit').alias('Q2_UNITS'))

qtr2_df = qtr2_df.join(total_df, on='household_id', how='inner')

qtr2_df = qtr2_df.withColumn('Q2_SPV',F.when((F.col('Q2_VISITS').isNull()) | (F.col('Q2_VISITS') == 0), 0)\
                                 .otherwise(F.col('Q2_SPEND') /F.col('Q2_VISITS')))\
               .withColumn('Q2_UPV',F.when((F.col('Q2_VISITS').isNull()) | (F.col('Q2_VISITS') == 0), 0)\
                                 .otherwise(F.col('Q2_UNITS') /F.col('Q2_VISITS')))\
               .withColumn('Q2_SPU',F.when((F.col('Q2_UNITS').isNull()) | (F.col('Q2_UNITS') == 0), 0)\
                                 .otherwise(F.col('Q2_SPEND') /F.col('Q2_UNITS')))\
               .withColumn('PCT_Q2_SPEND',F.col('Q2_SPEND') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Q2_VISITS',F.col('Q2_VISITS') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Q2_UNITS',F.col('Q2_UNITS') * 100 /F.col('Total_Units'))\
               .drop('Total_Spend', 'Total_Visits', 'Total_Units', 'app_year_qtr')


# qtr1_df.display()


#------------------------------------------------------------------------------------------------------------------------------------
#Q3

qtr3_df = full_flag_df.filter(F.col('q3_flag') == 'Y')\
                       .groupBy('household_id','app_year_qtr')\
                       .agg(F.sum('net_spend_amt').alias('Q3_SPEND'), \
                       F.count_distinct('unique_transaction_uid').alias('Q3_VISITS'), \
                        F.sum('unit').alias('Q3_UNITS'))
                                               
qtr3_df = qtr3_df.join(total_df, on='household_id', how='inner')

qtr3_df = qtr3_df.withColumn('Q3_SPV',F.when((F.col('Q3_VISITS').isNull()) | (F.col('Q3_VISITS') == 0), 0)\
                                 .otherwise(F.col('Q3_SPEND') /F.col('Q3_VISITS')))\
               .withColumn('Q3_UPV',F.when((F.col('Q3_VISITS').isNull()) | (F.col('Q3_VISITS') == 0), 0)\
                                 .otherwise(F.col('Q3_UNITS') /F.col('Q3_VISITS')))\
               .withColumn('Q3_SPU',F.when((F.col('Q3_UNITS').isNull()) | (F.col('Q3_UNITS') == 0), 0)\
                                 .otherwise(F.col('Q3_SPEND') /F.col('Q3_UNITS')))\
               .withColumn('PCT_Q3_SPEND',F.col('Q3_SPEND') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Q3_VISITS',F.col('Q3_VISITS') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Q3_UNITS',F.col('Q3_UNITS') * 100 /F.col('Total_Units'))\
               .drop('Total_Spend', 'Total_Visits', 'Total_Units', 'app_year_qtr')

# qtr1_df.display()


#------------------------------------------------------------------------------------------------------------------------------------
#Q4

qtr4_df = full_flag_df.filter(F.col('q4_flag') == 'Y')\
                       .groupBy('household_id','app_year_qtr')\
                       .agg(F.sum('net_spend_amt').alias('Q4_SPEND'), \
                       F.count_distinct('unique_transaction_uid').alias('Q4_VISITS'), \
                        F.sum('unit').alias('Q4_UNITS'))
                                               
qtr4_df = qtr4_df.join(total_df, on='household_id', how='inner')

qtr4_df = qtr4_df.withColumn('Q4_SPV',F.when((F.col('Q4_VISITS').isNull()) | (F.col('Q4_VISITS') == 0), 0)\
                                 .otherwise(F.col('Q4_SPEND') /F.col('Q4_VISITS')))\
               .withColumn('Q4_UPV',F.when((F.col('Q4_VISITS').isNull()) | (F.col('Q4_VISITS') == 0), 0)\
                                 .otherwise(F.col('Q4_UNITS') /F.col('Q4_VISITS')))\
               .withColumn('Q4_SPU',F.when((F.col('Q4_UNITS').isNull()) | (F.col('Q4_UNITS') == 0), 0)\
                                 .otherwise(F.col('Q4_SPEND') /F.col('Q4_UNITS')))\
               .withColumn('PCT_Q4_SPEND',F.col('Q4_SPEND') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Q4_VISITS',F.col('Q4_VISITS') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Q4_UNITS',F.col('Q4_UNITS') * 100 /F.col('Total_Units'))\
               .drop('Total_Spend', 'Total_Visits', 'Total_Units', 'app_year_qtr')


# COMMAND ----------

# DBTITLE 1,Save Quarter
qtr1_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_qtr1_agg_data_tmp")
qtr2_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_qtr2_agg_data_tmp")
qtr3_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_qtr3_agg_data_tmp")
qtr4_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_qtr4_agg_data_tmp")


# COMMAND ----------

# DBTITLE 1,Weekly Spend
weekly_df = full_flag_df.groupBy('household_id', 'week_of_month')\
                                .agg(F.sum('net_spend_amt').alias('Spend'), \
                               F.count_distinct('unique_transaction_uid').alias('Visits'), \
                                F.sum('unit').alias('Units'))\
                                .fillna(0)

weekly_df = weekly_df.groupBy('household_id').agg(
        round(F.avg(F.coalesce(F.col("Spend"),F.lit(0))), 2).alias("AVG_SPEND_WK"),
        round(F.stddev(F.coalesce(F.col("Spend"),F.lit(0))), 2).alias("SD_SPEND_WK"),
        round(F.avg(F.coalesce(F.col("Visits"),F.lit(0))), 2).alias("AVG_VISITS_WK"),
        round(F.stddev(F.coalesce(F.col("Visits"),F.lit(0))), 2).alias("SD_VISITS_WK"),
        round(F.avg(F.coalesce(F.col("Units"),F.lit(0))), 2).alias("AVG_UNITS_WK"),
        round(F.stddev(F.coalesce(F.col("Units"),F.lit(0))), 2).alias("SD_UNITS_WK")
    ).fillna(0)
              

# weekly_df.display()

# COMMAND ----------

# DBTITLE 1,Save Weekly
weekly_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_weekly_agg_data_tmp")


# COMMAND ----------

# DBTITLE 1,Festival Spend
fest_df = full_flag_df.groupBy('household_id','fest_flag')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                                               
fest_df = fest_df.join(total_df, on='household_id', how='inner')

fest_df = fest_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# fest_df.display()

pivot_columns = fest_df.select("fest_flag").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_fest_df = fest_df.groupBy("household_id").pivot("fest_flag", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    pivoted_fest_df = pivoted_fest_df.withColumnRenamed(c +"_Spend", "FEST_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "FEST_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "FEST_" + c + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "FEST_" + c + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "FEST_" + c + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "FEST_" + c + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_FEST_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_FEST_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_FEST_" + c + "_UNITS")

# pivoted_fest_df.display()

# COMMAND ----------

# DBTITLE 1,Save Fest
pivoted_fest_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_fest_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Count Distinct
count_df = full_flag_df.groupBy('household_id')\
                                .agg(countDistinct('department_id').alias('N_DISTINCT_DEP'),\
                                    F.count_distinct('division_id').alias('N_DISTINCT_DIV'),\
                                    F.count_distinct('section_code').alias('N_DISTINCT_SEC'),\
                                    F.count_distinct('class_code').alias('N_DISTINCT_CLASS'),\
                                    F.count_distinct('subclass_code').alias('N_DISTINCT_SUBCLASS'),\
                                    F.count_distinct('store_id').alias('N_STORES'))\
                                .fillna(0)

# count_df.display()

# COMMAND ----------

# DBTITLE 1,Save Count
count_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_count_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Store Region (PCT_SA_BKK_UNITS)
# region_list = ['Unidentified', 'South', 'Central', 'BKK & Vicinities', 'North', 'Northeast']

store_region_df = full_flag_df.groupBy('household_id','store_region')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                                               
store_region_df = store_region_df.join(total_df, on='household_id', how='inner')

store_region_df = store_region_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# store_region_df.display()

pivot_columns = store_region_df.select("store_region").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_store_region_df = store_region_df.groupBy("household_id").pivot("store_region", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    pivoted_store_region_df = pivoted_store_region_df.withColumnRenamed(c +"_Spend", "SA_" + c.upper().replace(" ","") + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "SA_" + c.upper().replace(" ","") + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "SA_" + c.upper().replace(" ","") + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "SA_" + c.upper().replace(" ","") + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "SA_" + c.upper().replace(" ","") + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "SA_" + c.upper().replace(" ","") + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_SA_" + c.upper().replace(" ","") + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_SA_" + c.upper().replace(" ","") + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_SA_" + c.upper().replace(" ","") + "_UNITS")

# pivoted_store_region_df.display()
# ping()

# COMMAND ----------

# DBTITLE 1,Save Store Region
pivoted_store_region_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_store_region_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Time of Day
time_day_df = full_flag_df.groupBy('household_id','time_of_day')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                                               
time_day_df = time_day_df.join(total_df, on='household_id', how='inner')

time_day_df = time_day_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))\
               .withColumn('PCT_PCT_Spend',F.col('PCT_Spend') /F.col('Total_Spend'))\
               .withColumn('PCT_PCT_Visits',F.col('PCT_Visits') /F.col('Total_Visits'))\
               .withColumn('PCT_PCT_Units',F.col('PCT_Units') /F.col('Total_Units'))
               

# time_day_df.display()

pivot_columns = time_day_df.select("time_of_day").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_time_day_df = time_day_df.groupBy("household_id").pivot("time_of_day", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units"),
).fillna(0)

for c in pivot_columns:
    pivoted_time_day_df = pivoted_time_day_df.withColumnRenamed(c +"_Spend", "TIME_" + c.upper() + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "TIME_" + c.upper() + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "TIME_" + c.upper() + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "TIME_" + c.upper() + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "TIME_" + c.upper() + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "TIME_" + c.upper() + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_TIME_" + c.upper() + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_TIME_" + c.upper() + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_TIME_" + c.upper() + "_UNITS")


# pivoted_time_day_df.display()

# COMMAND ----------

# DBTITLE 1,Save Time of Day
pivoted_time_day_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_time_day_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Store Format
store_format_df = full_flag_df.groupBy('household_id','format_name')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                                               
store_format_df = store_format_df.join(total_df, on='household_id', how='inner')

store_format_df = store_format_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# store_format_df.display()

pivot_columns = store_format_df.select("format_name").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_store_format_df = store_format_df.groupBy("household_id").pivot("format_name", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    pivoted_store_format_df = pivoted_store_format_df.withColumnRenamed(c +"_Spend", "SF_" + c.upper().replace(" ","") + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "SF_" + c.upper().replace(" ","") + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "SF_" + c.upper().replace(" ","") + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "SF_" + c.upper().replace(" ","") + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "SF_" + c.upper().replace(" ","") + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "SF_" + c.upper().replace(" ","") + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_SF_" + c.upper().replace(" ","") + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_SF_" + c.upper().replace(" ","") + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_SF_" + c.upper().replace(" ","") + "_UNITS")

# pivoted_store_format_df.display()

# COMMAND ----------

# DBTITLE 1,Save Store Format
pivoted_store_format_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_store_format_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Premium / Budget Spend
price_level_df = full_flag_df.groupBy('household_id','price_level')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                                               
price_level_df = price_level_df.join(total_df, on='household_id', how='inner')

price_level_df = price_level_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))\
               .withColumn('PCT_PCT_Spend',F.col('Spend') /F.col('Total_Spend'))\
               .withColumn('PCT_PCT_Visits',F.col('Visits') /F.col('Total_Visits'))\
               .withColumn('PCT_PCT_Units',F.col('Units') /F.col('Total_Units'))
               

# price_level_df.display()

pivot_columns = price_level_df.select("price_level").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_price_level_df = price_level_df.groupBy("household_id").pivot("price_level", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units"),
    first("PCT_PCT_Spend").alias("PCT_PCT_Spend"),
    first("PCT_PCT_Visits").alias("PCT_PCT_Visits"),
    first("PCT_PCT_Units").alias("PCT_PCT_Units")
).fillna(0)

for c in pivot_columns:
    pivoted_price_level_df = pivoted_price_level_df.withColumnRenamed(c +"_PCT_Spend", "PCT_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_" + c + "_UNITS")\
                                   .withColumnRenamed(c +"_PCT_PCT_Spend", "PCT_PCT_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_PCT_Visits", "PCT_PCT_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_PCT_Units", "PCT_PCT_" + c + "_UNITS")

# pivoted_price_level_df.display()

# COMMAND ----------

# DBTITLE 1,Save Price Level
pivoted_price_level_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_price_level_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Last Weekend Spend
last_wknd_df = full_flag_df.filter(F.col('last_weekend_flag') == 'Y')\
                       .groupBy('household_id')\
                       .agg(F.sum('net_spend_amt').alias('LAST_WKND_SPEND'), \
                       F.count_distinct('unique_transaction_uid').alias('LAST_WKND_VISITS'), \
                        F.sum('unit').alias('LAST_WKND_UNITS'))
                                               
last_wknd_df = last_wknd_df.join(total_df, on='household_id', how='inner')

last_wknd_df = last_wknd_df.withColumn('LAST_WKND_SPV',F.when((F.col('LAST_WKND_VISITS').isNull()) | (F.col('LAST_WKND_VISITS') == 0), 0)\
                                 .otherwise(F.col('LAST_WKND_SPEND') /F.col('LAST_WKND_VISITS')))\
               .withColumn('LAST_WKND_UPV',F.when((F.col('LAST_WKND_VISITS').isNull()) | (F.col('LAST_WKND_VISITS') == 0), 0)\
                                 .otherwise(F.col('LAST_WKND_UNITS') /F.col('LAST_WKND_VISITS')))\
               .withColumn('LAST_WKND_SPU',F.when((F.col('LAST_WKND_UNITS').isNull()) | (F.col('LAST_WKND_UNITS') == 0), 0)\
                                 .otherwise(F.col('LAST_WKND_SPEND') /F.col('LAST_WKND_UNITS')))\
               .withColumn('PCT_LAST_WKND_SPEND',F.col('LAST_WKND_SPEND') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_LAST_WKND_VISITS',F.col('LAST_WKND_VISITS') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_LAST_WKND_UNITS',F.col('LAST_WKND_UNITS') * 100 /F.col('Total_Units'))\
               .withColumn('PCT_PCT_LAST_WKND_SPEND',F.col('LAST_WKND_SPEND')  /F.col('Total_Spend'))\
               .withColumn('PCT_PCT_LAST_WKND_VISITS',F.col('LAST_WKND_VISITS')  /F.col('Total_Visits'))\
               .withColumn('PCT_PCT_LAST_WKND_UNITS',F.col('LAST_WKND_UNITS')  /F.col('Total_Units'))\
               .fillna(0)\
               .drop('Total_Spend', 'Total_Visits', 'Total_Units')


# last_wknd_df.display()

# COMMAND ----------

# DBTITLE 1,Save Last Weekend
last_wknd_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_last_wknd_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Weekend_Y_Spend
wknd_df = full_flag_df.filter(F.col('weekend_flag') == 'Y')\
                       .groupBy('household_id')\
                       .agg(F.sum('net_spend_amt').alias('WKND_FLAG_Y_SPEND'), \
                       F.count_distinct('unique_transaction_uid').alias('WKND_FLAG_Y_VISITS'), \
                        F.sum('unit').alias('WKND_FLAG_Y_UNITS'))
                                               
wknd_df = wknd_df.join(total_df, on='household_id', how='inner')

wknd_df = wknd_df.withColumn('WKND_FLAG_Y_SPV',F.when((F.col('WKND_FLAG_Y_VISITS').isNull()) | (F.col('WKND_FLAG_Y_VISITS') == 0), 0)\
                                 .otherwise(F.col('WKND_FLAG_Y_SPEND') /F.col('WKND_FLAG_Y_VISITS')))\
               .withColumn('WKND_FLAG_Y_UPV',F.when((F.col('WKND_FLAG_Y_VISITS').isNull()) | (F.col('WKND_FLAG_Y_VISITS') == 0), 0)\
                                 .otherwise(F.col('WKND_FLAG_Y_UNITS') /F.col('WKND_FLAG_Y_VISITS')))\
               .withColumn('WKND_FLAG_Y_SPU',F.when((F.col('WKND_FLAG_Y_UNITS').isNull()) | (F.col('WKND_FLAG_Y_UNITS') == 0), 0)\
                                 .otherwise(F.col('WKND_FLAG_Y_SPEND') /F.col('WKND_FLAG_Y_UNITS')))\
               .withColumn('PCT_WKND_FLAG_Y_SPEND',F.col('WKND_FLAG_Y_SPEND') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_WKND_FLAG_Y_VISITS',F.col('WKND_FLAG_Y_VISITS') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_WKND_FLAG_Y_UNITS',F.col('WKND_FLAG_Y_UNITS') * 100 /F.col('Total_Units'))\
               .fillna(0)\
               .drop('Total_Spend', 'Total_Visits', 'Total_Units')

# wknd_df.display()

# COMMAND ----------

# DBTITLE 1,Save Weekend
wknd_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_wknd_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Weekend_N_Spend
wkday_df = full_flag_df.filter(F.col('weekend_flag') == 'N')\
                       .groupBy('household_id')\
                       .agg(F.sum('net_spend_amt').alias('WKND_FLAG_N_SPEND'), \
                       F.count_distinct('unique_transaction_uid').alias('WKND_FLAG_N_VISITS'), \
                        F.sum('unit').alias('WKND_FLAG_N_UNITS'))
                                               
wkday_df = wkday_df.join(total_df, on='household_id', how='inner')

wkday_df = wkday_df.withColumn('WKND_FLAG_N_SPV',F.when((F.col('WKND_FLAG_N_VISITS').isNull()) | (F.col('WKND_FLAG_N_VISITS') == 0), 0)\
                                 .otherwise(F.col('WKND_FLAG_N_SPEND') /F.col('WKND_FLAG_N_VISITS')))\
               .withColumn('WKND_FLAG_N_UPV',F.when((F.col('WKND_FLAG_N_VISITS').isNull()) | (F.col('WKND_FLAG_N_VISITS') == 0), 0)\
                                 .otherwise(F.col('WKND_FLAG_N_UNITS') /F.col('WKND_FLAG_N_VISITS')))\
               .withColumn('WKND_FLAG_N_SPU',F.when((F.col('WKND_FLAG_N_UNITS').isNull()) | (F.col('WKND_FLAG_N_UNITS') == 0), 0)\
                                 .otherwise(F.col('WKND_FLAG_N_SPEND') /F.col('WKND_FLAG_N_UNITS')))\
               .withColumn('PCT_WKND_FLAG_N_SPEND',F.col('WKND_FLAG_N_SPEND') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_WKND_FLAG_N_VISITS',F.col('WKND_FLAG_N_VISITS') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_WKND_FLAG_N_UNITS',F.col('WKND_FLAG_N_UNITS') * 100 /F.col('Total_Units'))\
               .fillna(0)\
               .drop('Total_Spend', 'Total_Visits', 'Total_Units')

# wkday_df.display()

# COMMAND ----------

# DBTITLE 1,Save Weekday
wkday_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_wkday_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Recency
l3_df = full_flag_df.filter(F.col('last_3_flag') == 'Y')\
                       .groupBy('household_id')\
                       .agg(F.sum('net_spend_amt').alias('L3_SPEND'), \
                       F.count_distinct('unique_transaction_uid').alias('L3_VISITS'), \
                        F.sum('unit').alias('L3_UNITS'))
                                               
l3_df = l3_df.join(total_df, on='household_id', how='inner')

l3_df = l3_df.withColumn('L3_SPV',F.when((F.col('L3_VISITS').isNull()) | (F.col('L3_VISITS') == 0), 0)\
                                 .otherwise(F.col('L3_SPEND') /F.col('L3_VISITS')))\
               .withColumn('L3_UPV',F.when((F.col('L3_VISITS').isNull()) | (F.col('L3_VISITS') == 0), 0)\
                                 .otherwise(F.col('L3_UNITS') /F.col('L3_VISITS')))\
               .withColumn('L3_SPU',F.when((F.col('L3_UNITS').isNull()) | (F.col('L3_UNITS') == 0), 0)\
                                 .otherwise(F.col('L3_SPEND') /F.col('L3_UNITS')))\
               .withColumn('PCT_L3_SPEND',F.col('L3_SPEND') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_L3_VISITS',F.col('L3_VISITS') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_L3_UNITS',F.col('L3_UNITS') * 100 /F.col('Total_Units'))\
               .drop('Total_Spend', 'Total_Visits', 'Total_Units', 'app_year_qtr')

# -------------------------------------------------------------------------------------------------------------


l6_df = full_flag_df.filter(F.col('last_6_flag') == 'Y')\
                       .groupBy('household_id')\
                       .agg(F.sum('net_spend_amt').alias('L6_SPEND'), \
                       F.count_distinct('unique_transaction_uid').alias('L6_VISITS'), \
                        F.sum('unit').alias('L6_UNITS'))
                                               
l6_df = l6_df.join(total_df, on='household_id', how='inner')

l6_df = l6_df.withColumn('L6_SPV',F.when((F.col('L6_VISITS').isNull()) | (F.col('L6_VISITS') == 0), 0)\
                                 .otherwise(F.col('L6_SPEND') /F.col('L6_VISITS')))\
               .withColumn('L6_UPV',F.when((F.col('L6_VISITS').isNull()) | (F.col('L6_VISITS') == 0), 0)\
                                 .otherwise(F.col('L6_UNITS') /F.col('L6_VISITS')))\
               .withColumn('L6_SPU',F.when((F.col('L6_UNITS').isNull()) | (F.col('L6_UNITS') == 0), 0)\
                                 .otherwise(F.col('L6_SPEND') /F.col('L6_UNITS')))\
               .withColumn('PCT_L6_SPEND',F.col('L6_SPEND') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_L6_VISITS',F.col('L6_VISITS') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_L6_UNITS',F.col('L6_UNITS') * 100 /F.col('Total_Units'))\
               .drop('Total_Spend', 'Total_Visits', 'Total_Units', 'app_year_qtr')

# -------------------------------------------------------------------------------------------------------------

l9_df = full_flag_df.filter(F.col('last_9_flag') == 'Y')\
                       .groupBy('household_id')\
                       .agg(F.sum('net_spend_amt').alias('L9_SPEND'), \
                       F.count_distinct('unique_transaction_uid').alias('L9_VISITS'), \
                        F.sum('unit').alias('L9_UNITS'))
                                               
l9_df = l9_df.join(total_df, on='household_id', how='inner')

l9_df = l9_df.withColumn('L9_SPV',F.when((F.col('L9_VISITS').isNull()) | (F.col('L9_VISITS') == 0), 0)\
                                 .otherwise(F.col('L9_SPEND') /F.col('L9_VISITS')))\
               .withColumn('L9_UPV',F.when((F.col('L9_VISITS').isNull()) | (F.col('L9_VISITS') == 0), 0)\
                                 .otherwise(F.col('L9_UNITS') /F.col('L9_VISITS')))\
               .withColumn('L9_SPU',F.when((F.col('L9_UNITS').isNull()) | (F.col('L9_UNITS') == 0), 0)\
                                 .otherwise(F.col('L9_SPEND') /F.col('L9_UNITS')))\
               .withColumn('PCT_L9_SPEND',F.col('L9_SPEND') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_L9_VISITS',F.col('L9_VISITS') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_L9_UNITS',F.col('L9_UNITS') * 100 /F.col('Total_Units'))\
               .drop('Total_Spend', 'Total_Visits', 'Total_Units', 'app_year_qtr')

# COMMAND ----------

# DBTITLE 1,Save Recency
l3_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l3_agg_data_tmp")
l6_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l6_agg_data_tmp")
l9_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l9_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Payment Method 
# payment_list = ['CASH', 'CARD', 'COUPON', 'VOUCHER']

payment_df = full_flag_df.groupBy('household_id','payment_flag')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                                               
payment_df = payment_df.join(total_df, on='household_id', how='inner')

payment_df = payment_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))


pivot_columns = payment_df.select("payment_flag").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_payment_df = payment_df.groupBy("household_id").pivot("payment_flag", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
    if (c == 'VOUCHER' or c == 'COUPON'):
      pivoted_payment_df = pivoted_payment_df.withColumnRenamed(c +"_Spend", "PYMNT_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "PYMNT_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "PYMNT_" + c + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "PYMNT_" + c + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "PYMNT_" + c + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "PYMNT_" + c + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_PYMNT_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_PYMNT_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_PYMNT_" + c + "_UNITS")
    else:
        pivoted_payment_df = pivoted_payment_df.withColumnRenamed(c +"_Spend", "PYMT_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "PYMT_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "PYMT_" + c + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "PYMT_" + c + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "PYMT_" + c + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "PYMT_" + c + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_PYMT_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_PYMT_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_PYMT_" + c + "_UNITS")

# pivoted_payment_df.display()



# COMMAND ----------

# DBTITLE 1,Save Payment Method
pivoted_payment_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_payment_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Promo General
discount_promo_df = full_flag_df.groupBy('household_id','promo_flag')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                                               
discount_promo_df = discount_promo_df.join(total_df, on='household_id', how='inner')

discount_promo_df = discount_promo_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))


pivot_columns = discount_promo_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_discount_promo_df = discount_promo_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
  if (c == 'PROMO' or c == 'MARK_DOWN'):
      pivoted_discount_promo_df = pivoted_discount_promo_df.withColumnRenamed(c +"_Spend", c + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", c + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", c + "_UNITS")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_" + c + "_UNITS")
  else:
      pivoted_discount_promo_df = pivoted_discount_promo_df.withColumnRenamed(c +"_Spend", "DISCOUNT_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "DISCOUNT_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "DISCOUNT_" + c + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "DISCOUNT_" + c + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "DISCOUNT_" + c + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "DISCOUNT_" + c + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_DISCOUNT_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_DISCOUNT_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_DISCOUNT_" + c + "_UNITS")


# COMMAND ----------

# DBTITLE 1,Save Promo
pivoted_discount_promo_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_discount_promo_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Promo Recency
#LAST 3
l3_promo_df = full_flag_df.filter(F.col('last_3_flag') == 'Y')\
                       .groupBy('household_id','promo_flag')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
l3_promo_df = l3_promo_df.join(total_df, on='household_id', how='inner')

l3_promo_df = l3_promo_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))


pivot_columns = l3_promo_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l3_promo_df = l3_promo_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
      pivoted_l3_promo_df = pivoted_l3_promo_df.withColumnRenamed(c +"_Spend", "L3_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "L3_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "L3_" + c + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "L3_" + c + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "L3_" + c + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "L3_" + c + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_L3_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_L3_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_L3_" + c + "_UNITS")

# -----------------------------------------------------------------------------------------------------------------------------------
#LAST 6

l6_promo_df = full_flag_df.filter(F.col('last_6_flag') == 'Y')\
                       .groupBy('household_id','promo_flag')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
l6_promo_df = l6_promo_df.join(total_df, on='household_id', how='inner')

l6_promo_df = l6_promo_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# dep_df.display()

pivot_columns = l6_promo_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l6_promo_df = l6_promo_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
      pivoted_l6_promo_df = pivoted_l6_promo_df.withColumnRenamed(c +"_Spend", "L6_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "L6_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "L6_" + c + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "L6_" + c + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "L6_" + c + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "L6_" + c + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_L6_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_L6_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_L6_" + c + "_UNITS")

# -----------------------------------------------------------------------------------------------------------------------------------
# LAST 9

l9_promo_df = full_flag_df.filter(F.col('last_9_flag') == 'Y')\
                       .groupBy('household_id','promo_flag')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
l9_promo_df = l9_promo_df.join(total_df, on='household_id', how='inner')

l9_promo_df = l9_promo_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

# dep_df.display()

pivot_columns = l9_promo_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l9_promo_df = l9_promo_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
      pivoted_l9_promo_df = pivoted_l9_promo_df.withColumnRenamed(c +"_Spend", "L9_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "L9_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "L9_" + c + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "L9_" + c + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "L9_" + c + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "L9_" + c + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_L9_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_L9_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_L9_" + c + "_UNITS")


# COMMAND ----------

# DBTITLE 1,Save Promo Recency
pivoted_l3_promo_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l3_promo_agg_data_tmp")
pivoted_l6_promo_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l6_promo_agg_data_tmp")
pivoted_l9_promo_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l9_promo_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Promo Time of Day
time_promo_df = full_flag_df.groupBy('household_id','time_of_day','promo_flag')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                                               
time_promo_df = time_promo_df.join(total_df, on='household_id', how='inner')

time_promo_df = time_promo_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))\
               .withColumn('time_promo',F.concat_ws('_',F.col('time_of_day'),F.col('promo_flag')))

# time_promo_df.display()

pivot_columns = time_promo_df.select("time_promo").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_time_promo_df = time_promo_df.groupBy("household_id").pivot("time_promo", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
      pivoted_time_promo_df = pivoted_time_promo_df.withColumnRenamed(c +"_Spend", "TIME_" + c.upper() + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "TIME_" + c.upper() + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "TIME_" + c.upper() + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "TIME_" + c.upper() + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "TIME_" + c.upper() + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "TIME_" + c.upper() + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_TIME_" + c.upper() + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_TIME_" + c.upper() + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_TIME_" + c.upper() + "_UNITS")


# COMMAND ----------

# DBTITLE 1,Save Promo Time 
pivoted_time_promo_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_time_promo_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Item Recency
# L3
l3_promo_item_df = full_flag_df.filter(F.col('last_3_flag') == 'Y')\
                       .groupBy('household_id', 'promo_flag')\
                       .agg(F.sum('unit').alias('Items'))

pivot_columns = l3_promo_item_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l3_promo_item_df = l3_promo_item_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
    first("Items")
).fillna(0)

for c in pivot_columns:
      pivoted_l3_promo_item_df = pivoted_l3_promo_item_df.withColumnRenamed(c, "L3_" + c + "_ITEMS")

# pivoted_l3_promo_item_df.display()

#------------------------------------------------------------------------------------------------------------------------------------
# L6

l6_promo_item_df = full_flag_df.filter(F.col('last_6_flag') == 'Y')\
                       .groupBy('household_id', 'promo_flag')\
                       .agg(F.sum('unit').alias('Items'))

pivot_columns = l6_promo_item_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l6_promo_item_df = l6_promo_item_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
    first("Items")
).fillna(0)

for c in pivot_columns:
      pivoted_l6_promo_item_df = pivoted_l6_promo_item_df.withColumnRenamed(c, "L6_" + c + "_ITEMS")

#------------------------------------------------------------------------------------------------------------------------------------
# L9

l9_promo_item_df = full_flag_df.filter(F.col('last_9_flag') == 'Y')\
                       .groupBy('household_id', 'promo_flag')\
                       .agg(F.sum('unit').alias('Items'))

pivot_columns = l9_promo_item_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_l9_promo_item_df = l9_promo_item_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
    first("Items")
).fillna(0)

for c in pivot_columns:
      pivoted_l9_promo_item_df = pivoted_l9_promo_item_df.withColumnRenamed(c, "L9_" + c + "_ITEMS")

# COMMAND ----------

# DBTITLE 1,Save Item Recency
pivoted_l3_promo_item_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l3_promo_item_agg_data_tmp")
pivoted_l6_promo_item_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l6_promo_item_agg_data_tmp")
pivoted_l9_promo_item_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_l9_promo_item_agg_data_tmp")

# COMMAND ----------

# DBTITLE 1,Last Weekend Promo / Markdown
last_weekend_promo_df = full_flag_df.filter(F.col('last_weekend_flag') == 'Y')\
                       .groupBy('household_id','promo_flag')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                       
last_weekend_promo_df = last_weekend_promo_df.join(total_df, on='household_id', how='inner')

last_weekend_promo_df = last_weekend_promo_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))


pivot_columns = last_weekend_promo_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
pivoted_last_weekend_promo_df = last_weekend_promo_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
    first("Spend").alias("Spend"),
    first("Visits").alias("Visits"),
    first("Units").alias("Units"),
    first("SPV").alias("SPV"),
    first("UPV").alias("UPV"),
    first("SPU").alias("SPU"),
    first("PCT_Spend").alias("PCT_Spend"),
    first("PCT_Visits").alias("PCT_Visits"),
    first("PCT_Units").alias("PCT_Units")
).fillna(0)

for c in pivot_columns:
      pivoted_last_weekend_promo_df = pivoted_last_weekend_promo_df.withColumnRenamed(c +"_Spend", "LAST_WKND_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_Visits", "LAST_WKND_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_Units", "LAST_WKND_" + c + "_UNITS")\
                                   .withColumnRenamed(c +"_SPV", "LAST_WKND_" + c + "_SPV")\
                                   .withColumnRenamed(c +"_UPV", "LAST_WKND_" + c + "_UPV")\
                                   .withColumnRenamed(c +"_SPU", "LAST_WKND_" + c + "_SPU")\
                                   .withColumnRenamed(c +"_PCT_Spend", "PCT_LAST_WKND_" + c + "_SPEND")\
                                   .withColumnRenamed(c +"_PCT_Visits", "PCT_LAST_WKND_" + c + "_VISITS")\
                                   .withColumnRenamed(c +"_PCT_Units", "PCT_LAST_WKND_" + c + "_UNITS")

# COMMAND ----------

# DBTITLE 1,Save Promo Weekend
pivoted_last_weekend_promo_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_last_weekend_promo_agg_data_tmp")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Final Join
# MAGIC

# COMMAND ----------

# full_flag_df = full_flag_df.filter(~(F.col('household_id') == -1))

# COMMAND ----------

# DBTITLE 1,Join Agg / Thick Flag / Affluence Flag / CC Tenure
truprice_df = spark.table('tdm_seg.srai_truprice_full_history')

max_period_id = 0
max_period_id_truprice = truprice_df.agg(F.max('period_id')).collect()[0][0]
max_period_id_txn = full_flag_df.agg(F.max('period_id')).collect()[0][0]

if (max_period_id_truprice >= max_period_id_txn):
  max_period_id = max_period_id_txn
else:
  max_period_id = max_period_id_truprice

truprice_df = spark.table('tdm_seg.srai_truprice_full_history').filter(F.col('period_id') == max_period_id)

#cc tenure
cc_tenure_df = full_flag_df.select('household_id', 'CC_TENURE').distinct()

# load table back 
total_df = spark.table("tdm_seg.kritawatkrai_th_year_total_agg_data_tmp")
pivoted_div_df = spark.table("tdm_seg.kritawatkrai_th_year_div_agg_data_tmp")
pivoted_l3_div_df = spark.table("tdm_seg.kritawatkrai_th_year_l3_div_agg_data_tmp")
pivoted_l6_div_df = spark.table("tdm_seg.kritawatkrai_th_year_l6_div_agg_data_tmp")
pivoted_l9_div_df = spark.table("tdm_seg.kritawatkrai_th_year_l9_div_agg_data_tmp")
pivoted_dep_df = spark.table("tdm_seg.kritawatkrai_th_year_dep_agg_data_tmp")
pivoted_l3_dep_df = spark.table("tdm_seg.kritawatkrai_th_year_l3_dep_agg_data_tmp")
pivoted_l6_dep_df = spark.table("tdm_seg.kritawatkrai_th_year_l6_dep_agg_data_tmp")
pivoted_l9_dep_df = spark.table("tdm_seg.kritawatkrai_th_year_l9_dep_agg_data_tmp")
pivoted_sec_df = spark.table("tdm_seg.kritawatkrai_th_year_sec_agg_data_tmp")
monthly_df = spark.table("tdm_seg.kritawatkrai_th_year_monthly_agg_data_tmp")
qtr1_df = spark.table("tdm_seg.kritawatkrai_th_year_qtr1_agg_data_tmp")
qtr2_df = spark.table("tdm_seg.kritawatkrai_th_year_qtr2_agg_data_tmp")
qtr3_df = spark.table("tdm_seg.kritawatkrai_th_year_qtr3_agg_data_tmp")
qtr4_df = spark.table("tdm_seg.kritawatkrai_th_year_qtr4_agg_data_tmp")
weekly_df = spark.table("tdm_seg.kritawatkrai_th_year_weekly_agg_data_tmp")
pivoted_fest_df = spark.table("tdm_seg.kritawatkrai_th_year_fest_agg_data_tmp")
count_df = spark.table("tdm_seg.kritawatkrai_th_year_count_agg_data_tmp")
pivoted_store_region_df = spark.table("tdm_seg.kritawatkrai_th_year_store_region_agg_data_tmp")
pivoted_time_day_df = spark.table("tdm_seg.kritawatkrai_th_year_time_day_agg_data_tmp")
pivoted_store_format_df = spark.table("tdm_seg.kritawatkrai_th_year_store_format_agg_data_tmp")
pivoted_price_level_df = spark.table("tdm_seg.kritawatkrai_th_year_price_level_agg_data_tmp")
last_wknd_df = spark.table("tdm_seg.kritawatkrai_th_year_last_wknd_agg_data_tmp")
wknd_df = spark.table("tdm_seg.kritawatkrai_th_year_wknd_agg_data_tmp")
wkday_df = spark.table("tdm_seg.kritawatkrai_th_year_wkday_agg_data_tmp")
l3_df = spark.table("tdm_seg.kritawatkrai_th_year_l3_agg_data_tmp")
l6_df = spark.table("tdm_seg.kritawatkrai_th_year_l6_agg_data_tmp")
l9_df = spark.table("tdm_seg.kritawatkrai_th_year_l9_agg_data_tmp")
pivoted_payment_df = spark.table("tdm_seg.kritawatkrai_th_year_payment_agg_data_tmp")
pivoted_discount_promo_df = spark.table("tdm_seg.kritawatkrai_th_year_discount_promo_agg_data_tmp")
pivoted_l3_promo_df = spark.table("tdm_seg.kritawatkrai_th_year_l3_promo_agg_data_tmp")
pivoted_l6_promo_df = spark.table("tdm_seg.kritawatkrai_th_year_l6_promo_agg_data_tmp")
pivoted_l9_promo_df = spark.table("tdm_seg.kritawatkrai_th_year_l9_promo_agg_data_tmp")
pivoted_time_promo_df = spark.table("tdm_seg.kritawatkrai_th_year_time_promo_agg_data_tmp")
pivoted_l3_promo_item_df = spark.table("tdm_seg.kritawatkrai_th_year_l3_promo_item_agg_data_tmp")
pivoted_l6_promo_item_df = spark.table("tdm_seg.kritawatkrai_th_year_l6_promo_item_agg_data_tmp")
pivoted_l9_promo_item_df = spark.table("tdm_seg.kritawatkrai_th_year_l9_promo_item_agg_data_tmp")
pivoted_last_weekend_promo_df = spark.table("tdm_seg.kritawatkrai_th_year_last_weekend_promo_agg_data_tmp")


more_agg_df = total_df.join(pivoted_div_df, on='household_id', how='left')\
                       .join(pivoted_l3_div_df, on='household_id', how='left')\
                       .join(pivoted_l6_div_df, on='household_id', how='left')\
                       .join(pivoted_l9_div_df, on='household_id', how='left')\
                       .join(pivoted_dep_df, on='household_id', how='left')\
                       .join(pivoted_l3_dep_df, on='household_id', how='left')\
                       .join(pivoted_l6_dep_df, on='household_id', how='left')\
                       .join(pivoted_l9_dep_df, on='household_id', how='left')\
                       .join(pivoted_sec_df, on='household_id', how='left')\
                       .join(weekly_df, on='household_id', how='left')\
                       .join(monthly_df, on='household_id', how='left')\
                       .join(qtr1_df, on='household_id', how='left')\
                       .join(qtr2_df, on='household_id', how='left')\
                       .join(qtr3_df, on='household_id', how='left')\
                       .join(qtr4_df, on='household_id', how='left')\
                       .join(pivoted_fest_df, on='household_id', how='left')\
                       .join(count_df, on='household_id', how='left')\
                       .join(pivoted_store_region_df, on='household_id', how='left')\
                       .join(pivoted_store_format_df, on='household_id', how='left')\
                       .join(pivoted_time_day_df, on='household_id', how='left')\
                       .join(pivoted_price_level_df, on='household_id', how='left')\
                       .join(last_wknd_df, on='household_id', how='left')\
                       .join(wknd_df, on='household_id', how='left')\
                       .join(wkday_df, on='household_id', how='left')\
                       .join(truprice_df.select('household_id','truprice_seg_desc'), on='household_id', how='left')\
                       .join(l3_df, on='household_id', how='left')\
                       .join(l6_df, on='household_id', how='left')\
                       .join(l9_df, on='household_id', how='left')\
                       .join(pivoted_payment_df, on='household_id', how='left')\
                       .join(pivoted_discount_promo_df, on='household_id', how='left')\
                       .join(pivoted_l3_promo_df, on='household_id', how='left')\
                       .join(pivoted_l6_promo_df, on='household_id', how='left')\
                       .join(pivoted_l9_promo_df, on='household_id', how='left')\
                       .join(pivoted_time_promo_df, on='household_id', how='left')\
                       .join(pivoted_l3_promo_item_df, on='household_id', how='left')\
                       .join(pivoted_l6_promo_item_df, on='household_id', how='left')\
                       .join(pivoted_l9_promo_item_df, on='household_id', how='left')\
                       .join(pivoted_last_weekend_promo_df, on='household_id', how='left')\
                       .join(cc_tenure_df, on='household_id', how='left')


more_agg_df = more_agg_df.withColumn('THICK_FLAG',F.when((F.col('Total_Spend')>2000) & (F.col('Total_Visits')>=8) & (F.col('Total_Units')>=100)\
                                                & (F.col('Q1_Spend')>0) & (F.col('Q2_Spend')>0) & (F.col('Q3_Spend')>0) & (F.col('Q4_Spend')>0), 1)\
                                         .otherwise(0))\
               .withColumn('AFFLUENCE_UM',F.when(F.col('truprice_seg_desc').isin('Most Price Insensitive', 'Price Insensitive'),F.lit(1))\
                                          .otherwise(F.lit(0)))\
               .withColumn('AFFLUENCE_LA',F.lit(0))\
               .drop('truprice_seg_desc')
               
full_agg_df = more_agg_df.withColumn('PCT_MIS_BIG_TICKET_UNITS',F.lit(0))\
               .withColumn('PCT_MIS_FOR_LATER_SPEND',F.lit(0))\
               .withColumn('PCT_MIS_FRESH_SPEND',F.lit(0))\
               .withColumn('PCT_MIS_FOR_LATER_UNITS',F.lit(0))\
               .withColumn('PCT_MIS_FOR_LATER_VISITS',F.lit(0))\
               .withColumn('PCT_MIS_MIXED_VISITS',F.lit(0))\
               .withColumn('PCT_MIS_ALC_TOB_VISITS',F.lit(0))\
               .withColumn('PCT_MIS_HOME_REFRESH_VISITS',F.lit(0))\
               .withColumn('PCT_CAT_L21_0204230_UNITS',F.lit(0))\
               .withColumn('PCT_CAT_L21_0204230_SPEND',F.lit(0))\
               .withColumn('PCT_CAT_L21_0204230_VISITS',F.lit(0))\
               .withColumnRenamed('Total_Spend', 'SPEND')\
               .withColumnRenamed('Total_Visits', 'VISITS')\
               .withColumnRenamed('Total_Units', 'UNITS')\
               .withColumn('PCT_CAT_L21_0140071_SPEND',F.col('PCT_CAT_DEP_%1_40%_SPEND'))\
               .withColumn('PCT_CAT_L21_0140071_VISITS',F.col('PCT_CAT_DEP_%1_40%_VISITS'))\
               .withColumn('PCT_CAT_L21_0140071_UNITS',F.col('PCT_CAT_DEP_%1_40%_UNITS'))

# create 0140071 using 0140 values (FROZEN)



# COMMAND ----------

full_agg_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_full_agg_data")


# COMMAND ----------

full_agg_df = spark.table("tdm_seg.kritawatkrai_th_year_full_agg_data")

# COMMAND ----------

# DBTITLE 1,Convert To Double and Round to 2dp
columns = [column for column in full_agg_df.columns if column != 'household_id']

rounded_full_agg_df = full_agg_df.select(
   F.col('household_id'),
    *[round(F.col(column).cast('double'), 2).alias(column) for column in columns]
)

# COMMAND ----------

rounded_full_agg_df = rounded_full_agg_df.fillna(0)
rounded_full_agg_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_rounded_full_agg_data")


# COMMAND ----------

t = spark.table('tdm.v_prod_dim_c').select(['upc_id','brand_name','division_id','division_name','department_id','department_name','department_code','section_id','section_name','section_code','class_id','class_name','class_code','subclass_id','subclass_name','subclass_code'])\
                                                 .filter(F.col('division_id').isin(product_division))\
                                                 .filter(F.col('country') == country)

# COMMAND ----------

t.filter(F.col('section_code') == '10_11_72').display()

# COMMAND ----------

rounded_full_agg_df.printSchema()
