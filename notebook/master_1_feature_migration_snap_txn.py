# Databricks notebook source
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

# DBTITLE 1,Load Config
from src.utils import conf

conf_path = "../config/feature_migration.json"

conf_mapper = conf.conf_reader(conf_path)

# COMMAND ----------

# decision date = latest data end date
decision_date =  datetime.strptime(conf_mapper["data"]["decision_date"], '%Y-%m-%d').date()
gap_decision_n_month = float(conf_mapper["data"]["gap_decision_n_month"])

# timeframe_end = end of months-2 from decision date
# timeframe_start = 1 year from timeframe_end
timeframe_end = decision_date - relativedelta(months=+float(gap_decision_n_month)) - relativedelta(days=+1)
timeframe_start = (timeframe_end - relativedelta(months=+11)).replace(day=1)

print(f"decision date : {decision_date}\ntxn start date : {timeframe_start}\ntxn end date : {timeframe_end}")
print(f"with gap months from decision date to data end date {gap_decision_n_month}")
print(f"gap days from decision - txn end : {(decision_date - timeframe_end).days}")
print(f"gap days from txn start - txn end : {(timeframe_end - timeframe_start).days}")

# Get week_id of time frame
date_dim = spark.table('tdm.v_date_dim').select('week_id', 'date_id')

start_week = date_dim.filter(F.col('date_id').between(timeframe_start, timeframe_end)).agg(F.min('week_id')).collect()[0][0]
end_week = date_dim.filter(F.col('date_id').between(timeframe_start, timeframe_end)).agg(F.max('week_id')).collect()[0][0]

print(f"start_week : {start_week}, end_week : {end_week}")

# COMMAND ----------

# DBTITLE 1,Writeback config
conf_mapper["data"]["timeframe_end"] = timeframe_end.strftime("%Y-%m-%d")
conf_mapper["data"]["timeframe_start"] = timeframe_start.strftime("%Y-%m-%d")
conf_mapper["data"]["start_week"] = start_week
conf_mapper["data"]["end_week"] = end_week

conf.conf_writer(conf_mapper, conf_path)

# COMMAND ----------

# DBTITLE 1,Snap Txn
from src.utils import conf

conf_path = "../config/feature_migration.json"

conf_mapper = conf.conf_reader(conf_path)

# COMMAND ----------

from src.data import snap_txn
cc_txn = snap_txn.get_txn_cc_exc_trdr(spark, conf_path)

# COMMAND ----------

from src.utils.storage import save_hive

# COMMAND ----------

save_hive(cc_txn.sample(0.001), conf_mapper, "snap_txn")

# COMMAND ----------

# MAGIC %md ##Stamp time of day, day of month

# COMMAND ----------

from src.utils import conf
conf_path = "../config/feature_migration.json"
conf_mapper = conf.conf_reader(conf_path)

txn = spark.table(conf_mapper["storage"]["hive"]["prefix"] + "snap_txn")

# COMMAND ----------

from src.data import snap_txn
from src.utils.storage import save_hive

map_time = snap_txn.map_txn_time(spark, conf_mapper, txn)
map_prd = snap_txn.map_txn_prod_premium(spark, conf_mapper, map_time)
map_tndr = snap_txn.map_txn_tender(spark, conf_mapper, map_prd)
map_cust = snap_txn.map_txn_cust_issue_first_txn(spark, conf_mapper, map_tndr)

save_hive(map_time, conf_mapper, "snap_txn_map_details")

# COMMAND ----------

conf_mapper["storage"]["hive"]["snap_txn_map_details"] = conf_mapper["storage"]["hive"]["prefix"] + "snap_txn_map_details"
conf.conf_writer(conf_mapper, conf_path)

# COMMAND ----------

# MAGIC %md ##Map promo / markdown

# COMMAND ----------

from src.utils import conf
conf_path = "../config/feature_migration.json"
conf_mapper = conf.conf_reader(conf_path)

# COMMAND ----------

from src.data import snap_txn
from src.utils.storage import save_hive

txn = spark.table(conf_mapper["storage"]["hive"]["snap_txn_map_details"])

map_promo = snap_txn.map_txn_promo(spark, conf_mapper, txn)

# COMMAND ----------

save_hive(map_promo, conf_mapper, "snap_txn_map_promo")

# COMMAND ----------

conf_mapper["storage"]["hive"]["snap_txn_map_promo"] = conf_mapper["storage"]["hive"]["prefix"] + "snap_txn_map_promo"
conf.conf_writer(conf_mapper, conf_path)

# COMMAND ----------

# MAGIC %md ##Add dummpy hh & create combined product hierarchy

# COMMAND ----------

from src.utils import conf
conf_path = "../config/feature_migration.json"
conf_mapper = conf.conf_reader(conf_path)

# COMMAND ----------

conf_mapper

# COMMAND ----------

txn = spark.table(conf_mapper["storage"]["hive"]["snap_txn_map_promo"])

# COMMAND ----------

from src.data import snap_txn
from src.utils.storage import save_hive

dummy_hh = snap_txn.create_dummy_hh(spark, conf_mapper)
txn_w_dummy = txn.unionByName(dummy_hh, allowMissingColumns=True)
txn_comb_prod_heir = snap_txn.create_combined_prod_hier(spark, conf_mapper, txn_w_dummy)

# COMMAND ----------

save_hive(txn_comb_prod_heir, conf_mapper, "snap_txn_add_dummy_hh_combine_prod_hier")

# COMMAND ----------

conf_mapper["storage"]["hive"]["snap_txn_add_dummy_hh_combine_prod_hier"] = conf_mapper["storage"]["hive"]["prefix"] + "snap_txn_add_dummy_hh_combine_prod_hier"
conf.conf_writer(conf_mapper, conf_path)

# COMMAND ----------

conf_mapper

# COMMAND ----------


