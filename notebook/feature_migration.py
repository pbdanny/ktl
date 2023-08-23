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
from src.utils import files

conf_path = "../config/feature_migration.json"

conf_mapper = files.conf_reader(conf_path)

decision_date = conf_mapper["data"]["decision_date"]
gap_decision_n_month = conf_mapper["data"]["gap_decision_n_month"]
print(f"{decision_date} , with gap months from decision date to data end date {gap_decision_n_month}")

# COMMAND ----------

# decision date = latest data end date
decision_date =  datetime.strptime(decision_date, '%Y-%m-%d').date()

# timeframe_end = end of -2 months from decision date
# timeframe_start = 1 year from timeframe_end
timeframe_end = decision_date - relativedelta(months=+float(gap_decision_n_month)) - relativedelta(days=+1)
timeframe_start = (timeframe_end - relativedelta(months=+11)).replace(day=1)

print(f"decision date : {decision_date}\ntxn start date : {timeframe_start}\ntxn end date : {timeframe_end}")
print(f"gap days from decision - txn end : {(decision_date - timeframe_end).days}")
print(f"gap days from txn start - txn end : {(timeframe_end - timeframe_start).days}")

# Get week_id of time frame
date_dim = spark.table('tdm.v_date_dim').select('week_id', 'date_id', 'period_id', 'quarter_id', 'promoweek_id')

start_week = date_dim.filter(F.col('date_id').between(timeframe_start, timeframe_end)).agg(F.min('week_id')).collect()[0][0]
end_week = date_dim.filter(F.col('date_id').between(timeframe_start, timeframe_end)).agg(F.max('week_id')).collect()[0][0]

print(f"start_week : {start_week}, end_week : {end_week}")

# COMMAND ----------

# DBTITLE 1,Writeback config
conf_mapper["data"]["timeframe_end"] = timeframe_end.strftime("%Y-%m-%d")
conf_mapper["data"]["timeframe_start"] = timeframe_start.strftime("%Y-%m-%d")
conf_mapper["data"]["start_week"] = start_week
conf_mapper["data"]["end_week"] = end_week

files.conf_writer(conf_mapper, conf_path)

# COMMAND ----------

from src.utils import files

conf_path = "../config/snap_txn.json"

conf_mapper = files.conf_reader("../config/snap_txn.json")

# COMMAND ----------

from src.data import snap_txn

# COMMAND ----------

cc_txn = snap_txn.get_txn_cc_exc_trdr(spark, conf_path)

# COMMAND ----------

cc_txn_map_time = snap_txn.map_txn_time(spark, conf_path, cc_txn)

# COMMAND ----------

type(cc_txn_map_time)

# COMMAND ----------


