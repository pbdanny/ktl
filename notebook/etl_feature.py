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

conf_path = "../config/snap_txn.json"

conf_mapper = files.conf_reader("../config/snap_txn.json")
decision_date = conf_mapper["decision_date"]
print(f"{decision_date}")

# COMMAND ----------

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

# COMMAND ----------

# DBTITLE 1,Writeback config
conf_mapper["timeframe_end"] = timeframe_end.strftime("%Y-%m-%d")
conf_mapper["timeframe_start"] = timeframe_start.strftime("%Y-%m-%d")
conf_mapper["start_week"] = start_week
conf_mapper["end_week"] = end_week

files.conf_writer(conf_mapper, conf_path)

# COMMAND ----------

from src.utils import files

conf_path = "../config/snap_txn.json"

conf_mapper = files.conf_reader("../config/snap_txn.json")

# COMMAND ----------

from src.etl import snap_txn

# COMMAND ----------

cc_txn = snap_txn.get_txn_cc_exc_trdr(spark, conf_path)

# COMMAND ----------

cc_txn_map_time = snap_txn.map_txn_time(spark, conf_path, cc_txn)

# COMMAND ----------

type(cc_txn_map_time)

# COMMAND ----------


