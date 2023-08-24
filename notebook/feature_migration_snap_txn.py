# Databricks notebook source
import sys
import os
import logging
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
logging.basicConfig(filename='logs.log',
            filemode='w',
            level=logging.INFO)


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

conf_mapper["storage"]["hive"]["prefix"]

# COMMAND ----------

(cc_txn
 .sample(0.001)
 .write
 .mode("overwrite")
 .saveAsTable(conf_mapper["storage"]["hive"]["prefix"] + "snap_txn")
)

# COMMAND ----------

conf_mapper["storage"]["hive"]["snap_txn"] = conf_mapper["storage"]["hive"]["prefix"] + "snap_txn"
conf.conf_writer(conf_mapper, conf_path)
