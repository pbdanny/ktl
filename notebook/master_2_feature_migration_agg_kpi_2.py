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

txn = spark.table(conf_mapper["storage"]["hive"]["snap_txn_add_dummy_hh_combine_prod_hier"])

# COMMAND ----------

from src.features import time_period_recency
from src.utils import storage

# COMMAND ----------

# MAGIC %md ###Monthly kpi 

# COMMAND ----------

montly_kpi = time_period_recency.get_agg_monthly(spark, conf_mapper, txn)

# COMMAND ----------

storage.save_hive(montly_kpi, conf_mapper, "feat_agg_monthly")

# COMMAND ----------

# MAGIC %md ###Quarterly Q1,Q2,Q3,Q4

# COMMAND ----------

q1_kpi = time_period_recency.get_agg_quarter(spark, conf_mapper, txn, "q1_flag")
q2_kpi = time_period_recency.get_agg_quarter(spark, conf_mapper, txn, "q2_flag")
q3_kpi = time_period_recency.get_agg_quarter(spark, conf_mapper, txn, "q3_flag")
q4_kpi = time_period_recency.get_agg_quarter(spark, conf_mapper, txn, "q4_flag")

# COMMAND ----------

storage.save_hive(q1_kpi, conf_mapper, "feat_agg_q1_kpi")
storage.save_hive(q2_kpi, conf_mapper, "feat_agg_q2_kpi")
storage.save_hive(q3_kpi, conf_mapper, "feat_agg_q3_kpi")
storage.save_hive(q4_kpi, conf_mapper, "feat_agg_q4_kpi")

# COMMAND ----------

# MAGIC %md ###Weekly KPI

# COMMAND ----------

wkly_kpi = time_period_recency.get_agg_wkly(spark, conf_mapper, txn)

# COMMAND ----------

storage.save_hive(wkly_kpi, conf_mapper, "feat_agg_wkly_kpi")

# COMMAND ----------

# MAGIC %md ###Festive

# COMMAND ----------

festive_kpi = time_period_recency.get_agg_festive(spark, conf_mapper, txn)
storage.save_hive(festive_kpi, conf_mapper, "feat_agg_festive_kpi")

# COMMAND ----------

# MAGIC %md ###Time of Day

# COMMAND ----------

time_of_day_kpi = time_period_recency.get_agg_time_of_day(spark, conf_mapper, txn)
storage.save_hive(time_of_day_kpi, conf_mapper, "feat_agg_time_of_day_kpi")

# COMMAND ----------

# MAGIC %md ###Weekend

# COMMAND ----------

wkend_kpi = time_period_recency.get_agg_wkend(spark, conf_mapper, txn)
storage.save_hive(wkend_kpi, conf_mapper, "feat_agg_wkend_kpi")

# COMMAND ----------

# MAGIC %md ###Last Weekend

# COMMAND ----------

last_wkend_kpi = time_period_recency.get_agg_last_wkend(spark, conf_mapper, txn)
storage.save_hive(last_wkend_kpi, conf_mapper, "feat_agg_last_wkend_kpi")

# COMMAND ----------

# MAGIC %md ###Weekday

# COMMAND ----------

wkday_kpi = time_period_recency.get_agg_wkday(spark, conf_mapper, txn)
storage.save_hive(wkday_kpi, conf_mapper, "feat_agg_wkday_kpi")

# COMMAND ----------

# MAGIC %md ###Recency - l3, l6, l9

# COMMAND ----------

l3_kpi = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "last_3_flag")
l6_kpi = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "last_6_flag")
l9_kpi = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "last_9_flag")

storage.save_hive(l3_kpi, conf_mapper, "feat_agg_l3_kpi")
storage.save_hive(l6_kpi, conf_mapper, "feat_agg_l6_kpi")
storage.save_hive(l9_kpi, conf_mapper, "feat_agg_l9_kpi")
