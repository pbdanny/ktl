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

from src.features import prod_hierarcy_recency

# COMMAND ----------

# MAGIC %md ###Division , Division Recency l3, l6, l9

# COMMAND ----------

division_total = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "division_id", "")
division_l3 = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "division_id", "last_3_flag")
division_l6 = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "division_id", "last_6_flag")
division_l9 = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "division_id", "last_9_flag")

# COMMAND ----------

from src.utils import storage

# COMMAND ----------

storage.save_hive(division_total, conf_mapper, "feat_agg_div_total")
storage.save_hive(division_l3, conf_mapper, "feat_agg_div_l3")
storage.save_hive(division_l6, conf_mapper, "feat_agg_div_l6")
storage.save_hive(division_l9, conf_mapper, "feat_agg_div_l9")

# COMMAND ----------

# MAGIC %md ###Department, Department recency l3, l6, l9

# COMMAND ----------

dept_total = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "grouped_department_code", "")
dept_l3 = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "grouped_department_code", "last_3_flag")
dept_l6 = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "grouped_department_code", "last_6_flag")
dept_l9 = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "grouped_department_code", "last_9_flag")

# COMMAND ----------

storage.save_hive(dept_total, conf_mapper, "feat_agg_dep_total")
storage.save_hive(dept_l3, conf_mapper, "feat_agg_dep_l3")
storage.save_hive(dept_l6, conf_mapper, "feat_agg_dep_l6")
storage.save_hive(dept_l9, conf_mapper, "feat_agg_dep_l9")

# COMMAND ----------

# MAGIC %md ###Section

# COMMAND ----------

sec_total = prod_hierarcy_recency.get_agg_prd_hier_recency(spark, conf_mapper, txn, "grouped_section_code", "")

# COMMAND ----------

storage.save_hive(sec_total, conf_mapper, "feat_agg_sec_total")

# COMMAND ----------

# MAGIC %md ###Distinct product & store

# COMMAND ----------

distinct_prd_str = prod_hierarcy_recency.get_agg_distinct_prod_store(spark, conf_mapper, txn)

# COMMAND ----------

storage.save_hive(distinct_prd_str, conf_mapper, "feat_agg_distinct_prod_hier_store")

# COMMAND ----------


