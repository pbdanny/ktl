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

from src.utils import storage
from src.features import store_type

# COMMAND ----------

# MAGIC %md ###By store format

# COMMAND ----------

store_fmt_kpi = store_type.get_agg_store_format(spark, conf_mapper, txn)

# COMMAND ----------

storage.save_hive(store_fmt_kpi, conf_mapper, "feat_agg_store_format")

# COMMAND ----------

# MAGIC %md ###By store region

# COMMAND ----------

store_region_kpi = store_type.get_agg_store_region(spark, conf_mapper, txn)

# COMMAND ----------

storage.save_hive(store_region_kpi, conf_mapper, "feat_agg_store_region")

# COMMAND ----------

# MAGIC %md ###Premium / Budget prod

# COMMAND ----------

from src.features import prod_price

prod_price_kpi = prod_price.get_agg_prod_price(spark, conf_mapper, txn)

# COMMAND ----------

storage.save_hive(prod_price_kpi, conf_mapper, "feat_agg_prod_price")

# COMMAND ----------

# MAGIC %md ###Tender

# COMMAND ----------

from src.features import tender

tender_kpi = tender.get_agg_tender(spark, conf_mapper, txn)

# COMMAND ----------

storage.save_hive(tender_kpi, conf_mapper, "feat_agg_tender")

# COMMAND ----------

# MAGIC %md ###Promo Total + Promo Recency

# COMMAND ----------

from src.features import promo_recency

promo_total_kpi = promo_recency.get_agg_promo(spark, conf_mapper, txn)

storage.save_hive(promo_total_kpi, conf_mapper, "feat_agg_promo_total")

# COMMAND ----------

from src.features import promo_recency

promo_l3 = promo_recency.get_agg_promo_recency(spark, conf_mapper, txn, "last_3_flag")
promo_l6 = promo_recency.get_agg_promo_recency(spark, conf_mapper, txn, "last_6_flag")
promo_l9 = promo_recency.get_agg_promo_recency(spark, conf_mapper, txn, "last_9_flag")

storage.save_hive(promo_l3, conf_mapper, "feat_agg_promo_l3")
storage.save_hive(promo_l6, conf_mapper, "feat_agg_promo_l6")
storage.save_hive(promo_l9, conf_mapper, "feat_agg_promo_l9")

# COMMAND ----------

promo_time_of_day = promo_recency.get_agg_promo_time_of_day(spark, conf_mapper, txn)
storage.save_hive(promo_time_of_day, conf_mapper, "feat_agg_promo_time_of_day")

# COMMAND ----------

promo_item_l3 = promo_recency.get_agg_promo_item_recency(spark, conf_mapper, txn, "last_3_flag")
promo_item_l6 = promo_recency.get_agg_promo_item_recency(spark, conf_mapper, txn, "last_6_flag")
promo_item_l9 = promo_recency.get_agg_promo_item_recency(spark, conf_mapper, txn, "last_9_flag")

storage.save_hive(promo_item_l3, conf_mapper, "feat_agg_promo_item_l3")
storage.save_hive(promo_item_l6, conf_mapper, "feat_agg_promo_item_l6")
storage.save_hive(promo_item_l9, conf_mapper, "feat_agg_promo_item_l9")

# COMMAND ----------

promo_last_wkend = promo_recency.get_agg_promo_last_wkend(spark, conf_mapper, txn)
storage.save_hive(promo_last_wkend, conf_mapper, "feat_agg_promo_last_wkend")

# COMMAND ----------


