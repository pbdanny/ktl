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

# MAGIC %sql
# MAGIC select distinct price_level
# MAGIC from tdm_seg.th_lotuss_thanakrit_ktl_snap_txn_add_dummy_hh_combine_prod_hier
# MAGIC where household_id != -1;

# COMMAND ----------

    price_level_df = txn.where(F.col("household_id") != -1).groupBy('household_id','price_level')\
                        .agg(F.sum('net_spend_amt').alias('Spend'), \
                        F.count_distinct('transaction_uid').alias('Visits'), \
                            F.sum('unit').alias('Units'))

# COMMAND ----------

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

# COMMAND ----------

premium_prod_df.display()

# COMMAND ----------

spark.table("tdm_seg.v_latest_txn118wk").printSchema()

# COMMAND ----------

from src.utils import conf

conf_path = "../config/feature_migration.json"

conf_mapper = conf.conf_reader(conf_path)

# COMMAND ----------

# DBTITLE 1,Load Config
decision_date = conf_mapper["data"]["decision_date"]
start_week = conf_mapper["data"]["start_week"]
end_week = conf_mapper["data"]["end_week"]
timeframe_start = conf_mapper["data"]["timeframe_start"]
timeframe_end = conf_mapper["data"]["timeframe_end"]

scope_date_dim = (spark
            .table('tdm.v_date_dim')
            .select(['date_id','period_id','quarter_id','year_id','month_id','weekday_nbr','week_id',
                    'day_in_month_nbr','day_in_year_nbr','day_num_sequence','week_num_sequence'])
            .where(F.col("week_id").between(start_week, end_week))
            .where(F.col("date_id").between(timeframe_start, timeframe_end))
                    .dropDuplicates()
            )

# COMMAND ----------

max_week_december = (scope_date_dim
                        .where((F.col("month_id") % 100) == 12)
                        .filter(F.col("week_id").startswith(F.col("month_id").substr(1, 4))) 
                        .agg(F.max(F.col("week_id")).alias("max_week_december")).collect()[0]["max_week_december"]
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct subclass_code, subclass_name, class_name
# MAGIC from tdm.v_prod_dim_c
# MAGIC where subclass_name like "%PANTS%";

# COMMAND ----------

# DBTITLE 1,Save Division
pivoted_div_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_div_agg_data_tmp")


# COMMAND ----------


