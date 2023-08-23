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

from pprint import pprint
pprint(conf_mapper)

# COMMAND ----------

# DBTITLE 1,Total Spend Unit Visit
snap_txn = spark.table(conf_mapper["storage"]["hive"]["snap_txn"])
total_kpi_tbl_nm = conf_mapper["storage"]["hive"]["prefix"] + "total_kpi"
total_df = snap_txn.groupBy('household_id')\
                       .agg(F.sum('net_spend_amt').alias('Total_Spend'), \
                       F.count_distinct('unique_transaction_uid').alias('Total_Visits'), \
                        F.sum('unit').alias('Total_Units'))
total_df.write.mode("overwrite").saveAsTable("total_kpi_tbl_nm")
conf_mapper["storate"]["hive"]["total_kpi"] = total_kpi_tbl_nm
conf.conf_writer(conf_mapper, conf_path)

# COMMAND ----------

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

