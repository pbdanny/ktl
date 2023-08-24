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


