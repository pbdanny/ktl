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

# DBTITLE 1,Save Division
pivoted_div_df.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_year_div_agg_data_tmp")


# COMMAND ----------


