# Databricks notebook source
# MAGIC %run /Users/thanakrit.boonquarmdee@lotuss.com/utils/std_import

# COMMAND ----------

from datetime import datetime, timedelta
from edm_class import txnItem
from edm_helper import to_pandas

from functools import reduce
import os
import pyspark.sql.types as T
from pyspark.sql import functions as F
spark.sparkContext.setCheckpointDir('dbfs:/FileStore/niti/temp/checkpoint')

# COMMAND ----------

"""
Save to outbound
"""
abfss_prefix = "abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead"
prjct_nm = "edm_202305"
stage = "lead"
substage = "outbound"

outb = spark.read.parquet(os.path.join(abfss_prefix, prjct_nm, stage, substage, "all_outbound.parquet"))

# COMMAND ----------

outb.display()

# COMMAND ----------

outb.select("LifeStyle9", "CashorCreditCard").withColumn(
    "LifeStyle9",
    F.when(F.col("LifeStyle9").cast(T.DoubleType()) > 0.2, ">20%").otherwise("<20%"),
).groupBy("LifeStyle9", "CashorCreditCard").count().display()

# COMMAND ----------


