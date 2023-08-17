# Databricks notebook source
# MAGIC %run /Users/thanakrit.boonquarmdee@lotuss.com/utils/std_import

# COMMAND ----------

import os
from utils import files, etl
from utils import logger
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql import types as T
from edm_class import txnItem

# COMMAND ----------

spark.sparkContext.setCheckpointDir('dbfs:/FileStore/niti/temp/checkpoint') # must set checkpoint before passing txnItem

# COMMAND ----------

from features import quarter_recency, store_format, get_txn_cust

# COMMAND ----------

txn_cust = get_txn_cust(spark, "test", True)

# COMMAND ----------

spark.table('tdm.v_resa_group_resa_tran_tender').select("country").distinct().display()

# COMMAND ----------

quarter_recency.main(spark, "test", True)

# COMMAND ----------

store_format.main(spark, "test", True)

# COMMAND ----------


