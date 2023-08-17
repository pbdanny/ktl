# Databricks notebook source
# MAGIC %md 
# MAGIC https://drive.google.com/file/d/1BIoJBa1n4X6g23CuTghDHM925Ifu81gw/view?usp=sharing

# COMMAND ----------

# MAGIC %run /Users/thanakrit.boonquarmdee@lotuss.com/utils/std_import

# COMMAND ----------

import os
from utils import files, segmentation
from edm_class import txnItem
from pathlib import Path


# COMMAND ----------

from etl import staging

# COMMAND ----------

from features import ground_truth
from features import transaction
from features import get_txn_cust, quarter_recency, store_format, product_group, combine

# COMMAND ----------

prjct_nm = "202307_model"
test = False
params = [spark, prjct_nm, test]
test_suffix = "_test" if test else ""
conf_mapper = files.conf_reader("../config/etl.json")
mnt_mapper = files.conf_reader("../config/mnt.json")
abfss_prefix, dbfs_path = (
    mnt_mapper["abfss_prefix"], Path(mnt_mapper["dbfs_path"]))

spark.sparkContext.setCheckpointDir('dbfs:/FileStore/niti/temp/checkpoint') # must set checkpoint before passing txnItem

# COMMAND ----------

# MAGIC %md
# MAGIC ## transaction prep

# COMMAND ----------

aac_fdbck = spark.read.load("abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/feedback/staging/Feedback_AAC.delta")
cig_fdbck = spark.read.load("abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/feedback/staging/Feedback_CIG.delta")

# COMMAND ----------

aac_fdbck.groupBy("src_file").count().display()
cig_fdbck.groupBy("src_file").count().display()

# COMMAND ----------

# staging.feedback(spark, "CIGNA", abfss_prefix, dbfs_path, "feedback", "landing", conf_mapper, "1", "202301")

# COMMAND ----------

aac_fdbck.select("Contact_Status").distinct().display()
aac_fdbck.select("Contact_Status", "Bought_Status").distinct().display()

# COMMAND ----------

cig_fdbck.select("Contact_Status").distinct().display()

# COMMAND ----------

ground_truth.main(spark, sqlContext, "2023-06-18",test)

# COMMAND ----------

transaction.main(spark, dbutils, prjct_nm, txnItem, test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## feature modules

# COMMAND ----------

quarter_recency.main(*params)

# COMMAND ----------

store_format.main(*params)

# COMMAND ----------

product_group.main(*params)

# COMMAND ----------

combine.main(spark, prjct_nm, test)

# COMMAND ----------

test = spark.read.parquet("abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/feedback/storage/cust_details_seg.parquet")

# COMMAND ----------

feat = spark.read.parquet("abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/202307_model/features/all_feature.parquet")

# COMMAND ----------

from threading import Thread
if __name__ == '__main__':
#     Thread(target = store_format(*params).start()

# COMMAND ----------

# Thread(target = quarter_recency, args=params).start()


# COMMAND ----------


