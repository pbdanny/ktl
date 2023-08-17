# Databricks notebook source
from utils import files

# COMMAND ----------

conf = files.conf_reader("../config/etl.json")

# COMMAND ----------

conf

# COMMAND ----------

conf.update({"code_version":"2023-07-22"})

# COMMAND ----------

conf

# COMMAND ----------

files.conf_writer(conf, "../config/etl.json")

# COMMAND ----------


