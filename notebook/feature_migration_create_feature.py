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


