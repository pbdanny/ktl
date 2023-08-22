from utils.logger import logger
import os
import sys
from pyspark.sql import functions as F
from pathlib import Path
import argparse
from datetime import datetime, timedelta

sys.path.append('../')


@logger
def average_month(spark, prjct_nm, txn_cust_r360, test=False):
    """
    :description:
    (3) Proportion Discount sales, Online flag, Avg monthly sales
    ----
    Recency : 360 days
    Store_Format : allFmt

    - Proportion discount.  
    - Ever have online transaction.  
    - Avg monthly sales.  
    """
    from utils import files
    mnt_mapper = files.conf_reader("../config/mnt.json")
    abfss_prefix = mnt_mapper["abfss_prefix"]

    date_period = spark.table("tdm.v_date_dim").select(
        "date_id", "period_id").drop_duplicates()

    cust_promo_online_monthly = \
        (txn_cust_r360
         .join(date_period, "date_id", "inner")

         .groupBy("household_id")

         .agg((F.sum("discount_amt") / F.sum("net_spend_amt")).alias("prop_promo_sales"),
              F.sum(F.when(F.col("offline_online_other_channel") == "ONLINE", F.col(
                  "net_spend_amt")).otherwise(0)).alias('sales_online'),
              (F.sum("net_spend_amt") / F.count_distinct("period_id")).alias("avg_monthly_sales"))
         .withColumn("ever_online", F.when(F.col("sales_online") > 0, F.lit("Y")).otherwise(F.lit("N")))
         )
    filename = "cust_allfmt_r360_promo_online_avg_mth_sales.parquet" if not test else "cust_allfmt_r360_promo_online_avg_mth_sales_test.parquet"
    files.save(cust_promo_online_monthly, os.path.join(abfss_prefix, prjct_nm, "features",
               filename), format="parquet", mode="overwrite", overwriteSchema=True)
