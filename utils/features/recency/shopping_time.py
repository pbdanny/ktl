from utils.logger import logger
import os
import sys
from pyspark.sql import functions as F
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from pyspark.sql import Window

sys.path.append('../')


@logger
def get_prop_sales_visit_pv(sf, cust_tt, col_nm):
    """
    Calculate prop of sales / prop of visits of col_nm and pivot
    """
    out = (sf
           .groupBy("household_id", col_nm)
           .agg(F.sum(F.col("net_spend_amt")).alias("sales_360_days"),
                F.count_distinct(F.col("transaction_uid")).alias("visits_360_days"))
           .join(cust_tt, "household_id")
           .withColumn("prop_visits_360_days", F.col("visits_360_days")/F.col("total_visits_360_days"))
           .withColumn("prop_sales_360_days", F.col("sales_360_days")/F.col("total_sales_360_days"))
           .groupBy("household_id")
           .pivot(col_nm)
           .agg(F.first("sales_360_days").alias("sales_360_days"),
                F.first("visits_360_days").alias("visits_360_days"),
                F.first("prop_visits_360_days").alias("prop_visits_360_days"),
                F.first("prop_sales_360_days").alias("prop_sales_360_days"))
           .fillna(0)
           )
    return out


@logger
def shopping_time(spark, prjct_nm, txn_cust_r360, test=False):
    """
    :description:
    (4) Time of day, Day of week
    ----
    Recency : 360 days
    Store_Format : allFmt

    - % Visit by time of day.  
    - % Visist by day of week.  
    """
    from utils import files
    mnt_mapper = files.conf_reader("../config/mnt.json")
    abfss_prefix = mnt_mapper["abfss_prefix"]
    # Total visits
    cust_tt = \
        (txn_cust_r360
         .groupBy("household_id")
         .agg(F.count_distinct("transaction_uid").alias("total_visits_360_days"),
              F.sum("net_spend_amt").alias("total_sales_360_days"))
         )

    # Flag day part, weekend - weekday
    cust_time_part = \
        (txn_cust_r360
         .withColumn("hour_of_day", F.hour(F.col("tran_datetime")))
         .withColumn('part_of_day',
                     F.when(F.col('hour_of_day').isin(
                         [0, 1, 2, 3, 4, 5, 6, 7, 8]), 'early_morning')
                     .when(F.col('hour_of_day').isin([9, 10]), 'morning')
                     .when(F.col('hour_of_day').isin([11, 12]), 'lunch')
                     .when(F.col('hour_of_day').isin([13, 14, 15, 16]), 'afterlunch')
                     .when(F.col('hour_of_day').isin([17, 18]), 'evening')
                     .when(F.col('hour_of_day').isin([19, 20]), 'late_evening')
                     .when(F.col('hour_of_day').isin([21, 22, 23]), 'night')
                     .otherwise(F.lit('missing')))

         .withColumn("day_section",
                     F.when(F.col("part_of_day").isin(
                         ["early_morning", "morning", "lunch"]), "Morning_6AM_12PM")
                     .when(F.col("part_of_day").isin(["afterlunch"]), "Afternoon_12PM_4PM")
                     .when(F.col("part_of_day").isin(["evening", "late_evening"]), "Evening_4PM_8PM")
                     .when(F.col("part_of_day").isin(["night"]), "Night_8PM_12AM")
                     .otherwise(F.lit("NA")))

         .withColumn("name_of_day", F.date_format("tran_datetime", 'EEE'))

         .withColumn("wk_end_wk_day",
                     F.when(F.col("name_of_day").isin(
                         ["Mon", "Tue", "Wed", "Thu", "Fri"]), "Weekday")
                     .when(F.col("name_of_day").isin(["Sat", "Sun"]), "Weekend")
                     .otherwise("NA"))
         .withColumn("wken_day", F.concat("wk_end_wk_day", "day_section"))
         )
    cust_part_of_day_pv = get_prop_sales_visit_pv(
        cust_time_part, cust_tt, "part_of_day")

    cust_day_section_pv = get_prop_sales_visit_pv(
        cust_time_part, cust_tt, "day_section")

    cust_wk_pv = get_prop_sales_visit_pv(
        cust_time_part, cust_tt, "wk_end_wk_day")

    cust_wk_day_pv = get_prop_sales_visit_pv(
        cust_time_part, cust_tt, "wken_day")

    cust_wk_day_flag = \
        (cust_time_part
         .groupBy("household_id", "wken_day")
         .agg(F.count_distinct(F.col("transaction_uid")).alias("visits"),
              F.sum(F.col("net_spend_amt")).alias("sales"))
         .withColumn("row_nm_visit_sales", F.row_number().over(Window.partitionBy("household_id").orderBy(F.col("visits").desc(), F.col("sales").desc())))
         .where(F.col("row_nm_visit_sales") == 1)
         .select("household_id", F.col("wken_day").alias("shopping_time"))
         .drop_duplicates()
         )
    pct = \
        (cust_tt
         .join(cust_part_of_day_pv, "household_id", "left")
         .join(cust_day_section_pv, "household_id", "left")
         .join(cust_wk_pv, "household_id", "left")
         .join(cust_wk_day_pv, "household_id", "left")
         .join(cust_wk_day_flag, "household_id", "left")
         .fillna(0)
         )

    filename = "cust_allfmt_r360_shopping_time.parquet" if not test else "cust_allfmt_r360_shopping_time_test.parquet"
    files.save(pct, os.path.join(abfss_prefix, prjct_nm, "features",
               filename), format="parquet", mode="overwrite", overwriteSchema=True)
