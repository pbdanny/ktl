from utils.logger import logger
import os
import sys
from pyspark.sql import functions as F
from pathlib import Path
import argparse
from datetime import datetime, timedelta

sys.path.append('../')


@logger
def cash_card_recency(spark, prjct_nm, txn_cust_r360, test=False):
    '''
        main function
        :param:
            spark, spark session
            prjct_nm, prjct_nm
            df, txn_cust df
            test, test
        :description:
        (1) Cash/Card Customer
        ----
        Recency : 360 days
        Store_Format : HDE
        if customer with proportion of visit@HDE by card vs total visit@HDE >= 0.2 -> flag as card customer
    '''
    from src.utils import conf
    conf_mapper = conf.conf_reader("../config/transaction.json")
    mnt_mapper = conf.conf_reader("../config/mnt.json")
    abfss_prefix = mnt_mapper["abfss_prefix"]
    PROP_HDE_CCARD_VISIT = conf_mapper["PROP_HDE_CCARD_VISIT"]

    agg_cust_hde = \
        (txn_cust_r360
         .where(F.col("store_format_name") == "HDE")
         .groupBy("household_id")
         .pivot("tender_type_group")
         .agg(F.sum(F.col("net_spend_amt")).alias("sales_360_days"),
              F.countDistinct(F.col("transaction_uid")).alias("visits_360_days"))
         .fillna(value=0)
         )
    # use lower(tender type group) for column name -> pivot
    lower_case_column_names = [c.lower() for c in agg_cust_hde.columns]
    agg_cust_hde = agg_cust_hde.toDF(*lower_case_column_names)

    cust_gr_hde = \
        (agg_cust_hde
         .withColumn("total_visits_360_days", (F.col("cash_visits_360_days") + F.col("ccard_visits_360_days")))
         .withColumn("total_sales_360_days",  (F.col("cash_sales_360_days") + F.col("ccard_sales_360_days")))
         .withColumn("prop_visit_by_card_360_days", ((F.col("ccard_visits_360_days") / F.col("total_visits_360_days"))))
         .fillna(value=0)
         .withColumn("cash_card_customer_flag", F.when(F.col("prop_visit_by_card_360_days") >= PROP_HDE_CCARD_VISIT, "credit").otherwise("cash"))
         .select("household_id", "cash_card_customer_flag")
         )

    # Assign "cash" to the rest customer , that no transaction in HDE
    cust_gr = \
        (agg_cust_hde
         .select("household_id")
         .drop_duplicates()
         .join(cust_gr_hde, "household_id", "outer")
         .fillna("cash", subset=["cash_card_customer_flag"])
         )
    filename = "cust_hde_r360_cash_card.parquet" if not test else "cust_hde_r360_cash_card_test.parquet"
    conf.save(cust_gr, os.path.join(abfss_prefix, prjct_nm, "cust_cash_card",
               filename), format="parquet", mode="overwrite", overwriteSchema=True)
