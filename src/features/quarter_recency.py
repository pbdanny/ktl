from utils.logger import logger
import os
import sys
from pyspark.sql import functions as F
from pathlib import Path
import argparse
from datetime import datetime, timedelta

sys.path.append('../')


@logger
def main(spark, prjct_nm, test=False):
    """
        main() function to run the script
        :param: spark spark session
        :param: prjct_nm project name, e.g. edm_202210
    Load customer segment details
    Load snap txn : only MyLo, only customer in feedback 
    53weeks back, filter only single tender 
    """
    from . import recency
    from . import get_txn_cust
    txn_cust = get_txn_cust(spark, prjct_nm, test)
    txn_cust_r360 = txn_cust.where(F.col("date_id").between(
        F.col("r360_start_date"), F.col("first_contact_date")))
    recency.cash_card_recency(spark, prjct_nm, txn_cust_r360, test)
    recency.average_month(spark, prjct_nm, txn_cust_r360, test)
    recency.shopping_time(spark, prjct_nm, txn_cust_r360, test)
    recency.tender_kpi(spark, prjct_nm, txn_cust_r360, test)
    recency.all_kpi(spark, prjct_nm, txn_cust_r360, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # main() arguments parser later used in main()
