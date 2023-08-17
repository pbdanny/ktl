from utils.logger import logger
import os
import sys

sys.path.append('../')

@logger
def get_txn_cust(spark, prjct_nm, test=False):
    from utils import files
    mnt_mapper = files.conf_reader("../config/mnt.json")
    abfss_prefix = mnt_mapper["abfss_prefix"]
    filename = "cust_details_seg.parquet" if not test else "cust_details_seg_test.parquet"
    cust_recency = \
        (spark.read.parquet(os.path.join(abfss_prefix, prjct_nm, "feedback", filename))
         .select("household_id", "first_contact_date", "r360_start_date", "r180_start_date", "r90_start_date", "r30_start_date")
         .drop_duplicates()
         )
    filename = "txn_cc_sngl_tndr.parquet" if not test else "txn_cc_sngl_tndr_test.parquet"
    snap_txn = spark.read.parquet(os.path.join(
        abfss_prefix, prjct_nm, filename))

    txn_cust = snap_txn.join(cust_recency, "household_id", "inner")
    return txn_cust