from utils.logger import logger
import os
import sys
from pyspark.sql import functions as F
from pyspark.sql import Window, DataFrame
from pathlib import Path
import argparse
from functools import reduce

sys.path.append('../')


@logger
def rollup_hh_id(spark, abfss_prefix, staging_fdbck_abfss_prefix, test=False):
    '''
        rollup_hh_id
        rollup household_id from fdbck_all_hh to fdbck_hh_roll  (first call date)
        :param:
            abfss_prefix, abfss_prefix
            staging_fdbck_abfss_prefix, staging_fdbck_abfss_prefix
            test, test default False
        :return:S
            fdbck_hh_roll, fdbck_hh_roll (first call date)
    '''
    from utils import files
    storage_fdbck_abfss_prefix = os.path.join(
        abfss_prefix, "feedback", "storage")
    filename = "fdbck_all_hh.parquet" if not test else "fdbck_all_hh_test.parquet"
    fdbck_all_hh = spark.read.parquet(os.path.join(
        staging_fdbck_abfss_prefix, filename)).where(F.col("household_id").isNotNull())

    fdbck_hh_roll = (fdbck_all_hh
                     .withColumn("call_order", F.row_number().over(Window.partitionBy("household_id").orderBy(F.col("First_Call_Date").desc_nulls_last())))
                     .where(F.col("call_order") == 1)
                     .select("household_id", "Bought_Status", "Contact_Status", "Reason_code", F.col("First_Call_Date").alias("first_contact_date"))
                     )
    filename = "fdbck_hh_roll.parquet" if not test else "fdbck_hh_roll_test.parquet"
    files.save(fdbck_hh_roll, os.path.join(
        storage_fdbck_abfss_prefix, filename), format="parquet", mode="overwrite")
    return fdbck_hh_roll


@logger
def main(spark, sqlContext,  date_end, test=False):
    '''
        main function
        :param:
            spark, spark session
            date_end, date_end
            test, test

    '''
    from utils import files, etl
    from features import recency, exclusion
    conf_mapper = files.conf_reader("../config/etl.json")
    mnt_mapper = files.conf_reader("../config/mnt.json")
    abfss_prefix, dbfs_path = (
        mnt_mapper["abfss_prefix"], Path(mnt_mapper["dbfs_path"]))

    staging_fdbck_abfss_prefix = os.path.join(
        abfss_prefix, "feedback", "staging")
    insurer_list = []
    for insurer in conf_mapper:
        insurer_nm = conf_mapper[insurer]["insurer_nm"]
        date_format = conf_mapper[insurer]["date_format"]
        filename = f"Feedback_{insurer_nm}.delta" if not test else f"Feedback_{insurer_nm}_test.delta"
        insurer_path = os.path.join(
            staging_fdbck_abfss_prefix, filename)
        _tmp = spark.read.load(insurer_path)
        # .where(F.col("Bought_Status").isin(["Bought", "Not Bought"]))
        _tmp = _tmp.withColumn("Contact_Status", F.when(F.col(
            "Contact_Status") == "CONTACT", "Contactable").otherwise(F.col("Contact_Status")))
        _tmp = (_tmp
                .where(F.col("Contact_Status").isin(["Contactable", "Uncontactable"]))
                .withColumn("First_Call_Date", F.to_date("First_call_date", date_format))
                .select("dib_cust_code", "Bought_Status", "Contact_Status", "Reason_code", "First_Call_Date")
                )
        insurer_list.append(_tmp)
        del [_tmp]
    # fdbck_all = reduce(DataFrame.unionByName(
    #     allowMissingColumns=True), insurer_list)
    fdbck_all = reduce(lambda x, y: x.unionByName(
        y, allowMissingColumns=True), insurer_list)
    fdbck_all = fdbck_all.where(F.col("First_Call_Date").isNotNull())
    cust_fdbck_hh = etl.ngc_to_household(spark, fdbck_all)
    filename = "fdbck_all_hh.parquet" if not test else "fdbck_all_hh_test.parquet"
    files.save(cust_fdbck_hh, os.path.join(staging_fdbck_abfss_prefix,
               filename), format="parquet", mode="overwrite")
    fdbck_hh_roll = rollup_hh_id(
        spark, abfss_prefix, staging_fdbck_abfss_prefix, test=test)
    cust_detail_wk = recency.recency_txn(spark, sqlContext, recency_day=[30, 90, 180, 360],
                                         cust_fdbck_hh=fdbck_hh_roll, date_end=date_end)
    print(test)
    exclusion.main(spark, cust_detail_wk, test=test)
    del [cust_fdbck_hh, fdbck_hh_roll, cust_detail_wk]


if __name__ == "__main__":
    '''
        python features/ground_truth.py -d 2020-12-31 -t
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", help='''data end date
                                                ''')
    parser.add_argument("-t", "--test",  action='store_true', help='''set test data flag
                                                ''')
    args = parser.parse_args()
    test = args.test
    date_end = args.date
    # main(test=test, date_end=date_end) # for local test try to find a way to pass spark session
