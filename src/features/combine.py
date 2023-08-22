from utils.logger import logger
from pyspark.sql import functions as F
from datetime import datetime, timedelta
import sys
import os
from functools import reduce


@logger
def outer_join_hh(left, right):
    return left.join(right, "household_id", "outer").checkpoint()


@logger
def combine_features(spark, prjct_nm, test):
    """
    Load Model features
    ----
    """
    from utils import files
    mnt_mapper = files.conf_reader("../config/mnt.json")
    feature_mapper = files.conf_reader("../config/features.json")
    abfss_prefix = mnt_mapper["abfss_prefix"]
    test_suffix = "_test" if test else ""
    feats = {}
    for feat in feature_mapper:
        name = feature_mapper[feat]["name"]
        sub_nm = feature_mapper[feat]["sub_nm"]
        print(f">>> feat: {feat} from {sub_nm}/{name}")
        feats[feat] = spark.read.parquet(os.path.join(
            abfss_prefix, prjct_nm, sub_nm, f"{name}{test_suffix}.parquet"))
    return feats


@logger
def aggregate_features(prjct_nm, test, feats):
    '''
    Aggregate features
    ----
    convert features to aggregated features by household_id and store_id (if any)
    params:
        feats: dict of features

    '''
    from utils import files
    from functools import reduce

    mnt_mapper = files.conf_reader("../config/mnt.json")
    feature_mapper = files.conf_reader("../config/features.json")
    abfss_prefix = mnt_mapper["abfss_prefix"]
    test_suffix = "_test" if test else ""
    '''
    Cust details features
    ----
    > Remove Reason code `U` = Uncontact
    > Keep Null Reason Code with Bought status
    '''
    cust_details_contacted = \
        (feats["cust_details"].withColumn("Reason_code", F.coalesce(F.col("Reason_code"), F.col("Bought_Status")))
         .where(~F.col("Reason_code").isin(["U"]))
         )

    cust_details_feat = cust_details_contacted.select(
        feature_mapper["cust_details"]["columns"])
    # list all features in feature_mapper with sub_nm = "features" and aggregate them
    feats_agg_dict = {k: v for k, v in feature_mapper.items()
                      if v["sub_nm"] == "features"}
    feats_agg = {}
    for feat in feats_agg_dict:
        name = feature_mapper[feat]["name"]
        sub_nm = feature_mapper[feat]["sub_nm"]
        columns = feats_agg_dict[feat]["columns"]
        print(f">>> feat: {feat} from {sub_nm}/{name}")
        feats[feat].printSchema()
        feats_agg[feat] = feats[feat].select(columns)

    sfs = [feats["cust_cash_card"], *list(feats_agg.values())]

    all_feat = reduce(outer_join_hh, sfs)

    # Filter only customer was contacted
    all_feat_cust_details = all_feat.join(
        cust_details_feat, "household_id", "inner")

    files.save(all_feat_cust_details, os.path.join(abfss_prefix, prjct_nm, "features", f"all_feature{test_suffix}.parquet"
                                                   ), format="parquet", mode="overwrite", overwriteSchema=True)


def main(spark, prjct_nm, test=False):
    """
    main() function to run the script
    :param: spark spark session
    :param: prjct_nm project name, e.g. edm_202210
    Load customer segment details
    Load snap txn : only MyLo, only customer in feedback
    53weeks back, filter only single tender
    """
    feats = combine_features(spark, prjct_nm, test)
    aggregate_features(prjct_nm, test, feats)
