class Data():
    def __init__(self, spark, prjct_nm, test) -> None:
        from . import files
        self.spark = spark
        self.mnt_mapper = files.conf_reader("../config/mnt.json")
        self.feature_mapper = files.conf_reader("../config/features.json")
        self.abfss_prefix = self.mnt_mapper["abfss_prefix"]
        self.prjct_nm = prjct_nm
        self.test_suffix = "_test" if test else ""
        self.feats = {}
        self.all_feat = None
        self.all_feat = self.load()

    def load(self):
        import os
        from features import combine
        self.feats = combine.combine_features(
            self.spark, self.prjct_nm, self.test_suffix)
        combine.aggregate_features(self.prjct_nm, self.is_test(), self.feats)
        self.all_feat = self.spark.read.parquet(os.path.join(
            self.abfss_prefix, self.prjct_nm, "features", f"all_feature{self.test_suffix}.parquet"))
        return self.all_feat

    def get(self, cash_card_customer_flag=None):
        from pyspark.sql import functions as F
        if cash_card_customer_flag is None:
            return self.all_feat
        else:
            return self.all_feat.where(F.col("cash_card_customer_flag") == cash_card_customer_flag)

    def get_features(self, feat):
        return self.feats[feat]

    def is_test(self):
        return self.test_suffix == "_test"

    def set_abfss_prefix(self, abfss_prefix):
        self.abfss_prefix = abfss_prefix

    def set_project_name(self, prjct_nm):
        self.prjct_nm = prjct_nm

    def set_test_suffix(self, test):
        self.test_suffix = "_test" if test else ""

    def to_pandas(self, cash_card_customer_flag=None):
        from pyspark.sql import types as T
        from pyspark.sql import functions as F
        df = self.get(cash_card_customer_flag)
        col_name_type = df.dtypes
        select_cast_exp = [F.col(c[0]).cast(T.DoubleType()) if c[1].startswith(
            'decimal') else F.col(c[0]) for c in col_name_type]
        df = df.select(*select_cast_exp)

        return df.toPandas()
