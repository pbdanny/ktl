# Databricks notebook source
from utils import files
from utils import etl
from etl import staging

import pandas as pd
import pyspark.sql.functions as F

import unittest


# COMMAND ----------

'''
  Testing module
'''
from evaluation import contactable

# COMMAND ----------

conf_mapper = files.conf_reader("../config/etl.json")
mnt_mapper = files.conf_reader("../config/mnt.json")
abfss_prefix, dbfs_path = (mnt_mapper["abfss_prefix"], mnt_mapper["dbfs_path"])

# COMMAND ----------

def gen(spark, dbfs_path, insurer, conf_mapper, fdbk_type):
    from utils import etl
    fdbk_df, filename = etl.get_df(
        dbfs_path, insurer, conf_mapper, fdbk_type=fdbk_type)
    # cust_fdbck_hh = etl.ngc_to_household(
    #     spark, spark.createDataFrame(fdbk_df, schema=None))
    return spark.createDataFrame(fdbk_df, schema=None)

# COMMAND ----------

# use latest delta instead
insurer = "AZAY"
feedback1_azay = gen(spark, dbfs_path, insurer, conf_mapper, 1)
feedback2_azay = gen(spark, dbfs_path, insurer, conf_mapper, 2)

# COMMAND ----------

insurer = "CIGNA"
feedback1_cigna = gen(spark, dbfs_path, insurer, conf_mapper, 1)
feedback2_cigna = gen(spark, dbfs_path, insurer, conf_mapper, 2)

# COMMAND ----------

len_f1_a = feedback1_azay.select("src_file").distinct().count()
len_f2_a = feedback2_azay.select("src_file").distinct().count()
len_f1_c = feedback1_cigna.select("src_file").distinct().count()
len_f2_c = feedback2_cigna.select("src_file").distinct().count()

# COMMAND ----------

latest_month = {
  "aac": {
    "max":11,
    "len":0
  },
  "cig": {
    "max":12,
    "len":0
  }
} 
min_year = 2020
max_year = 2022
for insurer in ["aac","cig"]:
  for year in range(min_year,max_year+1):
    bound = range(1,13)
    if year == 2020:
      bound = range(10,13)
    elif year == 2022:
      bound = range(1,latest_month[insurer]["max"]+1)
    for month in bound:
      latest_month[insurer]["len"]+=1

# COMMAND ----------

assert len_f1_a == len_f2_a == latest_month["aac"]["len"] , "test src file len"
assert len_f1_c == len_f2_c == latest_month["cig"]["len"] , "test src file len"

# COMMAND ----------

mapper_path = "/dbfs/FileStore/niti/LMP/Lead_mapper.csv"
lead_map = pd.read_csv(mapper_path)
lead_map = spark.createDataFrame(lead_map)

# COMMAND ----------

lead_map.display()

# COMMAND ----------

# concat lot for azay
feedback1_azay = feedback1_azay.withColumn("Filename", F.split("src_file",".txt")[0])
feedback1_azay = feedback1_azay.withColumn("lot", F.split("Filename","_")[4])
feedback2_azay = feedback2_azay.withColumn("Filename", F.split("src_file",".txt")[0])
feedback2_azay = feedback2_azay.withColumn("lot", F.split("Filename","_")[4])

# COMMAND ----------

# concat lot for lot
feedback1_cigna = feedback1_cigna.withColumn("Filename", F.split("src_file",".txt")[0])
feedback1_cigna = feedback1_cigna.withColumn("lot", F.split("Filename","_")[4])
feedback2_cigna = feedback2_cigna.withColumn("Filename", F.split("src_file",".txt")[0])
feedback2_cigna = feedback2_cigna.withColumn("lot", F.split("Filename","_")[4])

# COMMAND ----------

feedback1_azay = feedback1_azay.filter(F.col("lot").isin(["202209","202210","202211"]))
feedback2_azay = feedback2_azay.filter(F.col("lot").isin(["202209","202210","202211"]))

# COMMAND ----------

feedback1_cigna.filter(F.col("DHB_Model_Date") == "2022/07/31" ).select("lot").distinct().display()

# COMMAND ----------

feedback1_cigna = feedback1_cigna.filter(F.col("lot").isin(["202209","202210","202211"]))
feedback2_cigna = feedback2_cigna.filter(F.col("lot").isin(["202209","202210","202211"]))

# COMMAND ----------

print(feedback1_azay.count())
print(feedback2_azay.count())

# COMMAND ----------

print(feedback1_cigna.count())
print(feedback2_cigna.count())

# COMMAND ----------

feedback1_azay.display()
feedback2_azay.display()

# COMMAND ----------

feedback_azay = feedback1_azay.select(
    "dib_cust_code",
    "lead_type",
    "Contact_Status",
    "Bought_Status",
    "lead_shop_payment_type",
    "lot",
).join(
    feedback2_azay,
    ["dib_cust_code", "lot"],
    "left",
)
feedback_azay.filter(F.col("Payment_Type").isNull()).display()

# COMMAND ----------

feedback_azay.count()

# COMMAND ----------

feedback_cigna = feedback1_cigna.select(
    "dib_cust_code",
    "lead_type",
    "Contact_Status",
    "Bought_Status",
    "lead_shop_payment_type",
    "lot",
).distinct().join(feedback2_cigna.distinct(), ["dib_cust_code", "lot"], "left")
feedback_cigna.filter(F.col("Payment_Type").isNull()).display()

# COMMAND ----------

feedback_cigna.count()

# COMMAND ----------

feedback1_cigna.display()
feedback2_cigna.display()

# COMMAND ----------

feedback1_cigna.filter(F.col("lead_type") == "Normal").groupby("dib_cust_code").count().filter(F.col("count")>1).count()

# COMMAND ----------

feedback_azay.filter(F.col("lead_type") != "Pom").groupby("dib_cust_code","Bought_Status", "Payment_Type").count().filter(F.col("count")>1).display()

# COMMAND ----------

feedback_cigna.filter(F.col("lead_type") != "Pom").groupby("dib_cust_code","Bought_Status", "Payment_Type").count().filter(F.col("count")>1).display()

# COMMAND ----------

feedback1_cigna.filter(F.col("dib_cust_code") == "102111060001325866").display()
feedback_cigna.filter(F.col("dib_cust_code") == "102111060001325866").display()

# COMMAND ----------



# COMMAND ----------

feedback_azay.display()

# COMMAND ----------

# qucik check
feedback_azay.filter(F.col("lead_type") != "Pom").groupby("Bought_Status", "Payment_Type").count().display()
feedback_cigna.filter(F.col("lead_type") != "Pom").groupby("Bought_Status", "Payment_Type").count().display()

# COMMAND ----------

feedback_cigna.filter((F.col("lead_type") != "Pom")&(F.col("Bought_Status").isNull())&(F.col("Payment_Type") != "Credit Card")).display()

# COMMAND ----------

feedback_azay.groupby("dib_cust_code").count().filter("count > 1").count()

# COMMAND ----------

feedback_azay.filter(F.col("Payment_Type").isNull()).groupby("dib_cust_code").count().filter("count > 1").count()

# COMMAND ----------

feedback_azay.filter(F.col("Payment_Type").isNull()).groupby("dib_cust_code").count().filter("count > 1").display()

# COMMAND ----------

feedback_azay.groupby("dib_cust_code").count().filter("count > 1").join(feedback_azay.filter(F.col("Payment_Type").isNull()).groupby("dib_cust_code").count().filter("count > 1"), "dib_cust_code", "inner" ).count()

# COMMAND ----------

feedback_azay.join(feedback_azay.filter(F.col("Payment_Type").isNull()).groupby("dib_cust_code").count().filter("count > 1"), "dib_cust_code", "inner" ).display()

# COMMAND ----------

# all customer have one premium?
feedback_azay.filter((F.col("Payment_Type").isNull())&(F.col("lead_type") != "Pom")).groupby("dib_cust_code").count().filter("count > 1").count()

# COMMAND ----------

feedback_azay.count()

# COMMAND ----------

feedback_azay.filter(F.col("Payment_Type").isNotNull()).count()

# COMMAND ----------

feedback_azay.filter(F.col("Payment_Type").isNull()).count()

# COMMAND ----------

feedback_azay.groupby("Bought_Status").count().display()

# COMMAND ----------

feedback_azay.groupby("lead_shop_payment_type", "Payment_Type").count().display()

# COMMAND ----------

feedback_azay.filter(F.col("lead_type") != "Pom").groupby("Bought_Status", "Payment_Type").count().display()

# COMMAND ----------

feedback_azay.filter(F.col("lead_type") != "Pom").groupby("Contact_status","Bought_Status", "Payment_Type").count().display()

# COMMAND ----------

feedback_cigna.filter(F.col("lead_type") != "Pom").groupby(
    "lot", "Contact_status", "Bought_Status", "Payment_Type"
).count().display()

# COMMAND ----------

feedback_azay.filter((F.col("lead_type") != "Pom") & (F.col("Bought_Status")=="Bought") & (F.col("Payment_Type").isNull())).display()

# COMMAND ----------

feedback_azay.filter(
    (F.col("lead_type") != "Pom")
    & (F.col("Bought_Status") == "Bought")
    & (F.col("Payment_Type").isNull())
).select("dib_cust_code").join(feedback2_azay, "dib_cust_code", "inner").display()

# COMMAND ----------

feedback1_azay.groupby("Bought_Status").count().display()

# COMMAND ----------

feedback1_azay.groupby("lead_type","Bought_Status").count().display()

# COMMAND ----------

feedback1_azay.filter(F.col("lead_type") == "Normal").groupby(
    "lead_shop_payment_type", "Bought_Status"
).count().display()

# COMMAND ----------

feedback2_azay.groupby("Payment_Type").count().display()

# COMMAND ----------

class Evaluation:
    def __init__(self, df):
        """
        df: dataframe that include all feedback 1 and feedback 2
        """
        self.df = df.filter(F.col("lead_type").isin(["Hot", "Normal"]))
        self.total = df.count()

    def contactable(self, filtered=[], cols=[]):
        contact_rate = self.df
        for col in filtered:
            contact_rate = contact_rate.filter(F.col(col).isNotNull())
        total = contact_rate.count()
        contact_rate = contact_rate.groupby("Contact_Status", *cols).count()
        contact_rate = contact_rate.withColumn("ration", F.col("count") / total)
        return contact_rate

    def get_cust(
        self, bought=None, lead_type: str = None, lead_shop_payment_type: str = None
    ):
        tmp_df = self.df
        if bought is not None:
            if bought:
                tmp_df = tmp_df.filter(F.col("Bought_Status") == "Bought")
            else:
                tmp_df = tmp_df.filter(F.col("Bought_Status") != "Bought")
        if lead_type is not None:
            tmp_df = tmp_df.filter(F.col("lead_type") == lead_type)
        if lead_shop_payment_type is not None:
            tmp_df = tmp_df.filter(
                F.col("lead_shop_payment_type") == lead_shop_payment_type
            )
        return tmp_df

# COMMAND ----------

eval_df = feedback_azay.filter(F.col("lead_type") != "Pom")

# COMMAND ----------

print(feedback_azay.count())
print(eval_df.count())

# COMMAND ----------

eval =  Evaluation(feedback_azay)

# COMMAND ----------

feedback_azay_bought = feedback1_azay.filter(F.col("Bought_Status")=="Bought").select("dib_cust_code").distinct()

# COMMAND ----------

feedback_azay_bought.count()

# COMMAND ----------

feedback_azay_bought = feedback_azay_bought.join(feedback2_azay, "dib_cust_code", "inner")

# COMMAND ----------

feedback_azay_bought.count()

# COMMAND ----------

feedback_azay_bought.filter(F.col("Total_Premium_Paid").isNull()).display()

# COMMAND ----------

eval.total

# COMMAND ----------

eval.get_cust(lead_type="Hot").select("dib_cust_code").count()

# COMMAND ----------

eval.get_cust(lead_type="Normal").select("dib_cust_code").count()

# COMMAND ----------

eval.get_cust(lead_type="Hot").select("dib_cust_code").distinct().count()

# COMMAND ----------

eval.get_cust(lead_type="Normal").select("dib_cust_code").distinct().count()

# COMMAND ----------

print(eval.get_cust(lead_type="Hot",lead_shop_payment_type="credit").select("dib_cust_code").count())
print(eval.get_cust(lead_type="Hot",lead_shop_payment_type="cash").select("dib_cust_code").count())

# COMMAND ----------

print(eval.get_cust(lead_type="Normal",lead_shop_payment_type="credit").select("dib_cust_code").count())
print(eval.get_cust(lead_type="Normal",lead_shop_payment_type="cash").select("dib_cust_code").count())

# COMMAND ----------

eval.get_cust(bought=True).display()
eval.get_cust(bought=False).display()

# COMMAND ----------

eval.get_cust(bought=True).filter(F.col("Total_Premium_Paid").isNotNull()).count()

# COMMAND ----------

eval.get_cust(bought=True).filter(F.col("Total_Premium_Paid").isNotNull()).display()

# COMMAND ----------

eval.get_cust(bought=True).groupby("lead_type","Payment_Type").count().display()

# COMMAND ----------

eval.get_cust(bought=True).filter(F.col("Payment_Type").isNull()).display()

# COMMAND ----------

eval.get_cust(bought=False).groupby("Contact_Status").count().display()

# COMMAND ----------

eval.contactable().display()

# COMMAND ----------

eval.contactable(cols = ["Bought_Status","lead_type"]).display()

# COMMAND ----------

eval.contactable(filtered = ["Payment_Type"], cols = ["Bought_Status", "lead_type", "lead_shop_payment_type", "Payment_Type"]).display()

# COMMAND ----------

eval.contactable(["lead_type", "lead_shop_payment_type"]).display()

# COMMAND ----------

eval.contactable(filtered = ["Payment_Type"], cols = ["lead_type", "Payment_Type"]).display()

# COMMAND ----------

eval.contactable( cols = ["lead_type", "Payment_Type"]).display()

# COMMAND ----------

eval.contactable(
    cols=[
        "lead_type",
        "Bought_Status",
        "Payment_Type",
        "Premium_Frequency",
        "Policy_type_self",
        "Policy_plan_self",
    ]
).display()

# COMMAND ----------

eval_cig =  Evaluation(feedback_cigna)

# COMMAND ----------

eval_cig.contactable(
    cols=[
        "lead_type",
        "Bought_Status",
        "Payment_Type",
        "Premium_Frequency",
        "Policy_type_self",
        "Policy_plan_self",
    ]
).display()

# COMMAND ----------


