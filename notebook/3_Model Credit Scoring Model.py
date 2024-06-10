# Databricks notebook source
!pip install scikit-learn==0.22.1

# COMMAND ----------

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

# COMMAND ----------

# file_path = '/Workspace/Users/kritawats.kraiwitchaicharoen@lotuss.com/Project/(Clone) KTL/Credit_scoring_model'
file_path = "/Volumes/prod/tdm_dev/edminput/filestore/model/ktl_credit_scoring/"

# COMMAND ----------

# DBTITLE 1,Load Data 
df = spark.table("tdm_dev.th_lotuss_ktl_txn_year_rounded_full_agg_data")

# COMMAND ----------

pd.set_option('display.precision', 15)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.4f}'.format
id_column = 'APPLICATION_NO'
target_column = 'BAD60_FLAG'

# COMMAND ----------

selected_features = [
 'PCT_PYMT_CARD_VISITS'
,'PCT_PYMT_CASH_VISITS'
,'CC_TENURE'
,'SD_UNITS_MNTH'
,'PCT_WKND_FLAG_Y_VISITS'
,'PCT_WKND_FLAG_Y_UNITS'
,'PCT_WKND_FLAG_Y_SPEND'
,'MAX_UNITS_MNTH'
,'AFFLUENCE_UM'
,'CAT_L22_0204_SPU'
,'CAT_L22_0204_SPV'
,'PCT_TIME_LATE_VISITS'
,'PCT_TIME_LATE_UNITS'
,'PREMIUM_SPV'
,'SD_VISITS_MNTH'
,'CAT_L22_0204_L9_SPV'
,'CAT_L22_0204_L9_SPU'
,'AVG_UNITS_MNTH'
,'CAT_L22_1305_SPU'
,'PCT_CAT_L21_0204230_UNITS'
,'PCT_CAT_L21_0204230_VISITS'
,'PREMIUM_SPU'
,'PCT_TIME_LATE_SPEND'
,'PCT_CAT_L21_0115231_VISITS'
,'Q1_UPV'
,'AFFLUENCE_LA'
,'CAT_L22_1305_SPV'
,'PCT_CAT_L21_0204230_SPEND'
,'CAT_L22_0101_L9_SPU'
,'CAT_L22_1305_L9_SPU'
,'PCT_LAST_WKND_PROMO_UNITS'
,'CAT_L22_0204_L6_SPV'
,'CAT_L30_02_SPV'
,'CAT_L30_13_SPU'
,'CAT_L22_1011_L9_SPU'
,'PCT_CAT_L21_0115231_SPEND'
,'PCT_CAT_L21_0115231_UNITS'
,'PCT_CAT_L30_02_SPEND'
,'CAT_L22_0204_L6_SPU'
,'PCT_PCT_LAST_WKND_VISITS'
,'PCT_CAT_L30_02_L6_VISITS'
,'LAST_WKND_SPU'
,'CAT_L22_1305_L6_SPU'
,'PCT_CAT_L22_0115_VISITS'
,'PCT_CAT_L21_1306012_UNITS'
,'CAT_L30_02_L6_SPV'
,'PCT_TIME_LATE_PROMO_SPEND'
,'PCT_CAT_L30_02_VISITS'
,'PCT_CAT_L21_0314068_SPEND'
,'PCT_CAT_L21_1306011_VISITS'
,'CAT_L22_0101_L6_SPV'
,'PCT_CAT_L22_0203_L9_VISITS'
,'PCT_CAT_L22_0204_L9_VISITS'
,'PCT_TIME_LATE_PROMO_UNITS'
,'CAT_L30_13_L6_SPU'
,'CAT_L22_0101_L9_SPV'
,'Q2_SPV'
,'CAT_L30_02_L9_SPV'
,'AVG_SPEND_MNTH'
,'Q2_UPV'
,'CAT_L30_13_L9_SPU'
,'PCT_CAT_L21_1043025_SPEND'
,'PCT_CAT_L30_02_L9_VISITS'
,'PCT_CAT_L21_0102070_VISITS'
,'CAT_L22_1306_L9_SPU'
,'PREMIUM_UPV'
,'SD_SPEND_MNTH'
,'PCT_CAT_L22_0204_L6_VISITS'
,'CAT_L22_1011_SPU'
,'CAT_L22_0101_SPU'
,'Q1_SPV'
,'PCT_CAT_L22_0203_L6_VISITS'
,'PCT_CAT_L22_0115_L9_UNITS'
,'PCT_CAT_L30_02_L9_SPEND'
,'L9_PROMO_ITEMS'
,'PCT_TIME_EVENING_PROMO_UNITS'
,'MAX_SPEND_MNTH'
,'AVG_UNITS_WK'
,'PCT_CAT_L22_0203_L3_VISITS'
,'PCT_CAT_L21_0204009_SPEND'
,'PCT_TIME_LATE_MARK_DOWN_UNITS'
,'CAT_L22_1306_SPU'
,'PCT_CAT_L30_01_L6_UNITS'
,'UNITS'
,'PCT_CAT_L30_02_L9_UNITS'
,'PCT_CAT_L30_02_UNITS'
,'PCT_CAT_L22_0233_UNITS'
,'PCT_CAT_L30_01_L9_VISITS'
,'Q1_SPU'
,'FEST_APRIL_SPV'
,'PCT_CAT_L21_1043025_UNITS'
,'PCT_CAT_L22_0140_L3_UNITS'
,'PCT_CAT_L22_0140_L6_UNITS'
,'PCT_CAT_L21_0204223_SPEND'
,'PCT_CAT_L22_0115_UNITS'
,'CAT_L30_02_L3_SPV'
,'PCT_CAT_L21_1046020_VISITS'
,'CAT_L30_02_L3_SPU'
,'PCT_SF_EXPRESS_SPEND'
,'CAT_L30_02_L9_UPV'
,'CAT_L22_0203_L3_SPU'
,'PCT_CAT_L21_1306011_UNITS'
,'CAT_L30_10_L9_SPU'
,'CAT_L22_0115_L9_SPV'
,'SPEND'
,'Q4_SPV'
,'PCT_CAT_L22_0203_VISITS'
,'PCT_PROMO_UNITS'
,'AVG_VISITS_MNTH'
,'CAT_L30_02_SPU'
,'FEST_XMAS_SPU'
,'CAT_L22_0204_L3_SPU'
,'PCT_L9_PROMO_SPEND'
,'CAT_L30_01_L6_SPU'
,'LAST_WKND_UPV'
,'PCT_TIME_AFTERNOON_PROMO_UNITS'
,'PCT_CAT_L21_1043025_VISITS'
,'PCT_CAT_L30_13_VISITS'
,'PCT_MIS_MIXED_VISITS'
,'PCT_CAT_L22_0102_SPEND'
,'PCT_CAT_L22_1306_VISITS'
,'PCT_CAT_L22_0937_L9_UNITS'
,'PCT_CAT_L30_13_UNITS'
,'PCT_CAT_L22_0102_L3_VISITS'
,'CAT_L22_1305_L3_SPU'
,'PCT_CAT_L21_1306012_SPEND'
,'PCT_CAT_L21_0203001_VISITS'
,'PCT_CAT_L21_1306011_SPEND'
,'PCT_PROMO_SPEND'
,'PCT_CAT_L22_1306_L6_VISITS'
,'PCT_L6_UNITS'
,'CAT_L22_0115_SPU'
,'L6_PROMO_ITEMS'
,'PCT_CAT_L30_13_L9_VISITS'
,'PCT_CAT_L22_1046_UNITS'
,'AVG_VISITS_WK'
,'PCT_CAT_L21_0102070_SPEND'
,'CAT_L30_13_L6_SPV'
,'PCT_TIME_MORNING_VISITS'
,'AVG_SPEND_WK'
,'PCT_PYMNT_VOUCHER_VISITS'
,'PCT_CAT_L21_1011026_SPEND'
,'CAT_L22_0204_L3_SPV'
,'Q3_SPV'
,'CAT_L30_01_L3_UPV'
,'PCT_CAT_L22_0204_L3_VISITS'
,'CAT_L22_0115_SPV'
,'CAT_L22_0203_UPV'
,'PCT_CAT_L22_0937_L9_VISITS'
,'PCT_FEST_APRIL_UNITS'
,'PCT_SA_BKK_UNITS'
,'CAT_L22_0101_L9_UPV'
,'PCT_CAT_L30_13_L6_VISITS'
,'CAT_L22_0138_L9_SPV'
,'CAT_L30_10_L6_SPU'
,'PCT_CAT_L22_1046_SPEND'
,'PCT_CAT_L22_0115_L9_VISITS'
,'FEST_XMAS_UPV'
,'PCT_FEST_APRIL_VISITS'
,'CAT_L30_01_L3_SPV'
,'PCT_L6_VISITS'
,'PCT_CAT_L30_03_L9_UNITS'
,'CAT_L30_01_L6_SPV'
,'Q3_SPU'
,'LAST_WKND_SPV'
,'PCT_CAT_L22_0102_L9_VISITS'
,'PCT_CAT_L21_0203004_VISITS'
,'PCT_CAT_L22_1305_L6_VISITS'
,'PCT_CAT_L30_13_L3_VISITS'
,'PCT_MIS_ALC_TOB_VISITS'
,'PCT_CAT_L30_02_L3_SPEND'
,'PCT_CAT_L22_1306_L9_VISITS'
,'CAT_L22_1305_L9_SPV'
,'PCT_CAT_L22_0204_L9_SPEND'
,'PCT_CAT_L30_01_L6_VISITS'
,'CAT_L22_0309_L6_SPV'
,'PCT_CAT_L21_0203003_SPEND'
,'PCT_CAT_L22_1305_L9_VISITS'
,'CAT_L22_0203_L3_SPV'
,'CAT_L30_02_UPV'
,'PCT_CAT_L30_02_L3_VISITS'
,'PCT_CAT_L30_09_L9_UNITS'
,'PCT_L9_UNITS'
,'PCT_CAT_L21_1046243_SPEND'
,'PCT_CAT_L21_0101235_UNITS'
,'PCT_TIME_MORNING_UNITS'
,'PCT_CAT_L30_01_L3_SPEND'
,'PCT_MIS_FRESH_SPEND'
,'PCT_CAT_L30_03_L3_VISITS'
,'CAT_L22_0115_L6_SPV'
,'PCT_CAT_L21_0138218_UNITS'
,'CAT_L22_1306_L6_SPV'
,'Q4_UPV'
,'PCT_Q2_VISITS'
,'PCT_Q3_UNITS'
,'PCT_CAT_L30_13_L6_SPEND'
,'PCT_CAT_L30_01_L3_UNITS'
,'N_DISTINCT_L20'
,'PCT_CAT_L21_0140071_UNITS'
,'CAT_L30_02_L6_SPU'
,'CAT_L22_0102_L9_SPV'
,'FEST_XMAS_SPV'
,'CAT_L22_0309_L3_SPV'
,'PCT_CAT_L22_0204_VISITS'
,'CAT_L22_0102_SPV'
,'PCT_MIS_FOR_LATER_UNITS'
,'PCT_CAT_L22_1011_L9_VISITS'
,'PCT_CAT_L30_04_UNITS'
,'PCT_MIS_BIG_TICKET_UNITS'
,'PCT_CAT_L22_0204_L6_SPEND'
,'PCT_CAT_L30_13_L9_SPEND'
,'CAT_L22_0115_L9_SPU'
,'PCT_CAT_L30_03_VISITS'
,'PCT_MIS_HOME_REFRESH_VISITS'
,'PCT_CAT_L30_04_L9_UNITS'
,'PCT_CAT_L22_0937_UNITS'
,'CAT_L30_02_L9_SPU'
,'PCT_CAT_L30_09_VISITS'
,'CAT_L30_01_L9_SPV'
,'PCT_CAT_L22_0204_SPEND'
,'CAT_L30_13_L3_SPU'
,'VISITS'
,'PCT_CAT_L30_01_L9_UNITS'
,'CAT_L30_10_L6_SPV'
,'PCT_CAT_L21_0314080_UNITS'
,'PCT_CAT_L22_0309_VISITS'
,'PCT_Q2_UNITS'
,'PCT_CAT_L21_0204223_VISITS'
,'PCT_CAT_L21_0138218_VISITS'
,'Q2_SPU'
,'CAT_L30_04_L9_SPV'
,'CAT_L22_0140_L3_UPV'
,'PCT_L6_PROMO_SPEND'
,'PCT_CAT_L21_0139221_UNITS'
,'PCT_CAT_L21_0203002_UNITS'
,'PCT_CAT_L30_10_L9_VISITS'
,'PCT_CAT_L21_0308043_VISITS'
,'PCT_MIS_FOR_LATER_VISITS'
,'PCT_CAT_L22_0140_L9_UNITS'
,'PCT_CAT_L22_0204_L3_UNITS'
,'PCT_CAT_L22_1305_L3_VISITS'
,'PCT_CAT_L21_0937212_SPEND'
,'PCT_CAT_L22_1306_L9_UNITS'
,'PCT_SF_EXPRESS_VISITS'
,'PCT_CAT_L22_0444_VISITS'
,'PCT_CAT_L21_0204222_SPEND'
,'PCT_CAT_L22_0203_L6_UNITS'
,'CAT_L22_0203_SPU'
,'PCT_CAT_L22_0937_VISITS'
,'PCT_CAT_L21_0102070_UNITS'
,'PCT_CAT_L21_0102016_SPEND'
,'PCT_CAT_L21_0203006_VISITS'
,'PCT_CAT_L21_1046020_UNITS'
,'PCT_CAT_L22_0204_L3_SPEND'
,'PCT_L6_SPEND'
,'PCT_CAT_L22_1306_L6_SPEND'
,'PCT_MIS_FOR_LATER_SPEND'
,'CAT_L30_13_UPV'
,'CAT_L30_13_L3_SPV'
,'PCT_Q4_UNITS'
,'CAT_L30_10_L3_SPU'
,'CAT_L30_01_L6_UPV'
,'PCT_CAT_L21_0203001_UNITS'
,'PCT_TIME_AFTERNOON_SPEND'
,'PCT_CAT_L21_0314050_VISITS'
,'PCT_TIME_LUNCH_PROMO_SPEND'
,'PCT_CAT_L22_0115_L3_VISITS'
,'PCT_CAT_L21_0233707_UNITS'
,'PCT_CAT_L22_1305_L6_UNITS'
,'Q4_SPU'
,'PCT_CAT_L21_0115017_SPEND'
]


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Mapping

# COMMAND ----------

import re

new_col = df.columns

# rename according to old features
replacements = {
    'DIV': 'L30',
    'DEP': 'L22',
    'SEC': 'L21',
    'CLASS': 'L20',
    '%3_14_50&3_51_416%': '0314050',
    '%2_3_195&2_3_52%': '0233701',
    '%13_77&13_78&13_79%': '1305',
    '%1_2_187&1_2_86%': '0102016',
    'Total_Spend': 'SPEND',
    'Total_Units': 'UNITS',
    'Total_Visits': 'VISITS',
    'MINISUPERMARKET': 'EXPRESS',
    'BKK&VICINITIES': 'BKK',
    '%1_71_136%': '0101235',
    '%2_66_353%': '0233707'
}
replaced_col = list(map(lambda s: re.sub('|'.join(re.escape(key) for key in replacements.keys()), lambda match: replacements[match.group(0)], s), new_col))

# transform category code to old format
transformed_list = []

for string in replaced_col:
    parts = string.split('%')
    
    if len(parts) > 1:
        parts_between_percent = parts[1].split('_')
        
        for i in range(len(parts_between_percent)):
            if len(parts_between_percent[i]) == 1:
                parts_between_percent[i] = '0' + parts_between_percent[i]
        
        if len(parts_between_percent) >= 3 and len(parts_between_percent[2]) < 3:
            parts_between_percent[2] = '0' * (3 - len(parts_between_percent[2])) + parts_between_percent[2]
        
        transformed_string = parts[0] + ''.join(parts_between_percent)  + ''.join(parts[2:])
        transformed_list.append(transformed_string)
    else:
        transformed_list.append(string)

# print(transformed_list)

# COMMAND ----------

dic = dict(zip(transformed_list, new_col))

filtered_dic = {key: value for key, value in dic.items() if key in selected_features}


# COMMAND ----------

def getPosteriorAdj(pred, constant):
    offset = constant
    x_beta = (offset*pred)/(1+((offset-1)*pred))
    return x_beta

def calcScore(proba):
    E = 497
    D = 46
    P = proba
    scores = E + (D/np.log(2))*np.log((1-P)/P)
    return scores

# COMMAND ----------

# pkl_filename1 = file_path + r'/KTL_Credit_Scoring_Final_RF_v1.pkl'

pkl_filename1 = file_path + r'/KTL_Credit_Scoring_Final_RF_v1.pkl'

with open(pkl_filename1, 'rb') as file:
    RF = pickle.load(file) 

# COMMAND ----------

df_oot = df.select(list(filtered_dic.values()))
df_oot = df_oot.fillna(-999999)

# COMMAND ----------

# DBTITLE 1,Rename Columns To Old Name
for new_name, old_name in filtered_dic.items():
    df_oot = df_oot \
    .withColumnRenamed(old_name, new_name)

# COMMAND ----------

# DBTITLE 1,Convert to Pandas
df_oot = df_oot.toPandas()

# COMMAND ----------

xoot=df_oot
test_oot_pred = getPosteriorAdj(RF.predict_proba(xoot)[:,1],1.7889)
df_oot['Probability'] = test_oot_pred
df_oot['Score'] = calcScore(test_oot_pred)
selected_features.append('Probability')
selected_features.append('Score')

# COMMAND ----------

results = spark.createDataFrame(df_oot) 
# results.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_credit_prediction")
