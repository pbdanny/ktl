import mlflow
from mlflow.models.signature import infer_signature

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, roc_curve, RocCurveDisplay, recall_score, f1_score, auc, average_precision_score, precision_recall_curve, PrecisionRecallDisplay

from hyperopt.pyll import scope
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, SparkTrials

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
sel_col = ['prop_visit_by_cash_360_days', 'prop_visit_by_card_360_days', 'prop_promo_sales', 'afterlunch_prop_visits_360_days', 'early_morning_prop_visits_360_days', 'evening_prop_visits_360_days', 'late_evening_prop_visits_360_days', 'lunch_prop_visits_360_days', 'morning_prop_visits_360_days', 'night_prop_visits_360_days', 'Weekday_prop_visits_360_days', 'Weekend_prop_visits_360_days', 'total_sales_360_days', 'total_visits_360_days', 'apparel_prop_sales', 'baby_prop_sales', 'baby_kids_prop_sales', 'babyaccessories_prop_sales', 'babysupplement_prop_sales', 'bakery_prop_sales', 'bakeryboughtin_prop_sales', 'ball_prop_sales', 'basicfood_prop_sales', 'beveragesparty_prop_sales', 'beveragesparty_prop_visits', 'breakfast_icecream_prop_sales', 'breakfast_icecream_prop_visits', 'cookameal_prop_sales', 'cookameal_prop_visits', 'cosmetics_prop_sales', 'dairy_snacksbreaktime_prop_sales', 'dairy_snacksbreaktime_prop_visits', 'delicatessen_prop_sales', 'fishseafood_prop_sales', 'foodcourt_prop_sales', 'foodsupplement_prop_sales', 'freshmeat_seafood_prop_sales', 'frozen_prop_sales', 'fruit_prop_sales', 'fruits_healthy_prop_sales', 'giftgrocery_prop_sales', 'haircare_prop_sales', 'health_prod_prop_sales', 'health_prod_prop_visits', 'hle_prop_sales', 'hlh_prop_sales', 'hlh_prop_visits', 'hotbevsugar_prop_sales', 'household_hlh_prop_sales', 'household_hlh_prop_visits',
           'icecreamdessert_prop_sales', 'kid_prop_sales', 'kitchenstock_precooked_prop_sales', 'kitchenstock_precooked_prop_visits', 'leisure_prop_sales', 'luggage_prop_sales', 'mealsolutions_prop_sales', 'otherbakery_prop_sales', 'otherdairydrinks_prop_sales', 'otherhh_prop_sales', 'otherliquor_prop_sales', 'othermeat_seafoods_prop_sales', 'otherprecooked_prop_sales', 'othertobacco_prop_sales', 'othertoiletries_prop_sales', 'othertoiletries_prop_units', 'papergoods_prop_sales', 'personalcare_prop_sales', 'personalcare_prop_visits', 'petfoodsupplys_prop_sales', 'petshopstandalone_prop_sales', 'pharmacy_prop_sales', 'processedfood_prop_sales', 'rtd_prop_sales', 'skincare_prop_sales', 'ConsumptionforToday_prop_visits_360_days', 'Emergency_prop_visits_360_days', 'ImmediateConsumption_prop_visits_360_days', 'Others_prop_visits_360_days', 'StockUp_prop_visits_360_days', 'TopUp_prop_visits_360_days', 'Weekly_prop_visits_360_days', 'trader_prop_visits_360_days', 'online_all_sales_360_days', 'online_all_sales_90_days', 'online_all_sales_30_days', 'offline_hde_sales_360_days', 'offline_hde_visits_360_days', 'offline_hde_sales_180_days', 'offline_hde_sales_30_days', 'offline_gofresh_sales_360_days', 'offline_gofresh_sales_30_days', 'card_aging_day'] + ['truprice_seg_desc', 'facts_seg_desc', 'lifestage_seg_name', 'lifecycle_name'] + ["household_id", "Bought_Status"]


def get_train_test_data(data, target, train_size=0.8, test_size=0.5):
    """
    Split data into train, validation and test sets
    :param:
        data: pandas dataframe
        target: dictionary with target column name and positive label
        train_size: float, size of train set
        test_size: float, size of test set
    :return:
        X_train: pandas dataframe, train set features
        X_val: pandas dataframe, validation set features
        X_test: pandas dataframe, test set features
        y_train: numpy array, train set target
        y_val: numpy array, validation set target
        y_test: numpy array, test set target
    """
    X = data.drop(target["column"], axis=1)
    y = data[[target["column"]]].assign(target=lambda x: np.where(
        x == target["positive"], 1, 0))["target"].to_numpy()

    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, train_size=train_size, random_state=0, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=test_size, random_state=0, stratify=y_rem)
    return X_train, X_val, X_test, y_train, y_val, y_test


def main(spark, prjct_nm, payment, test):
    '''
    Main function to run the model
    :param: 
        spark: spark session
        prjct_nm: project name
        payment: payment type (cash or credit)
        test: test or not
    '''
    from utils.data import Data
    tmp_data = Data(spark, prjct_nm, test)
    tmp_df = tmp_data.get(cashed=payment)
    tmp_df = tmp_df.drop("cash_card_customer_flag")
    tmp_df = tmp_df.select(sel_col)
    tmp_df = tmp_data.to_pandas(tmp_df).set_index("household_id")
    # Onehot categorical variables
    data = pd.get_dummies(tmp_df,
                          prefix=["trprce", "fcts", "lfstg", "lfcyc"],
                          columns=["truprice_seg_desc", "facts_seg_desc", "lifestage_seg_name", "lifecycle_name"])
    del [tmp_data, tmp_df]
    # Replace space with underscore
    data.columns = data.columns.str.replace("\W", "_", regex=True)
    target = {
        "column": "Bought_Status",
        "positive": "Bought",
    }
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_data(
        data, target=target)
