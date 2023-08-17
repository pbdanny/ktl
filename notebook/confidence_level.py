# Databricks notebook source
import numpy as np
from scipy.stats import norm


# COMMAND ----------

# https://cosmiccoding.com.au/tutorials/abtests/

# COMMAND ----------

def get_confidence_ab_test(click_a, num_a, click_b, num_b):
    rate_a = click_a / num_a
    rate_b = click_b / num_b
    std_a = np.sqrt(rate_a * (1 - rate_a) / num_a)
    std_b = np.sqrt(rate_b * (1 - rate_b) / num_b)
    z_score = (rate_b - rate_a) / np.sqrt(std_a**2 + std_b**2)
    print(f"z-score is {z_score:0.3f}, with p-value {norm().sf(z_score):0.3f}")
    return norm.cdf(z_score)

# COMMAND ----------

get_confidence_ab_test(17, 937, 155, 4864)

# COMMAND ----------

get_confidence_ab_test(62, 3868, 248, 10445)

# COMMAND ----------

get_confidence_ab_test(1056, 52817, 761, 34946)

# COMMAND ----------

get_confidence_ab_test(1215, 62203, 539, 20576)

# COMMAND ----------


