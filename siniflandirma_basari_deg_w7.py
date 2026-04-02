import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

#####################################################################
#CASE 2


df = pd.DataFrame({"y": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                   "y_prob": [0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25]})

threshold = 0.5

df["y_pred"] = np.where(df["y_prob"] >= threshold, 1, 0)

TP = ((df["y"] == 1) & (df["y_pred"] == 1)).sum()
FP = ((df["y"] == 0) & (df["y_pred"] == 1)).sum()
TN = ((df["y"] == 0) & (df["y_pred"] == 0)).sum()
FN = ((df["y"] == 1) & (df["y_pred"] == 0)).sum()

accuracy = (TP + TN) / (TP + FP + TN + FN)
presicion = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (presicion * recall) / (presicion + recall)

print(f'accuracy:', accuracy)
print(f'presicion:', presicion)
print(f'recall:', recall)
print(f'F1:', F1)

#accuracy: 0.8
#presicion: 0.8333333333333334
#recall: 0.8333333333333334
#F1: 0.8333333333333334

















