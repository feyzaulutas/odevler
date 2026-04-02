import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


df = pd.DataFrame({"X": [5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
                  "y": [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]})

b = 275
w = 90
df["y_pred"] = b + w * df["X"]

df["error"] = df["y"] - df["y_pred"]

df["squared_error"] = df["error"] ** 2

df["abs_error"] = df["error"].abs()

MSE = df["squared_error"].mean()
RMSE = np.sqrt(MSE)
MAE = df["abs_error"].mean()

df["y"].mean()

print("MSE:", MSE)
print("RMSE:", RMSE)
print("MAE:", MAE)

