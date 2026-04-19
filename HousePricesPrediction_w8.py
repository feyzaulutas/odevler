
########### House Price Prediction #######################


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
import math
import warnings
warnings.filterwarnings("ignore") # Uyarıları kapatmak için

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from matplotlib.pyplot import hist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",20)
pd.set_option("display.width",None)
pd.set_option("display.expand_frame_repr",True)
pd.options.display.float_format = '{:,.3f}'.format



########## Görev 1 : Keşifçi Veri Analizi ########

# Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.


df_test = pd.read_csv('datasets/test.csv')
df_train = pd.read_csv('datasets/train.csv')
df_test.columns
df_train.columns
df = pd.concat([df_test, df_train], ignore_index=True)
df.head()
df_backup = df.copy()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T)
    print("##################### Info #####################")
    print(dataframe.info())
check_df(df)

# Adım 2:  Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    def is_cat_dtype(col):
        return (pd.api.types.is_object_dtype(col)
                or isinstance(col.dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(col))

    # kategorik kolonlar
    cat_cols = [col for col in dataframe.columns
                if is_cat_dtype(dataframe[col])]

    # numerik ama kategorik
    num_but_cat = [col for col in dataframe.columns
                   if pd.api.types.is_numeric_dtype(dataframe[col])
                   and dataframe[col].nunique() < cat_th]

    # kategorik ama kardinal
    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].nunique() > car_th
                   and is_cat_dtype(dataframe[col])]

    # gerçek kategorik kolonlar
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # numerik kolonlar
    num_cols = [col for col in dataframe.columns
                if pd.api.types.is_numeric_dtype(dataframe[col])]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

[(col, df[col].dtype) for col in df.columns if "Year" in col or "Yr" in col]


# Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

# numeric olması gerekenler
[col for col in num_cols if not pd.api.types.is_numeric_dtype(df[col])]
# []
# categorical olması gerekenler
wrong_cat = [col for col in cat_cols if df[col].dtype not in ["str", "O", "category"]]

for col in wrong_cat:
    print(col, df[col].nunique())
#OverallCond 9
#BsmtFullBath 4
#BsmtHalfBath 3
#FullBath 5
#HalfBath 3
#BedroomAbvGr 8
#KitchenAbvGr 4
#Fireplaces 5
#GarageCars 6
#YrSold 5

df[wrong_cat] = df[wrong_cat].astype("category")

# Adım 4:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_cols.remove("Id")

for col in num_cols:
    num_summary(df, col)

####################################################################################
import re

# 1. data_description.txt'den açıklamaları parse etmek için
with open("datasets/_houseprice.txt", "r") as f:
    desc_text = f.read()

def get_description(col_name, text):
    pattern = rf"{col_name}\s*:(.*?)(?=\n\w|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip()[:100] if match else "bulunamadı"


####################################################################################
def get_col_group(col):
    if col in cat_but_car:
        return "cat_but_car"
    elif col in cat_cols:
        return "cat_cols"
    elif col in num_cols:
        return "num_cols"
    else:
        return "unknown"

def print_summary(df):
    # print ayarları
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 50)

    # özet tablo
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "nunique": df.nunique(),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
        "sample_values": [str(df[col].unique()[:5].tolist()) for col in df.columns],
        "description": [get_description(col, desc_text) for col in df.columns],
        "col_group": [get_col_group(col) for col in df.columns]
    })

    print(summary[["col_group", "dtype", "nunique", "sample_values", "description"]])

print_summary(df)
####################################################################################



# Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)

# Adım 6: Aykırı gözlem var mı inceleyiniz.Adım 7: Eksik gözlem var mı inceleyiniz.


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in num_cols:
    low_limit, up_limit = outlier_thresholds(df, col, q1=0.05, q3=0.95)
    print(f"{col} -> low_limit: {low_limit:.4f} | up_limit: {up_limit:.4f}")


def check_outlier(dataframe, col_name):
    if not pd.api.types.is_numeric_dtype(dataframe[col_name]):
        print(f"{col_name} numeric değil, atlandı!")
        return False
    Q1 = dataframe[col_name].quantile(0.25)
    Q3 = dataframe[col_name].quantile(0.75)
    IQR = Q3 - Q1
    low_limit = Q1 - 1.5 * IQR
    up_limit = Q3 + 1.5 * IQR

    # ✅ Tüm DataFrame yerine sadece ilgili kolona .any() uygula
    outlier_exist = ((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)).any()
    return outlier_exist

for col in num_cols:
    print(col, check_outlier(df, col))

# def check_outlier(dataframe, col_name):
#    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
#    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
#        return True
#    else:
#        return False

target = "SalePrice"

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    print(grab_outliers(df, col, index=True))

df[num_cols].nunique()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

############ Görev 2: Feature Engineering #############

# Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    print(grab_outliers(df, col, index=True))

# "Yok" anlamına gelen kolonlar → "None" ile doldurduk
no_feature_cols = [
    "PoolQC", "MiscFeature", "Alley", "Fence",
    "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "BsmtQual",
    "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "MasVnrType"
]

for col in no_feature_cols:
    if col in df.columns:
        df[col] = df[col].fillna("None")

# Kontrol için
df[no_feature_cols].isnull().sum()

# Sayısal "yok" kolonları 0 ile doldurduk
no_feature_num_cols = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
    "MasVnrArea"
]

for col in no_feature_num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Kontrol için
df[no_feature_num_cols].isnull().sum()


# Kategorik → mod ile doldurduk
for col in ["Electrical", "Exterior1st", "Exterior2nd", "SaleType", "KitchenQual","MSZoning", "Utilities", "Functional"]:
    df[col] = df[col].fillna(df[col].mode()[0])


# Sayısal → medyan ile doldur
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

missing_values_table(df)

# Adım 2:  Rare Encoder uygulayınız.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns
                    if (pd.api.types.is_object_dtype(temp_df[col])
                        or isinstance(temp_df[col].dtype, pd.CategoricalDtype)
                        or pd.api.types.is_string_dtype(temp_df[col]))
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any()]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01)

rare_analyser(df, "SalePrice", cat_cols)

# Adım 3: Yeni değişkenler oluşturunuz.
fix_cols = ["YrSold", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]

for col in fix_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Toplam banyo sayısı
df["TotalBath"] = (df["FullBath"] + df["BsmtFullBath"] +
                   0.5 * df["HalfBath"] + 0.5 * df["BsmtHalfBath"])
# toplam alan
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

# Evin yaşı
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

# Veranda toplam alanı
df["TotalPorch"] = (df["OpenPorchSF"] + df["EnclosedPorch"] +
                    df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])

def grab_col_names(dataframe, cat_th=10, car_th=20):
    def is_cat_dtype(col):
        return (pd.api.types.is_object_dtype(col)
                or isinstance(col.dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(col))

    # kategorik kolonlar
    cat_cols = [col for col in dataframe.columns
                if is_cat_dtype(dataframe[col])]

    # numerik ama kategorik
    num_but_cat = [col for col in dataframe.columns
                   if pd.api.types.is_numeric_dtype(dataframe[col])
                   and dataframe[col].nunique() < cat_th]

    # kategorik ama kardinal
    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].nunique() > car_th
                   and is_cat_dtype(dataframe[col])]

    # gerçek kategorik kolonlar
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # numerik kolonlar
    num_cols = [col for col in dataframe.columns
                if pd.api.types.is_numeric_dtype(dataframe[col])]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Adım 4:  Encoding işlemlerini gerçekleştiriniz.


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns
               if df[col].nunique() == 2
               and not pd.api.types.is_numeric_dtype(df[col])]

for col in binary_cols:
    label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


#cat_cols, num_cols, cat_but_car = grab_col_names(df)
# 10 ve altı için OHE
ohe_cols = [col for col in df.columns
            if 10 >= df[col].nunique() > 2
            and (pd.api.types.is_object_dtype(df[col])
                 or pd.api.types.is_string_dtype(df[col])
                 or isinstance(df[col].dtype, pd.CategoricalDtype))]
# 10 üstü için le_cols
le_cols = [col for col in df.columns
           if df[col].nunique() > 10
           and (pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])
                or isinstance(df[col].dtype, pd.CategoricalDtype))]

df = one_hot_encoder(df, ohe_cols)

le = LabelEncoder()
for col in le_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df_backup[cat_cols].nunique().sort_values(ascending=False)


# Adım 1:  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)

train_df = df[df["SalePrice"].notna()]
test_df = df[df["SalePrice"].isna()]

# Adım 2:  Train verisi ile model kurup, model başarısını değerlendiriniz.
# Bonus: Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz.
# Not: Log'un tersini(inverse) almayı unutmayınız

X = train_df.drop(["SalePrice"], axis=1)
# Train'de SalePrice'a log uyguladık
y = np.log1p(train_df["SalePrice"])

X_test = test_df.drop("SalePrice", axis=1)

########################## Random Forrest Modeli ############################
rf_model = RandomForestRegressor().fit(X, y)
rf_model.get_params()
# {'bootstrap': True,
#  'ccp_alpha': 0.0,
#  'criterion': 'squared_error',
#  'max_depth': None,
#  'max_features': 1.0,
#  'max_leaf_nodes': None,
#  'max_samples': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'monotonic_cst': None,
#  'n_estimators': 100,
#  'n_jobs': None,
#  'oob_score': False,
#  'random_state': None,
#  'verbose': 0,
#  'warm_start': False}

cv_results = cross_validate(rf_model, X, y, cv=10,
                            scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"])

print("RMSE:", np.sqrt(-cv_results["test_neg_mean_squared_error"].mean()))
print("MAE:", -cv_results["test_neg_mean_absolute_error"].mean())
print("R2:", cv_results["test_r2"].mean())

# RMSE: 0.14371563462093956
# MAE: 0.0976521387350043
# R2: 0.8701520547557602

######################################################

########################## LightGMB ############################

lgbm_model = LGBMRegressor().fit(X, y)
lgbm_model.get_params()


cv_results_lgbm = cross_validate(lgbm_model, X, y, cv=10,
                            scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"])

print("RMSE:", np.sqrt(-cv_results_lgbm["test_neg_mean_squared_error"].mean()))
print("MAE:", -cv_results_lgbm["test_neg_mean_absolute_error"].mean())
print("R2:", cv_results_lgbm["test_r2"].mean())

# RMSE: 0.13280418031291344
# MAE: 0.08938065378120996
# R2: 0.8895950611956476

######################################################

# Adım3: Hiper paremetre optimizasyonu gerçekleştiriniz.

# LGBM ile devam ediyoruz

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1],
               "verbosity": [-1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# Fitting 5 folds for each of 24 candidates, totalling 120 fits
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_final = cross_validate(lgbm_final, X, y, cv=10,
                            scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"])

print("RMSE:", np.sqrt(-cv_results_final["test_neg_mean_squared_error"].mean()))
print("MAE:", -cv_results_final["test_neg_mean_absolute_error"].mean())
print("R2:", cv_results_final["test_r2"].mean())

# RMSE: 0.1260705670787244
# MAE: 0.08345698456817305
# R2: 0.9003744667599296

# Adım4: Değişken önem düzeyini inceleyeniz.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_final, X)

# Tahmin et ve inverse al
y_pred = np.expm1(lgbm_final.predict(X_test))

# Bonus: Test verisinde boş olan salePrice değişkenlerini tahminleyiniz ve Kaggle sayfasına submit
# etmeye uygun halde bir dataframe oluşturup sonucunuzu yükleyiniz.

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": y_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()











