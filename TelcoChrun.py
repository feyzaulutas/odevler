

# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
import missingno as msno
from datetime import date

from matplotlib.pyplot import hist

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_validate
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Görev 1 : Keşifçi Veri Analizi
######################################
# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

################################################################################################
df = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df.head()

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

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # kategorik kolonlar
    cat_cols = [col for col in dataframe.columns
                if pd.api.types.is_object_dtype(dataframe[col])
                or isinstance(dataframe[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(dataframe[col])]

    # numerik ama kategorik
    num_but_cat = [col for col in dataframe.columns
                   if pd.api.types.is_numeric_dtype(dataframe[col])
                   and dataframe[col].nunique() < cat_th]

    # kategorik ama kardinal
    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].nunique() > car_th
                   and (pd.api.types.is_object_dtype(dataframe[col])
                        or pd.api.types.is_string_dtype(dataframe[col]))]

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

#
# Observations: 7043
# Variables: 21
# cat_cols: 17
# num_cols: 2
# cat_but_car: 2
# num_but_cat: 1
# cat_cols
# ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn', 'SeniorCitizen']
# num_cols
# ['tenure', 'MonthlyCharges']
# cat_but_car
# ['customerID', 'TotalCharges']

##############################################################################################
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df["SeniorCitizen"] = df["SeniorCitizen"].astype("str")

df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["TotalCharges"].isnull().sum()

df.drop("customerID", axis=1, inplace=True)

df.info()

##############################################################################################
# Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T, end="\n\n\n")

for col in num_cols:
    num_summary(df, col)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

for col in cat_cols:
    cat_summary(df, col)

##############################################################################################
# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])
le.inverse_transform([0, 1])
df.head()


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

##############################################################################################
# Adım 5: Aykırı gözlem var mı inceleyiniz.Adım 6: Eksik gözlem var mı inceleyiniz.

# outlier

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
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
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(f"{col} -> Outlier var mı: {check_outlier(df, col)}")

# tenure -> Outlier var mı: False
# MonthlyCharges -> Outlier var mı: False

# missing

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)


# Görev 2 : Feature Engineering
##############################################################################################

# Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
##############################################################################################

# outliers
# tenure -> Outlier var mı: False
# MonthlyCharges -> Outlier var mı: False
# TotalCharges -> Outlier var mı: False

# missing

missing_values_table(df)

df[df["tenure"] == 0]

df["TotalCharges"] = df["TotalCharges"].fillna(0)

##############################################################################################
# Adım 2: Yeni değişkenler oluşturunuz.

df.describe([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T

df.loc[df["tenure"] == 0, "tenure_cat"] = "New"
df.loc[(df["tenure"] >= 1) & (df["tenure"] <= 12), "tenure_cat"] = "Short"
df.loc[(df["tenure"] >= 13) & (df["tenure"] <= 48), "tenure_cat"] = "Mid"
df.loc[df["tenure"] > 48, "tenure_cat"] = "Loyal"

df["MonthlyCharges_level"] = pd.cut(df["MonthlyCharges"],
                                  bins=4,
                                  labels=["Basic", "Standard", "Premium", "Platinum"])

df.loc[(df['gender'] == 'Male') & (df['SeniorCitizen'] == "0"), 'gender_cat'] = 'youngmale'
df.loc[(df['gender'] == 'Female') & (df['SeniorCitizen'] == "0"), 'gender_cat'] = 'youngfemale'
df.loc[(df['gender'] == 'Male') & (df['SeniorCitizen'] == "1"), 'gender_cat'] = 'seniormale'
df.loc[(df['gender'] == 'Female') & (df['SeniorCitizen'] == "1"), 'gender_cat'] = 'seniorfemale'

df.head()

###############################################################################################
# Adım 3:  Encoding işlemlerini gerçekleştiriniz.



def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns
               if df[col].nunique() == 2
               and not pd.api.types.is_numeric_dtype(df[col])]


for col in binary_cols:
    df = label_encoder(df, col)

df.head()
df.info()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


#cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


df = one_hot_encoder(df, ohe_cols)

##############################################################################################
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

df[num_cols].head()
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Görev 3 : Modelleme
######################################

# Adım 1:  Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip.
# En iyi 4 modeli seçiniz.
y = df["Churn"]

X = df.drop(["Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))


#               precision    recall  f1-score   support
#            0       0.86      0.91      0.88      1049
#            1       0.68      0.58      0.63       360
#     accuracy                           0.82      1409
#    macro avg       0.77      0.74      0.76      1409
# weighted avg       0.82      0.82      0.82      1409

y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.8488289701377614


##################################################################################################
# df1 olarak veri setini okuttum ve yeni feature eklemeden bir model daha deniyorum

df1 = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df1["SeniorCitizen"] = df1["SeniorCitizen"].astype("str")

df1["TotalCharges"] = df1["TotalCharges"].replace(" ", np.nan)
df1["TotalCharges"] = df1["TotalCharges"].astype(float)
df1["TotalCharges"].isnull().sum()

df1.drop("customerID", axis=1, inplace=True)
df1["TotalCharges"] = df1["TotalCharges"].fillna(0)

check_df(df1)

df1cat_cols, df1num_cols, df1cat_but_car = grab_col_names(df1)

df1_binary_cols = [col for col in df1.columns
               if df1[col].nunique() == 2
               and not pd.api.types.is_numeric_dtype(df1[col])]

for col in df1_binary_cols:
    df1 = label_encoder(df1, col)

df1.head()
df1.info()

df1_ohe_cols = [col for col in df1.columns if 10 >= df1[col].nunique() > 2]
df1 = one_hot_encoder(df1, df1_ohe_cols)
# standartlaştırma
rs1 = RobustScaler()
df1[df1num_cols] = rs1.fit_transform(df1[df1num_cols])

df1[df1num_cols].head()
df1.head()

df1cat_cols, df1num_cols, df1cat_but_car = grab_col_names(df1)
# model
y1 = df1["Churn"]

X1 = df1.drop(["Churn"], axis=1)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1,
                                                    y1,
                                                    test_size=0.20)

log_model1 = LogisticRegression(max_iter=1000).fit(X1_train, y1_train)

y1_pred = log_model1.predict(X1_test)
y1_prob = log_model1.predict_proba(X1_test)[:, 1]

print(classification_report(y1_test, y1_pred))

# feature eklemediğim df1 modelin başarısı
#               precision    recall  f1-score   support
#            0       0.85      0.89      0.87      1053
#            1       0.64      0.55      0.59       356
#     accuracy                           0.81      1409
#    macro avg       0.74      0.72      0.73      1409
# weighted avg       0.80      0.81      0.80      1409

y1_prob = log_model1.predict_proba(X1)[:, 1]
roc_auc_score(y1, y1_prob)
# 0.8476408878983549

###########################################
# feature eklediğim df modelin başarısı
#               precision    recall  f1-score   support
#            0       0.86      0.91      0.88      1049
#            1       0.68      0.58      0.63       360
#     accuracy                           0.82      1409
#    macro avg       0.77      0.74      0.76      1409
# weighted avg       0.82      0.82      0.82      1409

# roc_auc_score(y, y_prob)
# # 0.8488289701377614
#################################################

##################################################################################################

# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve
# bulduğunuz hiparparametrelerile modeli tekrar kurunuz


knn_model = KNeighborsClassifier().fit(X, y)


# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# acc 0.84
# f1 0.68
# AUC
roc_auc_score(y, y_prob)
# 0.896

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7718327593715724
cv_results['test_f1'].mean()
# 0.5557603228242334
cv_results['test_roc_auc'].mean()
# 0.7853321091120375


knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])


# KNN default parameters

# cv_results['test_accuracy'].mean()
# # 0.7718327593715724
# cv_results['test_f1'].mean()
# # 0.5557603228242334
# cv_results['test_roc_auc'].mean()
# # 0.7853321091120375

cv_results['test_accuracy'].mean()
#0.7998036163623459
cv_results['test_f1'].mean()
#0.5846447564494598
cv_results['test_roc_auc'].mean()
# 0.8333998348664343

#####################################################################################################
# df1 için KNN

knn_model1 = KNeighborsClassifier().fit(X1, y1)

# Confusion matrix için y_pred:
y1_pred = knn_model1.predict(X1)

# AUC için y_prob:
y1_prob = knn_model1.predict_proba(X1)[:, 1]

print(classification_report(y1, y1_pred))
roc_auc_score(y1, y1_prob)

#               precision    recall  f1-score   support
#            0       0.88      0.91      0.89      5174
#            1       0.72      0.64      0.68      1869
#     accuracy                           0.84      7043
#    macro avg       0.80      0.78      0.79      7043
# weighted avg       0.83      0.84      0.84      7043
# roc_auc_score(y1, y1_prob)
# 0.8997178550281142

cv_results1 = cross_validate(knn_model1, X1, y1, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results1['test_accuracy'].mean()
# 0.7711212215304213
cv_results1['test_f1'].mean()
# 0.5529570211093997
cv_results1['test_roc_auc'].mean()
# 0.7817373482939599

knn_params1 = {"n_neighbors": range(2, 50)}

knn_gs_best1 = GridSearchCV(knn_model1,
                            knn_params1,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(X1, y1)

knn_gs_best1.best_params_

knn_final1 = knn_model1.set_params(**knn_gs_best1.best_params_).fit(X1, y1)

cv_results1_final = cross_validate(knn_final1,
                                   X1,
                                   y1,
                                   cv=5,
                                   scoring=["accuracy", "f1", "roc_auc"])

# # KNN default parameters

# cv_results1['test_accuracy'].mean()
# # 0.7711212215304213
# cv_results1['test_f1'].mean()
# # 0.5529570211093997
# cv_results1['test_roc_auc'].mean()
# # 0.7817373482939599

cv_results1_final['test_accuracy'].mean()
# 0.8005115249370928
cv_results1_final['test_f1'].mean()
# 0.5999824144593767
cv_results1_final['test_roc_auc'].mean()
# 0.8344630133928611



