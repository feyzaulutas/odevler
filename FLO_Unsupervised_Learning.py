
# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV

import datetime as dt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Görev 1: Veriyi Hazırlama

# Adım 1: flo_data_20K.csv verisini okutunuz.
df_original = pd.read_csv("datasets/flo_data_20k.csv").copy()
df = pd.read_csv("datasets/flo_data_20k.csv")
df.head()

df.isnull().sum()
df.info()
df.describe().T
df_original.describe().T

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, 0.01, 0.99)
    dataframe[col_name] = dataframe[col_name].astype(float)
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")


# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı)
# gibi yeni değişkenler oluşturabilirsiniz.

# tenure
date_cols = df.columns[df.columns.str.contains("date", case=False)]
df[date_cols] = df[date_cols].apply(pd.to_datetime)

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 2)

df["tenure"] = (today_date - df["first_order_date"]).dt.days

# recency
df["recency"] = (df["last_order_date"] - df["first_order_date"]).dt.days

# frequency
df["frequency"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
# monetary
df["monetary"] = df["customer_value_total_ever_online"].fillna(0) + df["customer_value_total_ever_offline"].fillna(0)


# Görev 2: K-Means ile Müşteri Segmentasyonu

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

df[num_cols].describe([0.01, 0.50, 0.75, 0.95, 0.99]).T

for col in num_cols:
    replace_with_thresholds(df, col)

# Adım 1: Değişkenleri standartlaştırınız.
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# Adım 2: Optimum küme sayısını belirleyiniz.

kmeans = KMeans()
ssd = []
K = range(1, 30)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df[num_cols])
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()
# # KElbowVisualizer hata veriyor, bu sebeple plot üzerinden göz kontrolü ile ilerliyorum, 7 küme ile ilerliyoruz

# kmeans = KMeans()
# elbow = KElbowVisualizer(kmeans, k=(2, 20))
# KElbowVisualizer hata veriyor, bu sebeple plot üzerinden göz kontrolü ile ilerliyorum, 7 küme ile
# elbow.fit(df)
# elbow.show()
# elbow.elbow_value_


# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

kmeans = KMeans(n_clusters=7).fit(df[num_cols])

kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
# 7 kümenin merkezleri
kmeans.labels_
kmeans.inertia_
# 89950.83007711287 - SSE

clusters_kmeans = kmeans.labels_

df["cluster"] = clusters_kmeans

df["cluster"] = df["cluster"] + 1
df
# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.
df.groupby("cluster")[num_cols].agg(["count","mean","median", "min", "max", "std"])


# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu

# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
df_sample = df[num_cols].sample(1000, random_state=42)
hc_average = linkage(df_sample, "ward")

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=40, color='r', linestyle='--')
plt.show()
df.describe().T

# dendogramı inceleyerek 6 küme ile ilerlemeye karar verdik

# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.

from sklearn.cluster import AgglomerativeClustering

hc_cluster = AgglomerativeClustering(n_clusters=6, linkage="ward")

hc_cluster = hc_cluster.fit_predict(df[num_cols])

df["hc_cluster"] = hc_cluster
df["hc_cluster"] = df["hc_cluster"] + 1

df.head()

# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.

df.groupby("cluster")[num_cols].agg(["count","mean","median", "min", "max", "std"])
df.groupby("hc_cluster")[num_cols].agg(["count","mean","median", "min", "max", "std"])


