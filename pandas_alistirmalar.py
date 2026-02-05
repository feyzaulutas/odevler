

##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas import pivot_table

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df.head()
df.info()
df.describe().T
#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################

df = sns.load_dataset("Titanic")

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()


#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

df.nunique()


#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df["pclass"].unique()


#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
df[["pclass", "parch"]].nunique()


#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################
df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
df.info()


#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df["embarked"] == "C"
df[df["embarked"] == "C"]
len(df[df["embarked"] == "C"])

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"] != "S"]

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df.loc[(df["age"] < 30) & (df["sex"] == "female")]
df.query("age < 30 and sex == 'female'")


#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.

df.loc[(df["fare"] > 500) | (df["age"] > 70)]

#########################################
df[(df["fare"] >500) | (df["age"] >70)].head()
df.query('fare > 500 or age >70')

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isnull().sum()


#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop("who", axis=1).head()

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()


#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

df["deck"].fillna(df["age"].median())

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################
df.groupby(["pclass", "sex"]).agg({"survived" : ["sum", "count", "mean"]})

df.pivot_table("survived", ["pclass", "sex"], aggfunc= ["sum", "count", "mean"])

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)
df.head()

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

df1 = sns.load_dataset("tips")
df1.columns
df1.info()

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df1.pivot_table("total_bill", "time", aggfunc= ["min", "max", "mean"])
df1.groupby("time").agg({"total_bill": ["min", "max", "mean"]})


#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df1.pivot_table("total_bill", "day", aggfunc= ["min", "max", "mean"])
df1.groupby("day").agg({"total_bill": ["min", "max", "mean"]})


#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################
df1.pivot_table("total_bill", ["time", "sex"], aggfunc= ["sum", "min", "max", "mean"])


#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

df1.loc[(df1["size"] < 3) & (df1["total_bill"] > 10)]["total_bill"].mean()
df1.loc[(df1["size"] < 3) & (df1["total_bill"] > 10), "total_bill"].mean()


#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################
df1["total_bill_tip_sum"] = df1["total_bill"] + df1["tip"]
df1.head()


#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

new_df = df1.sort_values("total_bill_tip_sum", ascending=False)
new_df.head(30)

