
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

pd.set_option("display.max_rows", None)
df = pd.read_csv("xpersona.csv")

df.head()

## Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("xpersona.csv")

df.head()

df.info()
df.describe().T
df["AGE"].hist()
plt.show()
# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].nunique()
df["SOURCE"].value_counts()


# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.pivot_table("PRICE", "COUNTRY", aggfunc="sum", sort=True)
df.groupby("COUNTRY")["PRICE"].sum()

# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?

df["SOURCE"].value_counts()
df["SOURCE"].unique()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY")["PRICE"].mean()
df.pivot_table("PRICE", "COUNTRY")

df.groupby("COUNTRY").agg({"PRICE" : "mean"})

##Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE")["PRICE"].mean()

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean()

df.pivot_table("PRICE", ["COUNTRY", "SOURCE"])


# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

df.pivot_table("PRICE", ["COUNTRY", "SOURCE", "SEX", "AGE"])
df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"})


# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.

df.sort_values("PRICE", ascending=False).head()

agg_df = df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.


agg_df = agg_df.reset_index()

# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.

bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
my_labels = ["0_18", "19_23", "24_30", "31_40", "41_"+str(agg_df["AGE"].max())]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins, labels=my_labels)

agg_df.head()



# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.

agg_df["customers_level_based"] = (agg_df["COUNTRY"].str.upper() + "_" + agg_df["SOURCE"].str.upper() + "_" + agg_df["SEX"].str.upper() + "_" +agg_df["AGE_CAT"].astype(str))
agg_df["customers_level_based"].unique()
agg_df1 = (agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index())



# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,
##Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 seg

agg_df1["SEGMENT"] = pd.qcut(agg_df1["PRICE"], 4, labels=["D", "C", "B", "A"])

agg_df1.value_counts()

agg_df1.groupby("SEGMENT").agg({"PRICE": "mean"})

# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df1[agg_df1["customers_level_based"] == new_user]


new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df1[agg_df1["customers_level_based"] == new_user2]

def predict(x):
    new_customer = agg_df1[agg_df1["customers_level_based"] == x]
    segment = new_customer["SEGMENT"].mode()[0]
    price = new_customer["PRICE"].mean()
    print(f"The New customer's segment = {segment}, Estimated spending {price} ")


predict("FRA_IOS_FEMALE_31_40")
