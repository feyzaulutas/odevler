##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER


# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table.table import descr
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from sklearn.preprocessing import MinMaxScaler

###############################################################
# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.describe().T
df.isnull().sum()
def check_df(dataframe, head=5):
    print("###################### shape #############################")
    print(dataframe.shape)
    print("###################### columns #############################")
    print(dataframe.columns)
    print("###################### info #############################")
    print(dataframe.info())
    print("###################### describe().T #############################")
    print(dataframe.describe().T)
    print("###################### NA #############################")
    print(dataframe.isnull().sum())
    print("###################### Quantiles #############################")
    print(dataframe.describe([0.01, 0.50, 0.75, 0.95, 0.99]).T)

check_df(df)
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return round(up_limit), round(low_limit)

def replace_with_thresholds(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit

           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
date_cols = df.columns[df.columns.str.contains("date", case=False)]
df[date_cols] = df[date_cols].apply(pd.to_datetime)


# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 2)

           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
            # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

cltv_df = pd.DataFrame({"customer_id" : df["master_id"],
                        "recency_cltv_weekly" : (df["last_order_date"] - df["first_order_date"]).dt.days / 7,
                        "T_weekly" : (today_date - df["first_order_date"]).dt.days / 7,
                        "frequency" : df["order_num_total"],
                        "monetary_cltv_avg" : df["customer_value_total"] / df["order_num_total"]})
cltv_df = cltv_df[cltv_df["frequency"] > 1]

cltv_df.info()
cltv_df["frequency"] = cltv_df["frequency"].astype(int)
cltv_df.columns = ['customer_id', 'recency', 'T', 'frequency', 'monetary']
cltv_df.describe().T
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

## a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_month"] = bgf.predict(12, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"] = bgf.predict(24, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"], cltv_df["monetary"], time=6, freq="W", discount_rate=0.01)
cltv_df["cltv"] = cltv.values
                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv_df.sort_values("cltv", ascending=False).head(20)
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"],4, labels=["D", "C", "B", "A"])

           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz
cltv_df.groupby("cltv_segment").agg(
    {"customer_id":"count",
     "recency" :["mean", "sum"],
     "T" :["mean", "sum"],
     "frequency" :["mean", "sum"],
     "monetary" :["mean", "sum"],
     "exp_sales_3_month" :["mean", "sum"],
     "exp_sales_6_month" :["mean", "sum"],
     "exp_average_value" :["mean", "sum"],
     "cltv" :["mean", "sum"]})


# BONUS: Tüm süreci fonksiyonlaştırınız.
def create_cltv_p(dataframe, month=6):
    import datetime as dt
    import pandas as pd
    import matplotlib.pyplot as plt
    from astropy.table.table import descr
    from lifetimes import BetaGeoFitter
    from lifetimes import GammaGammaFitter
    from lifetimes.plotting import plot_period_transactions

    def outlier_thresholds(dataframe, variable):
        quartile1 = dataframe[variable].quantile(0.01)
        quartile3 = dataframe[variable].quantile(0.99)
        interquartile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquartile_range
        low_limit = quartile1 - 1.5 * interquartile_range
        return round(up_limit), round(low_limit)

    def replace_with_thresholds(dataframe, variable):
        up_limit, low_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
        dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit

    replace_with_thresholds(dataframe, "order_num_total_ever_online")
    replace_with_thresholds(dataframe, "order_num_total_ever_offline")
    replace_with_thresholds(dataframe, "customer_value_total_ever_offline")
    replace_with_thresholds(dataframe, "customer_value_total_ever_online")

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    date_cols = dataframe.columns[dataframe.columns.str.contains("date", case=False)]
    dataframe[date_cols] = dataframe[date_cols].apply(pd.to_datetime)
    today_date = dt.datetime(2021, 6, 2)

    cltv_df = pd.DataFrame({"customer_id": dataframe["master_id"],
                            "recency_cltv_weekly": (dataframe["last_order_date"] - dataframe["first_order_date"]).dt.days / 7,
                            "T_weekly": (today_date - dataframe["first_order_date"]).dt.days / 7,
                            "frequency": dataframe["order_num_total"],
                            "monetary_cltv_avg": dataframe["customer_value_total"] / dataframe["order_num_total"]})
    cltv_df = cltv_df[cltv_df["frequency"] > 1]

    cltv_df["frequency"] = cltv_df["frequency"].astype(int)
    cltv_df.columns = ['customer_id', 'recency', 'T', 'frequency', 'monetary']

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

    cltv_df["exp_sales_3_month"] = bgf.predict(12, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])
    cltv_df["exp_sales_6_month"] = bgf.predict(24, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df["frequency"], cltv_df["monetary"])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

    cltv = ggf.customer_lifetime_value(bgf, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"], cltv_df["monetary"],
                                       time=month, freq="W", discount_rate=0.01)
    cltv_df["cltv"] = cltv.values

    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df


df = df_.copy()

create_cltv_p(df)

cltv_df.groupby("cltv_segment").agg(
    {"customer_id":"count",
     "recency" :["mean", "sum"],
     "T" :["mean", "sum"],
     "frequency" :["mean", "sum"],
     "monetary" :["mean", "sum"],
     "exp_sales_3_month" :["mean", "sum"],
     "exp_sales_6_month" :["mean", "sum"],
     "exp_average_value" :["mean", "sum"],
     "cltv" :["mean", "sum"]})







