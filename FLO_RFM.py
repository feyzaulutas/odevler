
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)

###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

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
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.

import datetime as dt
import pandas as pd
from numpy.conftest import coerce
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
           # 2. Veri setinde
                     # a. İlk 10 gözlem,
df.head(10)
                     # b. Değişken isimleri,
df.columns
df["order_channel"].value_counts()
df["last_order_channel"].value_counts()
df["interested_in_categories_12"].value_counts()
df.nunique()
                     # c. Betimsel istatistik,
df.describe().T

                     # d. Boş değer
df.isnull().sum()

                     # e. Değişken tipleri, incelemesi yapınız.
df.info()

           # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını
           # ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
date_cols = df.columns[df.columns.str.contains("date", case=False)]

df[date_cols] = df[date_cols].apply(pd.to_datetime)
df.info()
## 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
channel_summary = df.groupby("order_channel").agg({"master_id" : "count",
                                                   "order_num_total_ever": "sum",
                                                   "customer_value_total_ever": "sum"})

           # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.sort_values("customer_value_total_ever", ascending=False).head(10)
##alternatif
df.sort_values("customer_value_total_ever", ascending=False)[:10]

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.sort_values("order_num_total_ever", ascending=False).head(10)
           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

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
    print(dataframe.describe([0.05, 0.50, 0.95, 0.99]).T)

check_df(df)

def pre_rfm(dataframe):
    dataframe["order_num_total_ever"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total_ever"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]
    date_cols = dataframe.columns[dataframe.columns.str.contains("date", case=False)]
    dataframe[date_cols] = dataframe[date_cols].apply(pd.to_datetime)
    return df
df = df_.copy()

pre_rfm(df)

# GÖREV 2: RFM Metriklerinin Hesaplanması
##Adım 1: Recency, Frequencyve Monetarytanımlarını yapınız
df["last_order_date"].max()
##2021-05-30
today_date = dt.datetime(2021, 6, 2)

# Adım 2: Müşteri özelinde Recency, Frequency ve Monetarymetriklerini hesaplayınız.
df.groupby("master_id").agg({'last_order_date' : lambda last_order_date : (today_date - last_order_date.max()).days,
                                   'order_num_total_ever': lambda order_num_total_ever: order_num_total_ever.sum(),
                                   'customer_value_total_ever' : lambda customer_value_total_ever: customer_value_total_ever.sum()})



# Adım 3: Hesapladığınız metrikleri rfmisimli bir değişkene atayınız.
rfm = df.groupby("master_id", as_index=False).agg({'last_order_date' : lambda last_order_date : (today_date - last_order_date.max()).days,
                                   'order_num_total_ever': lambda order_num_total_ever: order_num_total_ever.sum(),
                                   'customer_value_total_ever' : lambda customer_value_total_ever: customer_value_total_ever.sum()})


# Adım 4: Oluşturduğunuz metriklerin isimlerini  recency, frequencyve monetaryolarak değiştiriniz.
rfm.columns = ["customer_id", 'recency', 'frequency', 'monetary']


# recencydeğerini hesaplamak için analiz tarihini maksimum tarihten 2 gün sonrası seçebilirsiniz



# GÖREV 3: RF ve RFM Skorlarının Hesaplanması


##Adım 1: Recency, Frequencyve Monetarymetriklerini qcutyardımı ile 1-5 arasında skorlara çeviriniz.
# Adım 2: Bu skorları recency_score, frequency_scoreve monetary_scoreolarak kaydediniz.
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])



# Adım 3: recency_scoreve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)



# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

##Adım 1: Oluşturulan RF skorları için segmenttanımlamaları yapınız.
# Adım 2: Aşağıdaki seg_mapyardımı ile skorları segmentlereçeviriniz.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
rfm["segment"].unique()
# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm[["segment", 'recency', 'frequency', 'monetary']].groupby('segment').agg("mean")


# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor.Dahil ettiği markanın ürün fiyatları genelmüşteri
# tercihlerinin üstünde.Bu nedenle markanın tanıtımı ve ürüns atışları için ilgilenecek profildeki müşterilerle özel olarak
##iletişime geçmek isteniyor.Sadık müşterilerinden(champions,loyal_customers)ve kadın kategorisinden alışveriş
##yapan kişiler özel olarak iletişim kurulacak müşteriler.Bu müşterilerin id numaralarını csv dosyasına kaydediniz


target_seg_cust_id = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]
cust_id = df[(df["master_id"].isin(target_seg_cust_id)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_id.to_csv("yeni_marka_müşteri_id_v2", index=False)


# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
# alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor.
#Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
rfm["segment"].unique()

target_disc_seg_cust_id = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])]["customer_id"]
disc_cust_ids = df[(df["master_id"].isin(target_disc_seg_cust_id)) & ((df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_id.to_csv("disc_cust_id_v2", index=False)

# GÖREV 6: Tüm süreci fonksiyonlaştırınız.

def create_rfm(dataframe):
    dataframe["order_num_total_ever"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total_ever"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]
    date_cols = dataframe.columns[dataframe.columns.str.contains("date", case=False)]
    dataframe[date_cols] = dataframe[date_cols].apply(pd.to_datetime)

    dataframe["last_order_date"].max()
    today_date = dt.datetime(2025, 6, 2)

    rfm = dataframe.groupby("master_id", as_index=False).agg(
        {'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
         'order_num_total_ever': lambda order_num_total_ever: order_num_total_ever.sum(),
         'customer_value_total_ever': lambda customer_value_total_ever: customer_value_total_ever.sum()})

    rfm.columns = ["customer_id", 'recency', 'frequency', 'monetary']

    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
    return rfm

df = df_.copy()
create_rfm(df)






























