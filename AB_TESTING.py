#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.


#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri ab_testing.xlsx excel’inin ayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBidding uygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.width', 500)

#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


############################
# 1. Hipotezi Kur
############################

# H0: M1 = M2
## Kontrol grubu ve Test grubunun purchase ortalamaları arasında fark yoktur
# H1: M1 != M2
## Kontrol grubu ve Test grubunun purchase ortalamaları arasında fark vardır.

############################
# 2. Varsayım Kontrolü
############################

# Normallik Varsayımı
# Varyans Homojenliği

############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.


############################
# Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 3 ve 4. Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################



#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

df_control = pd.read_excel("ab_testing.xlsx", sheet_name="Control Group")
df_test = pd.read_excel("ab_testing.xlsx", sheet_name="Test Group")

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

df_control.describe().T
df_test.describe().T

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
df_control["Group"] = "Control"
df_test["Group"] = "Test"

df = pd.concat([df_control, df_test], ignore_index=True)


#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

# H0: M1 = M2
## Kontrol grubu ve Test grubunun purchase ortalamaları arasında fark yoktur
# H1: M1 != M2
## Kontrol grubu ve Test grubunun purchase ortalamaları arasında fark vardır

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
df.groupby("Group")["Purchase"].mean()

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.
# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz
##H0: Normallik varsayımı sağlanır
##H1: Normallik varsayımı sağlanmaz

test_stat, pvalue = shapiro(df[(df["Group"]== "Control")]["Purchase"])
print(f"Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}")

##Test Stat = 0.9773, p-value = 0.5891 p-value > 0.05 H0 reddedilemez normallik varsayımı sağlanır

test_stat, pvalue = shapiro(df[(df["Group"]== "Test")]["Purchase"])
print(f"Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}")
##Test Stat = 0.9589, p-value = 0.1541 p-value > 0.05 H0 reddedilemez normallik varsayımı sağlanır

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["Group"] == "Test", "Purchase"],
                           df.loc[df["Group"] == "Control", "Purchase"])
print(f"Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}")
##Test Stat = 2.6393, p-value = 0.1083 p-value > 0.05 H0 reddedilemez varyanslar homojendir

##Test Stat = 0.9773, p-value = 0.5891 p-value > 0.05 H0 reddedilemez normallik varsayımı sağlanır - kontrol grubu
##Test Stat = 0.9589, p-value = 0.1541 p-value > 0.05 H0 reddedilemez normallik varsayımı sağlanır - test grubu
##Test Stat = 2.6393, p-value = 0.1083 p-value > 0.05 H0 reddedilemez varyanslar homojendir

##Böylelikle parametrik test kullanılabilit iki örneklem t-test (varyans homojen)


# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

##parametrik test kullanılabilit iki örneklem t-test (varyans homojen)

test_stat, pvalue = ttest_ind(df.loc[df["Group"] == "Test", "Purchase"],
                              df.loc[df["Group"] == "Control", "Purchase"],
                              equal_var=True)
print(f"Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}")

##Test Stat = 0.9416, p-value = 0.3493 p-value > 0.05 H0 reddedilemez control ve test grubunun purchase ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.


##Test Stat = 0.9416, p-value = 0.3493 p-value > 0.05 H0 reddedilemez control ve test grubunun purchase ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

##Normallik varsayımı için shapiro wilks testini kullanarak normallik varsayımının sağlandığını gördük
# varyans homojenliği için de levene testi kullanarak varyans homojenliği varsayımının sağlandığını gördük


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

##%95 güven düzeyinde değerlendirildiğinde (alfa=0.05) iki grup arasında istatistiksel olarak anlamlı bir fark gözlenmemiştir. bu sebeple yeni uygulamaya geçilmesi purcahe üzerinde herhangi bir fark yaratmayabilir


