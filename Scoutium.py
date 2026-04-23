
# Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma

# Gerekli kütüphanelerin import edilmesi

import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from numba.np.arrayobj import record_static_setitem_int
from pandas import pivot_table
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV, validation_curve
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


import warnings
warnings.filterwarnings("ignore") # Uyarıları kapatmak için

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



# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
# (average, highlighted) oyuncu olduğunu tahminleme
#
# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

df_a = pd.read_csv("datasets/scoutium_attributes.csv", sep=";")
df_pl = pd.read_csv("datasets/scoutium_potential_labels.csv", sep=";")

df_a.head()
df_pl.head()

# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)

df = pd.merge(df_a, df_pl, how="left", on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])
df.head()

# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
df = df.loc[df["position_id"] != 1]

# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)
df = df.loc[df["potential_label"] != "below_average"]

# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
# olacak şekilde manipülasyon yapınız.

# Görevler
# Adım 1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
# “attribute_value” olacak şekilde pivot table’ı oluşturunuz.

dff = pd.pivot_table(df, values="attribute_value", index=["player_id", "position_id", "potential_label"], columns="attribute_id")
dff.head()
# Adım 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.
dff = dff.reset_index(drop=False)
dff.head()

dff.columns = dff.columns.astype(str)
# # Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.

dff["potential_label"] = LabelEncoder().fit_transform(dff["potential_label"])


# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.

num_cols = dff.columns[3:]

# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

dff[num_cols] = StandardScaler().fit_transform(dff[num_cols])

# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
y = dff["potential_label"]
X = dff.drop(["potential_label",    "player_id"], axis=1)

def base_models(X, y, scoring="roc_auc"):

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    classifiers = [
        ('LR',       LogisticRegression(max_iter=1000)),
        ('KNN',      KNeighborsClassifier()),
        ('SVC',      SVC()),
        ('CART',     DecisionTreeClassifier()),
        ('RF',       RandomForestClassifier()),
        ('Adaboost', AdaBoostClassifier()),          # 1.8.0'da parametre yok, direkt kullan
        ('GBM',      GradientBoostingClassifier()),
        ('XGBoost',  XGBClassifier(eval_metric='logloss')),
        ('CatBoost', CatBoostClassifier(verbose=False)),
        ('LightGBM', LGBMClassifier(verbose=-1, force_col_wise=True))]

    print("Base Models....")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, classifier in classifiers:
            cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
            print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")

base_models(X, y)

rf_params = {"max_depth": [5, 8, 10, 15, None],
             "max_features": [3, 5, 7, None],
             "min_samples_split": [10, 15, 20],
             "n_estimators": [100, 200, 300, 500]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


classifiers = [('CatBoost', CatBoostClassifier(verbose=False), catboost_params),
               ('LightGBM', LGBMClassifier(verbose=-1, force_col_wise=True), lightgbm_params),
               ("RF", RandomForestClassifier(), rf_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print("Hyperparameter Optimization....")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_).fit(X,y)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(
        estimators=[
            ('CatBoost', best_models["CatBoost"]),
            ('RF', best_models["RF"]),
            ('LightGBM', best_models["LightGBM"])
        ],
        voting='soft'
    )

    # Önce CV ile gerçek performansı ölç (fit etmeden)
    cv_results = cross_validate(voting_clf, X, y,
                                cv=3,
                                scoring=["accuracy", "f1", "roc_auc"])

    print(f"Accuracy : {cv_results['test_accuracy'].mean():.4f}")
    print(f"F1 Score : {cv_results['test_f1'].mean():.4f}")
    print(f"ROC_AUC  : {cv_results['test_roc_auc'].mean():.4f}")

    # Sonra tüm veriyle fit et → production modeli
    voting_clf.fit(X, y)

    return voting_clf  # artık tüm veriyle eğitilmiş model

voting_clf = voting_classifier(best_models, X, y)


# Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.


def plot_importance(model, features, name, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])

    plt.title(f'Feature Importance: {name}')
    plt.tight_layout()

    if save:
        plt.savefig(f'importances_{name}.png')

    plt.show()

plot_importance(best_models["RF"], X, "RF")
plot_importance(best_models["LightGBM"], X, "LightGBM")
plot_importance(best_models["CatBoost"], X, "CatBoost")
