########## Görev 1 : Keşifçi Veri Analizi ########

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

from matplotlib.pyplot import hist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Adım 1: Genel resmi inceleyiniz.

df = pd.read_csv("datasets/diabetes.csv")
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




# Adım 2: Numerik ve kategorik değişkenleri yakalayınız

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


# Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
# categoric değişken yalnızca target

### numeric değişken ve hedef değişken analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T, end="\n\n\n")

    if plot:
        missing = dataframe[numerical_col].isnull().sum()
        missing_pct = round(missing / len(dataframe) * 100, 2)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram - dağılım şekli
        dataframe[numerical_col].hist(bins=20, ax=axes[0])
        axes[0].set_xlabel(numerical_col)
        axes[0].set_title(f"{numerical_col} - Histogram")

        # Box Plot - outlier tespiti
        sns.boxplot(y=dataframe[numerical_col], ax=axes[1], color="steelblue")
        axes[1].set_title(f"{numerical_col} - Box Plot")

        fig.suptitle(f"Missing: {missing} değer ({missing_pct}%)", fontsize=10,
                     color="red" if missing > 0 else "green")

        plt.tight_layout()
        plt.show()


for col in num_cols:
    num_summary(df, col,plot=True)


# Adım 4:Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması
df["Outcome"].value_counts()


def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.boxplot(y=dataframe[numerical_col], ax=axes[0], color="steelblue")
        axes[0].set_title(f"{numerical_col} - Box Plot")
        sns.boxplot(data=dataframe, x=target, y=numerical_col, ax=axes[1],
                    hue=target, palette="Set2", legend=False)
        axes[1].set_title(f"{numerical_col} - {target} Gruplarına Göre")
        plt.tight_layout()
        plt.show()

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)




# Adım 5: Aykırı gözlem analizi yapınız

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


# Adım 6: Eksik gözlem analizi yapınız.

df.isnull().any()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)


# Adım 7: Korelasyon analizi yapınız.

corr_matrix = df.corr()

upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

######### Görev 2 : Feature Engineering ########

# Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta
# ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır.
# Bu durumudikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz

#outliers baskılama
for col in num_cols:
    print(f"{col} -> Outlier var mı: {check_outlier(df, col)}")


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe[col_name] = dataframe[col_name].astype(float)
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)


for col in num_cols:
    print(f"{col} > Outlier var mı? > {check_outlier(df, col)}")

# missing value analiz
independent_variable = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

df[independent_variable] = df[independent_variable].replace(0,np.nan)
missing_values_table(df)


# değişkenlerin standartlatırılması

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

dff = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

dff.head()
df.head()

missing_values_table(dff)

for col in dff.columns:
    result = check_outlier(dff, col)
    print(f"{col} -> {'⚠️ Outlier VAR' if result else 'Outlier yok'}")


replace_with_thresholds(dff, "SkinThickness")

msno.matrix(df)
plt.show()

missing_values_table(dff, True)
na_cols = missing_values_table(df, True)

dff.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(dff, "Outcome", na_cols)

# Adım 2: Yeni değişkenler oluşturunuz.

check_df(dff)

# age level
dff.loc[(dff['Age'] < 30), 'NEW_AGE_CAT'] = 'young'
dff.loc[(dff['Age'] >= 30) & (dff['Age'] < 55), 'NEW_AGE_CAT'] = 'mature'
dff.loc[(dff['Age'] >= 55), 'NEW_AGE_CAT'] = 'senior'

dff.head()
dff["NEW_AGE_CAT"].value_counts().sort_values(ascending=False).head()


# Glucose level
dff.loc[(dff['Glucose'] < 100), 'Glucose_level'] = 'Normal'
dff.loc[(dff['Glucose'] >= 100) & (dff['Glucose'] < 126), 'Glucose_level'] = 'Pre_diabetes'
dff.loc[(dff['Glucose'] >= 126), 'Glucose_level'] = 'diabetes'

dff["Glucose_level"].value_counts().sort_values(ascending=False).head()

# Insulin level
dff.loc[(dff['Insulin'] >= 16) & (dff['Insulin'] <= 166), 'Insulin_Cat'] = 'Normal'
dff.loc[(dff['Insulin'] < 16) | (dff['Insulin'] > 166), 'Insulin_Cat'] = 'Anormal'

dff["Insulin_Cat"].value_counts().sort_values(ascending=False).head()

# BloodPressure level
dff.loc[(dff['BloodPressure'] < 80), 'BloodPressure_level'] = 'Normal'
dff.loc[(dff['BloodPressure'] >= 80) & (dff['BloodPressure'] < 90), 'BloodPressure_level'] = 'Pre_hypertension'
dff.loc[(dff['BloodPressure'] >= 90), 'BloodPressure_level'] = 'Hypertension'

dff["BloodPressure_level"].value_counts().sort_values(ascending=False).head()
# SkinThickness level
dff.loc[(dff['SkinThickness'] < 20), 'SkinThickness_level'] = 'Low'
dff.loc[(dff['SkinThickness'] >= 20) & (dff['SkinThickness'] <= 40), 'SkinThickness_level'] = 'Normal'
dff.loc[(dff['SkinThickness'] > 40), 'SkinThickness_level'] = 'High'

dff["SkinThickness_level"].value_counts().sort_values(ascending=False).head()

# BMI level
dff.loc[(dff['BMI'] < 18.5), 'BMI_level'] = 'Underweight'
dff.loc[(dff['BMI'] >= 18.5) & (dff['BMI'] < 25), 'BMI_level'] = 'Normal'
dff.loc[(dff['BMI'] >= 25) & (dff['BMI'] < 30), 'BMI_level'] = 'Overweight'
dff.loc[(dff['BMI'] >= 30), 'BMI_level'] = 'Obese'

dff["BMI_level"].value_counts().sort_values(ascending=False).head()

# DiabetesPedigreeFunction level
dff.loc[(dff['DiabetesPedigreeFunction'] < 0.3), 'DPF_level'] = 'Low_risk'
dff.loc[(dff['DiabetesPedigreeFunction'] >= 0.3) & (dff['DiabetesPedigreeFunction'] <= 0.6), 'DPF_level'] = 'Mid_risk'
dff.loc[(dff['DiabetesPedigreeFunction'] > 0.6), 'DPF_level'] = 'High_risk'

dff["DPF_level"].value_counts().sort_values(ascending=False).head()

dff_cat_cols, dff_num_cols, dff_cat_but_car = grab_col_names(dff)

# kategorik değişkenler oluştuğu için cat summary analizi yapıldı
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in dff_cat_cols:
    cat_summary(dff, col)

#####

dff.head()

# Adım 3:  Encoding işlemlerini gerçekleştiriniz.


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in dff.columns
               if dff[col].nunique() == 2
               and not pd.api.types.is_numeric_dtype(dff[col])]


for col in binary_cols:
    label_encoder(dff, col)

dff.head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


#cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in dff.columns if 10 >= dff[col].nunique() > 2]


dff = one_hot_encoder(dff, ohe_cols)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

rs = RobustScaler()
dff[dff_num_cols] = rs.fit_transform(dff[dff_num_cols])

dff[dff_num_cols].head()
dff.head()

dff_cat_cols, dff_num_cols, dff_cat_but_car = grab_col_names(dff)


# Adım 5: Model oluşturunuz.


y = dff["Outcome"]
X = dff.drop(["Outcome"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

















