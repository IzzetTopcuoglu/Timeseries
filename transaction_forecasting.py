#####################################################
# Import Libraries
#####################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
# !pip install lightgbm
# conda install lightgbm
import time
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

########################
# Loading the data
########################

df = pd.read_csv("C:/Users/izzet/PycharmProjects/DSMLBC9/DSMLBC9- Ödevler/Modül_7_Time_Series/iyzico_data.csv", index_col=0, parse_dates=True)
#Veri seti içerisinde indeks bulunduğundan index_col=0 argumanı girdim.

#Adım 1: Iyzico_data.csv dosyasını okutunuz. transaction_date değişkeninin tipini date'e çeviriniz
df['transaction_date'] =  pd.to_datetime(df['transaction_date'])
#Object olan tarih bilgisini datetime tipine çevirdim.
df = df.set_index(pd.DatetimeIndex(df['transaction_date']))

########################
# EDA (Exploratory Data Analysis)
########################

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
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

#NA değer yok.
#Quantilesa bakıldığında çok ufak aykırı değerlerin bulunduğunu gözlemiyorum.

#Adım 2: Veri setinin başlangıc ve bitiş tarihleri nedir?

df['transaction_date'].min(), df['transaction_date'].max()
#tarihler 01.01.2018 ile 31.12.2020 arasında 3 tam yıldır.

#Adım 3:  Her üye iş yerindeki toplam işlem sayısı kaçtır?

df['merchant_id'].nunique()
#7 eşsiz üye iş yeri var.

df.groupby('merchant_id').agg({"Total_Transaction": ["sum", "mean", "median", "std"]})

df["Total_Transaction"].sum()
#8.391.252 kez alışveriş yapılmış.
df["Total_Paid"].sum()
#3.558.840.563 (üçmilyarbeşyüzellisekizmilyonsekizyüzkırkbinbeşyüzelliüç TL değerinde işlem hacmi)

#Adım 4: Her üye iş yerindeki toplam ödeme miktarı kaçtır?

df.groupby('merchant_id').agg({"Total_Paid": ["sum", "mean", "median", "std"]})

#Adım 5: Her üye iş yerinin her bir yıl içerisindeki transaction count grafiklerini gözlemleyiniz?

df["Total_Transaction"].plot(figsize=(15, 6))
plt.xlabel("Date")
plt.ylabel("Total Transaction")
plt.title("İyzico Total Transaction")
x = df["transaction_date"]
plt.tight_layout()
plt.show()


df["Total_Paid"].plot(figsize=(15, 6))
plt.xlabel("Date")
plt.ylabel("Total Paid")
plt.title("İyzico Total Paid")
x = df["transaction_date"]
plt.tight_layout()
plt.show()

df.groupby(["merchant_id"]).agg({
    "Total_Transaction": ["sum", "mean"],
    "Total_Paid": ["sum", "mean"]})

########################
# FEATURE  ENGINEERİNG
########################

#tarihler ile ilgili yeni kategorik değişkenler üretiyorum.
def create_date_features(df):
    df['month'] = df.transaction_date.dt.month
    df['day_of_month'] = df.transaction_date.dt.day
    df['day_of_year'] = df.transaction_date.dt.dayofyear
    df['week_of_year'] = df.transaction_date.dt.weekofyear
    df['day_of_week'] = df.transaction_date.dt.dayofweek
    df['year'] = df.transaction_date.dt.year
    df["is_wknd"] = df.transaction_date.dt.weekday // 4
    df['is_month_start'] = df.transaction_date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.transaction_date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)
df.head()

#aşırı öğrenmenin önüne gecmek için veri setine gürültü ekleme fonksiyonunu tanımlıyorum.
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))



df.columns = df.columns.str.replace("transaction_date","transaction_date_col")
df.head()

#Verileri tarih, işlem sayısı ve üye işyeri numarasına göre sıraladık. Bu sırlamadan emin değilim???
df.sort_values(by=['transaction_date','merchant_id', 'Total_Transaction'], axis=0, inplace=True)
df.head()

#Sıralanmıs veri ile farklı gecikmelere a,t faturelar oluşturuyoruz.
pd.DataFrame({"Total_Transaction": df["Total_Transaction"].values[0:10],
              "lag1": df["Total_Transaction"].shift(1).values[0:10],
              "lag2": df["Total_Transaction"].shift(2).values[0:10],
              "lag3": df["Total_Transaction"].shift(3).values[0:10],
              "lag4": df["Total_Transaction"].shift(4).values[0:10]})


df.groupby(['transaction_date','merchant_id', 'Total_Transaction'])['Total_Paid'].transform(lambda x: x.shift(1))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(['merchant_id', 'Total_Transaction'])['Total_Paid'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe
#Zaman periyodları listesinde gezip yeni featurelar üretir. Dinamik olarak isimlendirilip df'e eklenir.
df = lag_features(df, [91, 120, 152, 182, 242, 402, 542, 722])

check_df(df)
#Head kısmında Na degerler gelir, Tail kısmında gelmez Bunu kontrol etmeliyiz. Verilerimiz belirlediğimiz gün sayısı kadar
#gecikmeden etkilenebilir olduğundan ve tahmin edeceğimiz zaman aralığı 3 aylık projeksiyon olduğundan 90 gün ile 2 sene arası bir dilim aralığında
#lag features türetmeliyiz.

#şimdi na değerleir doldurma işlemini gercekleştiriyoruz.

pd.DataFrame({'Total_Paid': df['Total_Paid'].values[0:10],
              "roll2": df['Total_Paid'].rolling(window=2).mean().values[0:10],
              "roll3": df['Total_Paid'].rolling(window=3).mean().values[0:10],
              "roll5": df['Total_Paid'].rolling(window=5).mean().values[0:10]})

#GECMIS IKI DEGERIN ORTALAMASINI ALIRKEN ORNEGIN KENDISINI DE BARINDIRMAMASI ICIN SHIFT ALARAK BU ISI YAPIYORUZ.
pd.DataFrame({'Total_Paid': df['Total_Paid'].values[0:10],
              "roll2": df['Total_Paid'].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df['Total_Paid'].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df['Total_Paid'].shift(1).rolling(window=5).mean().values[0:10]})


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(['merchant_id', 'Total_Transaction'])['Total_Paid']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [91, 120, 152, 182, 242, 402, 542, 722])
#Burada gecikmelere ilişkin bilgileri de veri featrue olarak analize yansıtmaya calıstık.

pd.DataFrame({"'Total_Paid'": df['Total_Paid'].values[0:10],
              "roll2": df['Total_Paid'].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df['Total_Paid'].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df['Total_Paid'].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df['Total_Paid'].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df['Total_Paid'].shift(1).ewm(alpha=0.1).mean().values[0:10]})

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(['merchant_id', 'Total_Transaction'])['Total_Paid'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 120, 152, 182, 242, 402, 542, 722]

df = ewm_features(df, alphas, lags)

check_df(df)
#78 değişken oluşturduk. Oldukça fazla, grab_col_names kullanarak elemeye çalışabiliriz.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Observations: 7667
#Variables: 78
#cat_cols: 70
#num_cols: 8
#cat_but_car: 0
#num_but_cat: 70

num_cols
########################
# ENCODING (ONE HOT ENCODING)
########################

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in cat_cols if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.info()
#kolon sayısını 24'e düşürdük.

########################
# CUSTOM COST FUNCTION
########################

# MAE, MSE, RMSE, SSE

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

########################
# TRAIN / VALIDATION
########################
df.index
#tarihler 01.01.2018 ile 31.12.2020 arasında 3 tam yıldır. 2021 ilk 3 ay için tahmin isteniyor.

# 2020'nin başına kadar (2019'nın sonuna kadar) train seti.
train = df.loc[(df["transaction_date_col"] < "2020-10-01"), :]
train.head()
# 2020'nin ilk 3'ayı validasyon seti. Modelde mevsımsellik olduğundan dolayı, mevsim bilgisini kaçırmamak için bu işlemi gerçekleştirdik.
#bu durumda 2020 yılının son 3 çeyreğini kullanmamış olduk.
val = df.loc[(df["transaction_date_col"] >= "2020-10-01"), :]

cols = [col for col in df.columns if col not in ['transaction_date_col', "merchant_id", "Total_Paid"]]
#zaman değişkeni ve merchant id yi dahil etmeden model kurağımdan cols dfi olusturdum.

Y_train = train['Total_Paid']
X_train = train[cols]

Y_val = val['Total_Paid']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################
# LightGBM ile Zaman Serisi Modeli
########################

#!pip install lightgbm
#conda install lightgbm


# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs


#lgb modelinin kendi özel datasetine donusturerek tahmın yapıldıgında daha hızlı sonuc verıyor, bu nedenle aşağıdaki işlemi gerceklestirdik.
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

#Early stopping, best iteration is:
#[1]	training's l2: 2.71002e+11	training's SMAPE: nan	valid_1's l2: 7.58898e+11	valid_1's SMAPE: nan

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
#did not meet early stopping uyarısı gelir ise num_booost_rounf argumanını 10.000e cekmek gerekır.

smape(np.expm1(y_pred_val), np.expm1(Y_val))
#logariitmayı terse alma ıslemı
