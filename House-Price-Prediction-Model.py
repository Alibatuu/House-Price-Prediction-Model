import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_validate
from lightgbm import LGBMRegressor


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

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
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
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
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                       "TARGET_COUNT": dataframe.groupby(categorical_col)[target].count()}), end="\n\n")
def load():
    data = pd.read_csv("datasets/train.csv")
    return data

# Veriye bakış

df_train = load()
df_test = pd.read_csv("datasets/test.csv")
check_df(df_train) # Bağımlı değişken SalePrice
missing_values_table(df_train)


cat_cols, num_cols, cat_but_car = grab_col_names(df_train)
cat_cols_test, num_cols_test, cat_but_car_test = grab_col_names(df_test)

for col in cat_cols:
    cat_summary(df_train, col)
num_summary(df_train, num_cols)

for col in num_cols:
    target_summary_with_num(df_train, "SalePrice", col)

for col in cat_cols:
    target_summary_with_cat(df_train, "SalePrice", col)


df_train.drop("Id",axis=1, inplace=True)
df_test.drop("Id",axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df_train)
cat_cols_test, num_cols_test, cat_but_car_test = grab_col_names(df_test)

# Aykırı değer analizi

for col in num_cols:
    print(col, check_outlier(df_train, col))
for col in num_cols_test:
    print(col, check_outlier(df_test, col))

# Eksik gözlem analizi

na_col = missing_values_table(df_train,na_name=True)
na_col_test = missing_values_table(df_test,na_name=True)

# Aykırı değerleri baskılama

for col in num_cols:
    replace_with_thresholds(df_train, col)
for col in num_cols_test:
    replace_with_thresholds(df_test, col)

# DataFrame birleştirme

df = pd.concat((df_train.drop("SalePrice",axis=1), df_test), "index" )

# Eksik değer işlemleri

df["LotFrontage"].fillna(df_train["LotFrontage"].median(), inplace=True)
none_cols = ["Alley", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature","BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtQual","BsmtCond","MasVnrType","Electrical","MSZoning","Utilities","Functional"]
for col in none_cols:
    df[col].fillna('None', inplace=True)
zero_cols = ["GarageYrBlt","GarageArea","GarageCars","BsmtFullBath","BsmtHalfBath","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","MasVnrArea"]
for col in zero_cols:
    df[col].fillna(0,inplace=True)
df['Exterior1st'].fillna('Other',inplace=True)
df['Exterior2nd'].fillna('Other',inplace=True)
df['KitchenQual'].fillna('Other',inplace=True)
df['SaleType'].fillna('Other',inplace=True)

missing_values_table(df)

# Yeni veri türetme

df_train.corr().sort_values("SalePrice",ascending=False)
df["House_Total_Area"] = df["GrLivArea"] + df["BsmtFinSF2"] + df["TotalBsmtSF"] + df["1stFlrSF"]+ df["2ndFlrSF"]
df["Overall"] = df["OverallQual"] * df["OverallCond"]

# Encoding işlemleri

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

ohe_cols = [col for col in df.columns if 23 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

# Model Kurma

X_train_pre = df.head(1460)
X_test_pre = df.tail(1459)
y_train = df_train["SalePrice"]
X_train = X_train_pre.drop(["Neighborhood"], axis=1)
X_test = X_test_pre.drop(["Neighborhood"], axis=1)

reg_model = LinearRegression().fit(X_train, y_train)
y_pred = reg_model.predict(X_train)


y_train.mean() # 180829.75753424657
mean_squared_error(y_train, y_pred) # 401152908.77237374
np.sqrt(mean_squared_error(y_train, y_pred)) # 20028.801980457385
mean_absolute_error(y_train, y_pred) # 12830.264866515488
reg_model.score(X_train, y_train) # 0.9353893964182853

# Random Forests

# Default Model

rf_model = RandomForestRegressor()
cv_results = cross_validate(rf_model, X_train, y_train, cv=5,
                            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
                            n_jobs=-1, verbose=True)
cv_results['test_neg_root_mean_squared_error'].mean() # -28329.569448693113
cv_results['test_neg_mean_absolute_error'].mean() # -16972.121869863015
cv_results['test_r2'].mean() # 0.8688807768791472

# LightGBM

# Default Model

lgbm_model = LGBMRegressor()
cv_results = cross_validate(lgbm_model, X_train, y_train, cv=5,
                            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
                            n_jobs=-1, verbose=True)
cv_results['test_neg_root_mean_squared_error'].mean() # -28360.087223988194
cv_results['test_neg_mean_absolute_error'].mean() # -16930.20015508854
cv_results['test_r2'].mean() # 0.8691480683346544

# Random Forest Grid Search

rf_model.get_params()

rf_params = {"max_depth": [8, 11, 14, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [5, 10, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)

rf_final = rf_model.set_params(**rf_best_grid.best_params_).fit(X_train, y_train)

cv_results = cross_validate(rf_final, X_train, y_train, cv=5,
                            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
                            n_jobs=-1, verbose=True)
cv_results['test_neg_root_mean_squared_error'].mean()
cv_results['test_neg_mean_absolute_error'].mean()
cv_results['test_r2'].mean()

# LightGBM Forest Grid Search

lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.001, 0.01, 0.1, 0.5],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)



lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X_train, y_train)

cv_results = cross_validate(lgbm_final, X_train, y_train, cv=5,
                            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
                            n_jobs=-1, verbose=True)
cv_results['test_neg_root_mean_squared_error'].mean()
cv_results['test_neg_mean_absolute_error'].mean()
cv_results['test_r2'].mean()






