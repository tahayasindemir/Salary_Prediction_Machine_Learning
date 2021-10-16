import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helpers import eda, data_prep
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import missingno as msno
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Exploratory Data Analysis

df = pd.read_csv(r"C...\hitters.csv")

df.describe().T

eda.check_df(df)

df.columns = [col.upper() for col in df.columns]
cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

# Feature Analysis

for col in cat_cols:
    eda.cat_summary(df, col)

for col in num_cols:
    eda.num_summary(df, col)


def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show()


for col in num_cols:
    plot_numerical_col(df, col)

# We derive new variables by proportioning the last season played according to all career data.
df["NEW_C_RUNS_RATIO"] = df["RUNS"] / df["CRUNS"]
df["NEW_C_ATBAT_RATIO"] = df["ATBAT"] / df["CATBAT"]
df["NEW_C_HITS_RATIO"] = df["HITS"] / df["CHITS"]
df["NEW_C_HMRUN_RATIO"] = df["HMRUN"] / df["CHMRUN"]
df["NEW_C_RBI_RATIO"] = df["RBI"] / df["CRBI"]
df["NEW_C_WALKS_RATIO"] = df["WALKS"] / df["CWALKS"]
df["NEW_C_HIT_RATE"] = df["CHITS"] / df["CATBAT"]
df["NEW_C_RUNNER"] = df["CRBI"] / df["CHITS"]
df["NEW_C_HIT-AND-RUN"] = df["CRUNS"] / df["CHITS"]
df["NEW_C_HMHITS_RATIO"] = df["CHMRUN"] / df["CHITS"]
df["NEW_C_HMATBAT_RATIO"] = df["CATBAT"] / df["CHMRUN"]

# We can also average all career data by years.
df["NEW_CATBAT_MEAN"] = df["CATBAT"] / df["YEARS"]
df["NEW_CHITS_MEAN"] = df["CHITS"] / df["YEARS"]
df["NEW_CHMRUN_MEAN"] = df["CHMRUN"] / df["YEARS"]
df["NEW_CRUNS_MEAN"] = df["CRUNS"] / df["YEARS"]
df["NEW_CRBI_MEAN"] = df["CRBI"] / df["YEARS"]
df["NEW_CWALKS_MEAN"] = df["CWALKS"] / df["YEARS"]

# We can Level Players by Experience.
df['NEW_EXP_LEVEL'] = pd.qcut(x=df['YEARS'], q=4, labels=['BEGINNER', 'MID_LEVEL', 'EXPERIENCED', 'VETERAN'])

df.loc[(df["NEW_EXP_LEVEL"] == "BEGINNER") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "BEGINNER-EAST"
df.loc[(df["NEW_EXP_LEVEL"] == "BEGINNER") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "BEGINNER-WEST"
df.loc[(df["NEW_EXP_LEVEL"] == "MID_LEVEL") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "MID_LEVEL-EAST"
df.loc[(df["NEW_EXP_LEVEL"] == "MID_LEVEL") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "MID_LEVEL-WEST"
df.loc[(df["NEW_EXP_LEVEL"] == "EXPERINCED") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "EXPERINCED-EAST"
df.loc[(df["NEW_EXP_LEVEL"] == "EXPERINCED") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "EXPERINCED-WEST"
df.loc[(df["NEW_EXP_LEVEL"] == "VETERAN") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "VETERAN-EAST"
df.loc[(df["NEW_EXP_LEVEL"] == "VETERAN") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "VETERAN-WEST"

# Other variables that we can derive according to the performance metrics in the season:
df["NEW_HIT_RATE"] = df["HITS"] / df["ATBAT"]
df["NEW_TOUCHER"] = df["ASSISTS"] / df["PUTOUTS"]
df["NEW_RUNNER"] = df["RBI"] / df["HITS"]
df["NEW_HIT_RUN"] = df["RUNS"] / (df["HITS"])
df["NEW_HMHITS_RATIO"] = df["HMRUN"] / df["HITS"]
df["NEW_HMATBAT_RATIO"] = df["ATBAT"] / df["HMRUN"]
df["NEW_TOTAL_CHANCES"] = df["ERRORS"] + df["PUTOUTS"] + df["ASSISTS"]

# Data Preprocessing

cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

# Missing Values

df.isnull().values.any()
df.isnull().sum()

na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]

df.dropna(inplace=True)
# There were null values in the SALARY variable, I chose not to fill in the missing data in order not to
# break the structure.

# Outliers:
# We will pull it to threshold values with suppression.
for col in num_cols:
    print(col, data_prep.check_outlier(df, col))
    data_prep.replace_with_thresholds(df, col)


# Label Encoding:
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

# One-Hot Encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = data_prep.one_hot_encoder(df, ohe_cols)
cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

# RobustScaler
for col in num_cols:
    if col != 'SALARY':
        df[col] = RobustScaler().fit_transform(df[[col]])

# Model & Prediction

y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

# Model
reg_model = LinearRegression().fit(X, y)

reg_model.intercept_
# 0.21577010059223078

reg_model.coef_
# -750.74988071,  366.45172602, -233.00304947,  483.54155479,
#  358.25339771,    8.18079362, -803.49559654, -675.80273802,
#  531.2987659 ,  -21.17295285,  168.05779447,  389.25908672,
# -105.92965151,    7.87350239,  -41.98858441,   27.30649855,
#  15.14945411,  -46.83020113,   32.63744784, -271.77120964,
#  138.63558234, -211.21801668,  116.46723232,   21.33568296,
# -46.63473383,  -28.1973859 ,  111.35784934,  -71.5279651 ,
# -46.53518558,    3.76335406, -454.61405985,  204.08094032,
#  237.25047189,  122.96430493, -213.74347729,  113.86640034,
# -136.40338928,   70.31400937, -184.57164794,  -68.46387004,
#  149.98020766,   15.85433786,   20.57363948, -236.15511871,
# -45.11529751,    0.        ,  281.27041622, -150.69163168,
# -85.46348703,   26.63562255,  -71.75092006,  166.04459354,
#  115.22582267

# Prediction
y_pred = reg_model.predict(X)

# Model Evaluation & K-Fold Cross Validation
# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 172.18139420312636

# R-SQUARED
reg_model.score(X, y)
# 0.8488211175135632
# Our independent variables can explain 84.88% of the variance in the SALARY variable.

# 10-Fold RMSE
# Since the dataset is not large enough, we perform cv on the entire dataset without separating it as a test-train.
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))
# 249.71828169036507

# PREDICT:
random_user = X.sample(1, random_state=50)
reg_model.predict(random_user)
# 202.01585413
