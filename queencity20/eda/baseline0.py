%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from queencity20.utils.getData import *

from collections import defaultdict
df = getTrainingData()
df.head()


from plotnine import *

ggplot(df , aes(x = "target")) + geom_histogram()


df.isnull().any(axis=1)

df.describe()


means = df.mean(skipna=True)
fdf = df.fillna(means)
saveDf = fdf.copy()

df = fdf.copy()
#df = saveDf



def diffCols(df):
    colsims = defaultdict(list)
    for col in df:
        if "_" in col:
            pref = "_".join(col.split("_")[:-1])
            colsims[pref].append(col)

    for key,val in colsims.items():
        if len(val) > 1:
            df[f"{key}_min"] = df[val].min(axis=1)
            df[f"{key}_max"] = df[val].max(axis=1)
            df[f"{key}_mean"] = df[val].mean(axis=1)
            df = df.drop(val , axis=1)

    len(df.columns)


df.head()
fdf = df
X_train, X_test, y_train, y_test = testTrainSplit(fdf)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train , y_train)

r2_score(y_test , linreg.predict(X_test))


mean_squared_error(y_test , linreg.predict(X_test))

rfr = RandomForestRegressor(n_estimators=5 , min_samples_leaf=5 , max_features=0.3)
rfr.fit(X_train,y_train)
rfr.score(X_train,y_train)
mean_squared_error(y_test , rfr.predict(X_test))
r2_score(y_test , rfr.predict(X_test))


featureImpDf = pd.DataFrame({"feature" : X_train.columns , "imp":rfr.feature_importances_})
featureImpDf.sort_values("imp" , ascending=False).head(20)

ggplot(featureImpDf , aes(x = "feature" , y="imp")) +
