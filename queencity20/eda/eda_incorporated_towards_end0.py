%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from queencity20.utils.getData import *
from queencity20.utils.remove_correlated import *

from collections import defaultdict
df = getTrainingData()
df.head()


from sklearn.impute import SimpleImputer
#means = df.mean(skipna=True)
si = SimpleImputer(strategy="median")

df.loc[:,:] = si.fit_transform(df)

fdf = df

fdf = diffCols(fdf)

fdf["target"].describe()

fdf.shape

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
#X_train, X_test, y_train, y_test = testTrainSplit(fdf)
X = fdf.drop(["target"], axis=1)
y = fdf["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#class_weight={"exceptionally high":1, "high":1,"low":1,"medium":25 }

from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, accuracy_score , confusion_matrix

cormat = fdf.corr()

cormat["target"].sort_values(ascending=False).head(20)
np.abs(cormat["target"]).sort_values(ascending=False).head(20).index
corcols = list(set(find_correlation(fdf.drop("target" , axis=1), threshold=0.8)))
len(corcols)
fdf = fdf.drop(corcols , axis=1)


X = fdf.drop(["target"], axis=1)
y = fdf["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth=3)
rfr.fit(X_train , y_train)



featureImpDf = pd.DataFrame({"feature" : X_train.columns , "imp":rfr.feature_importances_})
featureImpDf.sort_values("imp" , ascending=False).head(20)["feature"].values


r2_score(y_test, rfr.predict(X_test))


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#rfr = RandomForestRegressor(n_estimators=5, max_samples=0.8 , max_features=30,ccp_alpha = 0.4,min_samples_split=4, max_depth=5)
rfr = RandomForestRegressor(n_estimators=50, max_samples=0.2 , max_features=0.7,ccp_alpha = 0.4,min_samples_split=4, max_depth=5)
rfr.fit(X,y)
cross_val_score(rfr , X,y ,scoring="neg_mean_squared_error" , cv=10)


testData = getTestData()
testData.loc[: , :] = si.fit_transform(testData)
#testData = testData.fillna(testData.mean(skipna=True))
testData = diffCols(testData)
testData = testData.drop(corcols , axis=1)
preds = rfr.predict(testData)


pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
