%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from queencity20.utils.getData import *
from queencity20.utils.remove_correlated import *

from collections import defaultdict
df = getTrainingData()
df = pd.read_csv("queen.csv")
df.head()
df.shape

df= df.drop("Unnamed: 0" , axis=1)
#df = getTestData()
df = diffCols(df)

df = df.fillna(df.mean(skipna=True))

df.shape
df = df.loc[:416 , :]
test = df.loc[416: , :]

test.shape
from plotnine import *

xCols = [c for c in df.columns if c != "target"]
yCol = "target"

ndf = df.copy()
ndf.shape

from sklearn.preprocessing import StandardScaler

#ndf = ndf.fillna(ndf.mean(skipna=True))
ndf.loc[:,xCols] = StandardScaler().fit_transform(ndf.loc[:,xCols])

from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

X = ndf.drop(yCol, axis=1)
y = ndf[yCol]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

svm = SVR()
svm.fit(X_train,y_train)

def score(model):
    print(r2_score(y_train , model.predict(X_train)))
    print(r2_score(y_test , model.predict(X_test)))


score(svm)


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#rfr = RandomForestRegressor(n_estimators=5, max_samples=0.8 , max_features=30,ccp_alpha = 0.4,min_samples_split=4, max_depth=5)
#rfr = RandomForestRegressor(n_estimators=50, max_samples=0.2 , max_features=0.7,ccp_alpha = 0.4,min_samples_split=4, max_depth=5)
rfr = RandomForestRegressor()
rfr.fit(X,y)
cross_val_score(rfr , X,y ,scoring="neg_mean_squared_error" , cv=10)


preds = rfr.predict(test.drop("target" , axis=1))

pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
