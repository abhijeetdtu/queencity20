%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from queencity20.utils.getData import *
from queencity20.utils.remove_correlated import *

from collections import defaultdict
df = getTrainingData()
df.head()

nacounts = df.isna().sum()

nacounts = (1- nacounts / df.shape[0])

ranks = np.abs(df.corr().loc["target"])*nacounts

ranks = ( ranks-ranks.mean() ) / ranks.std()
ranks.sort_values(ascending=False).std()

maybeCols = list(ranks[ranks > 0].index)

fdf = df[maybeCols]
means = fdf.mean(skipna=True)
fdf = fdf.fillna(means)




from sklearn.preprocessing import PowerTransformer

fdf = diffCols(fdf)
allButTarget = [c for c in fdf.columns if c != "target"]
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, accuracy_score , confusion_matrix


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
fdf.loc[:,allButTarget] = pt.fit_transform(fdf.loc[:,allButTarget])


X = fdf.drop(["target"], axis=1)
y = fdf["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth=3)
rfr.fit(X_train , y_train)


featureImpDf = pd.DataFrame({"feature" : X_train.columns , "imp":rfr.feature_importances_})
featureImpDf.sort_values("imp" , ascending=False).head(20)["feature"].values


r2_score(y_test, rfr.predict(X_test))



from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR

param_grid = [
  {"max_samples" : [0.2 , 0.5 , 0.8,0.9],"ccp_alpha":[  0.3 ,0.4 ,0.5, 1 , 2], "n_estimators":[2,3,4,5,6,10,12] ,'max_depth': [ 2,3,4,5] ,'max_features': [ 0.1 , 0.2 , 0.3 , 0.4 ,0.7 , 0.9]  , "min_samples_split":[2,3,4]  },
   {"max_samples" : [0.2 , 0.5 , 0.8, 0.9],"ccp_alpha":[  0.3 ,0.4 ,0.5, 1 , 2],"n_estimators":[50,80,90,100] ,'max_depth': [ 2,3,4,5] ,'max_features': [ 0.1 , 0.2 , 0.3 , 0.4 ,0.7 , 0.9]}

 ]


rssv = RandomizedSearchCV(rfr , param_grid,cv=10 , scoring="neg_mean_squared_error")
rssv.fit(X,y)
rssv.best_params_
rssv.best_score_



testData = getTestData()
testData = testData.fillna(testData.mean(skipna=True))
testData = diffCols(testData)
testData = testData.drop(corcols , axis=1)
preds = rfr.predict(testData)


pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
