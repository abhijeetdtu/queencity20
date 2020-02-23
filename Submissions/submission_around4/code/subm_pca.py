%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from queencity20.utils.getData import *
from queencity20.utils.remove_correlated import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from collections import defaultdict
df = getTrainingData()
df.head()

allButTarget = [c for c in df.columns if c != "target"]
means = df.mean(skipna=True)
fdf = df.fillna(means)

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
fdf.loc[:,allButTarget] = pt.fit_transform(fdf.loc[:,allButTarget])

fdf = diffCols(fdf)

#corcols = list(set(find_correlation(fdf.drop("target" , axis=1), threshold=0.9)))
#fdf = fdf.drop(corcols , axis=1)

from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

fdf = fdf[~np.any(np.logical_or(X > 3 , X < -3) , axis=1)]
X = fdf.drop(["target"], axis=1)
y = fdf["target"]


#X[~np.any(np.logical_or(X > 3 , X < -3) , axis=1)].shape
# weights = np.abs(fdf.corr()["target"])[:-1]
# class_weights_dict = {}
#
# for i,c in enumerate(list(weights.index)):
#     class_weights_dict[c] = weights.values[i]


from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
X.shape

exv = pd.DataFrame({"var" : np.cumsum(pca.explained_variance_ratio_) , "comp" :np.arange(0,X.shape[1]) })

from plotnine import *
ggplot(exv , aes(x="comp" , y="var")) + geom_point()

pca = PCA(n_components=60)
X_new = pca.fit_transform(X)


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10 , max_depth=2, max_features=7 , min_samples_split=2)

from sklearn.model_selection import cross_val_score
cross_val_score(rfr , X_new,y ,scoring="neg_mean_squared_error" , cv=20)


param_grid = [
  {"n_estimators":[2,3,4,5,6,10,12] ,'max_depth': [ 2,3,4,5] ,'max_features': [ 6,7,8,9,10,11,12,13,20,25 , 30,35,40]  , "min_samples_split":[2,3,4]  },
   {"n_estimators":[50,80,90,100] ,'max_depth': [ 2,3,4,5] ,'max_features': [ 0.1 ,0.2,0.5,0.8,0.75,0.85,0.9]}

 ]


rssv = RandomizedSearchCV(rfr , param_grid,cv=5 , scoring="neg_mean_squared_error")
rssv.fit(X_new,y)
rssv.best_params_
rssv.best_score_

rfr = RandomForestRegressor(n_estimators=4, min_samples_split=4, max_features=9 , max_depth=5)
rfr.fit(X_new,y)
cross_val_score(rfr , X_new,y ,scoring="neg_mean_squared_error" , cv=20)

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR


svm = SVR(C=0.001 , kernel="poly" ,degree=10)
cross_val_score(svm , X_new,y ,scoring="neg_mean_squared_error" , cv=20)

param_grid = [
  {"degree":[2,3,4,5,6,10] ,'C': [ 0.01 , 0.1,0.2,0.3,0.5,0.6,2,5 ,0.001 , 0.0001]},
 ]


rssv = RandomizedSearchCV(svm , param_grid,cv=5 , scoring="neg_mean_squared_error")
rssv.fit(X,y)
rssv.best_params_
rssv.best_score_


rfr.fit(X_new,y)
testData = getTestData()
testData = testData.fillna(testData.mean(skipna=True))
testData.loc[:,testData.columns] = pt.transform(testData.loc[:,testData.columns])
testData = diffCols(testData)

testData = pca.transform(testData)
#testData = testData.drop(corcols , axis=1)
preds = rfr.predict(testData)

pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
