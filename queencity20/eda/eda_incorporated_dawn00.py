%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from queencity20.utils.getData import *
from queencity20.utils.remove_correlated import *

from collections import defaultdict
df = getTrainingData()
df.head()

df["target"].describe()
df.describe()

df = df.fillna(df.mean(skipna=True))

allButTarget = [c for c in df.columns if c != "target"]
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
df.loc[:,allButTarget] = pt.fit_transform(df.loc[:,allButTarget])

X = df.drop("target" , axis=1)
y = df["target"]


from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_new = sel.fit_transform(X)
X_new.shape


from sklearn.feature_selection import SelectKBest , SelectPercentile
from sklearn.feature_selection import chi2 , mutual_info_regression

#X_new = SelectPercentile(mutual_info_regression, percentile=80).fit_transform(X_new, y)
#X_new.shape

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_new)
X_new.shape


exv = pd.DataFrame({"var" : np.cumsum(pca.explained_variance_ratio_) , "comp" :np.arange(0,X_new.shape[0]) })

from plotnine import *
ggplot(exv , aes(x="comp" , y="var")) + geom_point()


pca = PCA(n_components=50)
X_new = pca.fit_transform(X_new)


from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, accuracy_score , confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.8, random_state=42)

#ggplot(pd.DataFrame({"x" : X_train , "y":y_train}) , aes(x="x" , y="y")) + geom_point()

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth=3)
rfr.fit(X_train , y_train)


#featureImpDf = pd.DataFrame({"feature" : X_train.columns , "imp":rfr.feature_importances_})
#featureImpDf.sort_values("imp" , ascending=False).head(30)["feature"].values


r2_score(y_test, rfr.predict(X_test))



from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR

param_grid = [
  {"max_samples" : [0.2 , 0.5 , 0.8,0.9],"ccp_alpha":[  0.3 ,0.4 ,0.5, 1 , 2], "n_estimators":[2,3,4,5,6,10,12,50,100,200] ,'max_depth': [ 2,3,4,5] ,'max_features': [ 0.1 , 0.2 , 0.3 , 0.4 ,0.7 , 0.9]   },
   {"max_samples" : [0.2 , 0.5 , 0.8, 0.9],"ccp_alpha":[  0.3 ,0.4 ,0.5, 1 , 2],"n_estimators":[50,80,90,100] ,'max_depth': [ 2,3,4,5] ,'max_features': [ 0.1 , 0.2 , 0.3 , 0.4 ,0.7 , 0.9]}

 ]


rssv = RandomizedSearchCV(rfr , param_grid,cv=10 , scoring="neg_mean_squared_error")
rssv.fit(X_new,y)
rssv.best_params_
rssv.best_score_


from sklearn.model_selection import cross_val_score
rfr = RandomForestRegressor(max_depth=3)
rfr.fit(X_new,y)
cross_val_score(rfr , X_new,y ,scoring="neg_mean_squared_error" , cv=20)


from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR

param_grid = [
  {"rfr__max_depth":[2,3,4,5,6] ,'svm__C': [ 0.03 ,0.04,0.05, 0.09], 'svm__kernel': ['rbf' , "sigmoid"]},
 ]

estimators = [
    ('ridge' , RidgeCV()),
    ('svm', SVR()),
    ('rfr', RandomForestRegressor( n_estimators=100,max_depth=4,random_state=42))
]

stacking = StackingRegressor(estimators=estimators)


from sklearn.model_selection import RandomizedSearchCV

rssv = RandomizedSearchCV(stacking , param_grid,cv=5 , scoring="neg_mean_squared_error")
rssv.fit(X,y)
rssv.best_params_
rssv.best_score_

cross_val_score(stacking , X_new,y ,scoring="neg_mean_squared_error" , cv=10)


testData = getTestData()
testData = testData.fillna(testData.mean(skipna=True))
testData = diffCols(testData)
testData.loc[:,:] = pt.transform(testData)
testData = testData.drop(corcols , axis=1)
preds = rfr.predict(testData)


pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
