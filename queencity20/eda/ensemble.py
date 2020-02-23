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


means = df.mean(skipna=True)
fdf = df.fillna(means)

fdf = diffCols(fdf)
corcols = list(set(find_correlation(fdf.drop("target" , axis=1), threshold=0.8)))
fdf = fdf.drop(corcols , axis=1)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


X = fdf.drop(["target"], axis=1)
y = fdf["target"]

X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor
sdg = DecisionTreeRegressor(max_depth=1 , max_features=0.6 , max_leaf_nodes=100)
adb = AdaBoostRegressor(base_estimator=sdg , n_estimators=3 , learning_rate=0.3 , loss="square")
adb.fit(X_train , y_train)

from sklearn.metrics import r2_score

def score(model ):
    print(r2_score(y_train , model.predict(X_train)))
    print(r2_score(y_test , model.predict(X_test)))

score(adb)


from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR

param_grid = [
  {"rfr__max_depth":[2,3,4,5,6] ,'svm__C': [ 0.03 ,0.04,0.05, 0.09], 'svm__kernel': ['rbf' , "sigmoid"]},
 ]

estimators = [
    ('svm', SVR()),
    ('rfr', RandomForestRegressor( n_estimators=10,max_depth=4,random_state=42))
]

stacking = StackingRegressor(estimators=estimators)
#stacking.fit(X_train , y_train)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score
cross_val_score(stacking , X,y ,scoring="neg_mean_squared_error" , cv=10)


rssv = RandomizedSearchCV(stacking , param_grid,cv=5 , scoring="neg_mean_squared_error")
rssv.fit(X,y)
rssv.best_params_
rssv.best_score_
