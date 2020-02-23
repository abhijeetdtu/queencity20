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
corcols = list(set(find_correlation(fdf.drop("target" , axis=1), threshold=0.9)))
fdf = fdf.drop(corcols , axis=1)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


X = fdf.drop(["target"], axis=1)
y = fdf["target"]

X = scaler.fit_transform(X)

from sklearn.neural_network import MLPRegressor



mlp = MLPRegressor(learning_rate="invscaling" , max_iter=2000)

from sklearn.model_selection import cross_val_score
cross_val_score(mlp , X,y ,scoring="neg_mean_squared_error" , cv=10)


from sklearn.model_selection import RandomizedSearchCV

param_grid = [
  {'hidden_layer_sizes': [(2,4,6),(2,4) , (2,4,2) , (2,4,3)]}
]
rssv = RandomizedSearchCV(mlp , param_grid,cv=5 , scoring="r2")
rssv.fit(X,y)
rssv.best_params_
rssv.best_score_
