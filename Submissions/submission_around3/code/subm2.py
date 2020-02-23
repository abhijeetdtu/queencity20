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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


X = fdf.drop(["target"], axis=1)
y = fdf["target"]

X = scaler.fit_transform(X)

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

pca = PCA(n_components=25)
X_new = pca.fit_transform(X)


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100 , max_depth=3 , max_features=0.8)

from sklearn.model_selection import cross_val_score
cross_val_score(rfr , X_new,y ,scoring="neg_mean_squared_error" , cv=10)

rfr.fit(X_new,y)
testData = getTestData()
testData = testData.fillna(testData.mean(skipna=True))
testData = diffCols(testData)
testData = scaler.transform(testData)
testData = pca.transform(testData)
#testData = testData.drop(corcols , axis=1)
preds = rfr.predict(testData)

pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
