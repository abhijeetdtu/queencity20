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

corcols = list(set(find_correlation(fdf.drop("target" , axis=1), threshold=0.9)))
len(corcols)
#fdf = fdf.drop(corcols , axis=1)

fdf = fdf[corcols + ["target"]]
fdf = diffCols(fdf)

#corcols = list(set(find_correlation(fdf.drop("target" , axis=1), threshold=0.9)))
#fdf = fdf.drop(corcols , axis=1)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

X = fdf.drop(["target"], axis=1)
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

pca = PCA(n_components=50)
X_new = pca.fit_transform(X)

X_new.shape

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=5, max_samples=0.8 , max_features=30,ccp_alpha = 0.4,min_samples_split=4, max_depth=5)
rfr.fit(X_new,y)

from sklearn.ensemble import AdaBoostRegressor

adb = AdaBoostRegressor(base_estimator=rfr , n_estimators=100 , learning_rate=0.001 , loss="square")

cross_val_score(adb , X_new,y ,scoring="neg_mean_squared_error" , cv=20)

adb.fit(X_new , y)

testData = getTestData()
testData = testData.fillna(testData.mean(skipna=True))
testData.loc[:,testData.columns] = pt.transform(testData.loc[:,testData.columns])
#testData = diffCols(testData)
testData = testData[corcols]
testData = diffCols(testData)
testData = pca.transform(testData)
#testData = testData.drop(corcols , axis=1)
preds = adb.predict(testData)

pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
