%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from queencity20.utils.getData import *
from queencity20.utils.remove_correlated import *

from collections import defaultdict
df = getTrainingData()
df.head()


means = df.mean(skipna=True)
fdf = df.fillna(means)

#fdf = diffCols(fdf)

fdf["target"].describe()

def targetToCat_(x):
    if x < np.quantile(df["target"] , 0.25):
        return "low"
    if x < np.quantile(df["target"] , 0.75):
        return "medium"
    if x < np.quantile(df["target"] , 0.99):
        return "high"
    return "exceptionally high"

def targetToCat(x):
    if x < 0:
        return "negative"
    return "positive"

fdf["target_cat"] = fdf["target"].apply(targetToCat)
fdf["target_cat"] = fdf["target_cat"].astype("category")

fdf["target_cat"].value_counts()

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
#X_train, X_test, y_train, y_test = testTrainSplit(fdf)
X = fdf.drop(["target" , "target_cat"], axis=1)
y = fdf["target_cat"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42 , stratify=y)

#class_weight={"exceptionally high":1, "high":1,"low":1,"medium":25 }
rfc = RandomForestClassifier(criterion="entropy" , n_estimators=10)
rfc.fit(X_train , y_train)

from sklearn.svm import SVC

svm = SVC(tol=0.00001 , C=0.1 , kernel="poly")
svm.fit(X_train , y_train)


from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, accuracy_score , confusion_matrix


def score(model):
    print(accuracy_score(y_train , model.predict(X_train)))
    print(accuracy_score(y_test , model.predict(X_test)))


score(svm)


#fdf["target_is_positive"] = svm.predict(X)
#fdf["target_is_positive"] = fdf["target_is_positive"].astype("category")
#fdf = pd.get_dummies(fdf , columns=["target_is_positive"] , drop_first=True)

cormat = fdf.corr()

cormat["target"].sort_values(ascending=False).head(20)
np.abs(cormat["target"]).sort_values(ascending=False).head(20).index
corcols = list(set(find_correlation(fdf.drop("target" , axis=1), threshold=0.8)))
len(corcols)
fdf = fdf.drop(corcols , axis=1)
cormat.describe()
ggplot(cormat , aes(x) )


X = fdf.drop(["target" , "target_cat"], axis=1)
y = fdf["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestRegressor
#rfr = RandomForestRegressor(max_depth=3, n_estimators=155)
rfr = RandomForestRegressor(bootstrap=True,max_features=0.8, max_depth=3)
rfr.fit(X_train , y_train)

for col in X_train.columns:
    if col.find("target_is_positive") >= 0:
        print(col)




featureImpDf = pd.DataFrame({"feature" : X_train.columns , "imp":rfr.feature_importances_})
featureImpDf.sort_values("imp" , ascending=False).head(20)["feature"].values


r2_score(y_test, rfr.predict(X_test))



testData = getTestData()
testData = testData.fillna(testData.mean(skipna=True))
#testData = diffCols(testData)
testData = testData.drop(corcols , axis=1)
preds = rfr.predict(testData)


pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
