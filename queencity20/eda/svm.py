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

fdf = diffCols(fdf)

corcols = list(set(find_correlation(fdf.drop("target" , axis=1), threshold=0.8)))

fdf = fdf.drop(corcols , axis=1)
X = fdf.drop(["target"], axis=1)
y = fdf["target"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42 )


from sklearn.svm import SVR

svm = SVR(C=0.1 , kernel="rbf")
svm.fit(X_train , y_train)


from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, accuracy_score , confusion_matrix


def score(model):
    print(r2_score(y_train , model.predict(X_train)))
    print(r2_score(y_test , model.predict(X_test)))

score(svm)


testData = getTestData()
testData = testData.fillna(testData.mean(skipna=True))
testData = diffCols(testData)
testData = testData.drop(corcols , axis=1)
preds = svm.predict(testData)


pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
