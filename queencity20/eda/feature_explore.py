%load_ext autoreload
%autoreload 2

import pandas as pd

from queencity20.utils.getData import *

df = getTrainingData()
df.head()


from plotnine import *

ggplot(df , aes(x = "target")) + geom_histogram()


df.describe()

X_train, X_test, y_train, y_test = testTrainSplit(df)
from sklearn.ensemble import RandomForestRegressor


rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
