%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from queencity20.utils.getData import *
from queencity20.utils.remove_correlated import *

from collections import defaultdict
df = getTrainingData()
df.head()

df = diffCols(df)
cols = ['target', 'Transit_Ridership_Total_mean', 'Transit_Ridership_Total_max',
       'Transit_Ridership_Total_min', 'Fire_Call_Rate_min',
       'Fire_Call_Rate_mean', 'Commercial_Construction_min',
       'Fire_Call_Rate_max', 'Property_Crime_Rate_min',
       'Transit_Ridership_min', 'Transit_Ridership_mean',
       'Transit_Ridership_max', 'Disorder_Call_Rate_min',
       'Property_Crime_Rate_mean', 'Disorder_Call_Rate_mean',
       'Commercial_Construction_Permitted_Units_min', 'Disorder_Call_Rate_max',
       'Commercial_Construction_mean', 'Violent_Crime_Rate_min',
       'Commercial_Construction_Permitted_Units_mean']


from plotnine import *

(ggplot(df , aes(x="Transit_Ridership_Total_mean" , y="target"))
+ geom_point()
+ scale_x_continuous(limits=(0,5000))
+ scale_y_continuous(limits=(-.5,2))
)


xCols = [c for c in cols if c != "target"]
yCol = "target"

ndf = df[cols]
ndf.shape

from sklearn.preprocessing import StandardScaler

ndf = ndf.fillna(ndf.mean(skipna=True))
ndf.loc[:,xCols] = StandardScaler().fit_transform(ndf.loc[:,xCols])

from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

X = ndf.drop(yCol, axis=1)
y = ndf[yCol]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

svm = SVR()
svm.fit(X_train,y_train)

def score(model):
    print(r2_score(y_train , model.predict(X_train)))
    print(r2_score(y_test , model.predict(X_test)))


score(svm)
