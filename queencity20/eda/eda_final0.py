cols = ['Commercial_Construction_min', 'Fire_Calls_max',
       'Board_Committee_Participation_max', 'Sidewalk_Availability_2015',
       'Residential_Demolitions_min', 'Impervious_Surface_min',
       'Natural_Gas_Consumption_2013', 'Transit_Ridership_max',
       'Tree_Canopy_2012', 'Long_Commute_2017',
       'Commercial_Construction_mean', 'Population _min',
       'Commercial_Size_Total_max', 'Disorder_Calls_max',
       'Street_Connectivity_mean', 'Violent_Crime_Rate_mean',
       'Fire_Call_Rate_max', 'Residential_Canopy_Area_2012',
       'Commercial_Construction_Permitted_Units_mean',
       'Low_Cost_Healthcare_Proximate_Units_mean']


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

fdf["target"].describe()

fdf = fdf[cols + ["target"]]
X = fdf.drop(["target"], axis=1)
y = fdf["target"]

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

rfr.fit(X_train , y_train)

from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, accuracy_score , confusion_matrix


mean_squared_error(y_test, svm.predict(X_test))

testData = getTestData()
testData = testData.fillna(testData.mean(skipna=True))
testData = diffCols(testData)
#testData = testData.drop(corcols , axis=1)
testData = testData[cols]
preds = svm.predict(testData)


pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
