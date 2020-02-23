cols = ['Residential_Demolitions_min', 'Sidewalk_Availability_2015',
       'Street_Connectivity_max', 'Fire_Calls_max',
       'Commercial_Construction_Permitted_Units_min',
       'Commercial_Construction_mean', 'Subsidized_Housing_Units_max',
       'Population _min', 'Housing_Age_max',
       'Residential_Demolitions_mean', 'Commercial_Size_Total_max',
       'Commercial_Construction_Permitted_Units_max',
       'High_School_Diploma_moe_2017',
       'Fincancial_Services_Proximity_max', 'Residential_Occupancy_2017',
       'Pharmacy_Proximity_mean', 'Impervious_Surface_min',
       'Residential_Demolition_Permit_Units_min',
       'Residential_Canopy_Area_2012', '311_Requests_max']

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

rfr.fit(X_train , y_train)

from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, accuracy_score , confusion_matrix


r2_score(y_test, rfr.predict(X_test))

testData = getTestData()
testData = testData.fillna(testData.mean(skipna=True))
testData = diffCols(testData)
#testData = testData.drop(corcols , axis=1)
testData = testData[cols]
preds = rfr.predict(testData)


pd.DataFrame({"pred" : preds}).to_csv("submis.csv")
