%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from queencity20.utils.getData import *
from queencity20.utils.remove_correlated import *

from collections import defaultdict
df = getTrainingData()
df.head()

#df = getTestData()
df = diffCols(df)
cols = ["target" , 'Residential_Demolitions_min', 'Rental_Houses_2018_trend_mag',
       'Commercial_Construction_Permitted_Units_min',
       'Transit_Ridership_Total_2015_trend_mag',
       '311_Requests_2016_trend_mag', 'Street_Connectivity_max',
       'Disorder_Calls_2018_trend_mag', 'Housing_Age_2018_trend_mag',
       'Population_Density_2018_trend_mag',
       'Commercial_Construction_mean', 'Fire_Calls_max',
       'Population _min', 'Housing_Age_max', 'Sidewalk_Availability_2015',
       'Pharmacy_Proximate_Units_2018_trend_mag',
       'Subsidized_Housing_Units_max', 'Residential_Demolitions_mean',
       'Fincancial_Services_Proximity_max',
       'Neighborhood_School_Attendance_2017_trend_mag',
       'Commercial_Size_Total_max', 'Recycling_Participation_2013',
       'Adopt_a_Street_Length_2017',
       'SchoolAge_Proximate_Units_2017_trend_mag',
       'Residential_Occupancy_2017',
       'Commercial_Construction_2018_trend_mag',
       'Residential_Tree_Canopy_2012',
       'Subsidized_Housing_2017_trend_mag',
       'Commercial_Construction_Permitted_Units_max',
       'Student_Absenteeism_2017_trend_mag',
       'Adopt_a_Stream_Length_2017_trend_mag',
       'Fire_Calls_2018_trend_mag', 'Park_Proximate_Units_2018_trend_mag',
       'Single_Family_Units_2018_trend_mag',
       'Proficiency_Elementary_School_2017_trend_mag',
       'Home_Ownership_2017', 'Impervious_Surface_min',
       'Voter_Participation_2018_trend_mag', 'Housing_Density_max',
       'Commercial_Building_Age_max', 'Rental_Costs_2017',
       'Proficiency_High_School_2017_trend_mag',
       'Pharmacy_Proximity_2018_trend', 'Grocery_Proximity_mean',
       'Foreclosures_2017_trend_mag', 'Commercial_Size_2018_trend',
       'Population _2018_trend', 'Commercial_Size_max',
       'Property_Crimes_2018_trend_mag', 'Impervious_Surface_mean',
       'Nuisance_Violations_Total_mean',
       'Housing_Violations_2018_trend_mag',
       'Financial_Services_Proximate_Units_2018_trend_mag',
       'Public_Health_Insurance _2017_trend_mag',
       'Commercial_Building_Age_2018_trend_mag',
       'Highschool_Graduation_Rate_mean',
       'Low_Cost_Healthcare_Proximate_Units_max',
       'Animal_Control_Call_Rate_mean',
       'Bicycle_Friendliness_2018_trend_mag',
       'Solid_Waste_Diversion_Rate_2013', 'Housing_Density_2018_trend',
       'Natural_Gas_Consumption_2013',
       'Street_Connectivity_2018_trend_mag', 'Foreclosed_Units_mean',
       'Financial_Services_Proximate_Units_max',
       'Public_Nutrition_Assistance_2018_trend_mag',
       'Impervious_Surface_2018_trend_mag',
       'Pharmacy_Proximate_Units_2018_trend',
       'Low_Birthweight_2016_trend', 'Disorder_Calls_2018_trend',
       'Pharmacy_Proximity_mean', 'Impervious_Surface_Area_2018_trend',
       'Population _max', 'Nuisance_Violations_Total_2016_trend',
       'Neighborhood_School_Attendance_2017_trend',
       'Home_Sales_Price_2015_trend', '311_Requests_max',
       'Prenatal_Care_2016_trend', 'Violent_Crimes_2018_trend_mag',
       'Population_Density_max', '311_Calls_max',
       'Pharmacy_Proximate_Units_max', 'Housing_Age_2018_trend',
       'Property_Crime_Rate_2018_trend_mag', 'Adopt_a_Stream_Length_mean',
       'Bachelors_Degree_moe_2017', 'Street_Connectivity_2018_trend',
       'Animal_Control_Call_Rate_2018_trend', 'Voters_Participating_min',
       'Proficiency_Middle_School_2017_trend_mag', 'Adopt_a_Street_2017',
       'Early_Care_Proximate_Units_2017_trend_mag',
       'Highschool_Graduation_Rate_max', 'Transit_Ridership_max',
       'Commercial_Construction_Permitted_Units_2018_trend',
       'High_School_Diploma_moe_2017',
       'Violent_Crime_Rate_2018_trend_mag',
       'Low_Birthweight_2016_trend_mag', 'Residential_Canopy_Area_2012',
       'Electricity_Consumption_Total_2013_trend_mag',
       'Arts_Participation_2013']




# for test data
df = df[cols]
df = df.fillna(df.mean(skipna=True))

from plotnine import *

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


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#rfr = RandomForestRegressor(n_estimators=5, max_samples=0.8 , max_features=30,ccp_alpha = 0.4,min_samples_split=4, max_depth=5)
rfr = RandomForestRegressor(n_estimators=50, max_samples=0.2 , max_features=0.7,ccp_alpha = 0.4,min_samples_split=4, max_depth=5)
rfr.fit(X,y)
cross_val_score(rfr , X,y ,scoring="neg_mean_squared_error" , cv=10)


rfr.predict(X)
