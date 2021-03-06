import pandas as pd
import pathlib
import os
from collections import defaultdict

import numpy as np

class Paths:

    TRAINING = "training.csv"
    TESTING = "testing.csv"
    DICTIONARY = "data description.csv"

def getPathToData():
    try:
        __file__
        path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    except:
        path = pathlib.Path().resolve()
    dataPath = os.path.abspath(os.path.join(path ,"queencity20", "data"))
    return dataPath

def getPathToDataFile(filename):
    path = getPathToData()
    return os.path.abspath(os.path.join(path , filename))

def getTrainingData():
    df = pd.read_csv(getPathToDataFile(Paths.TRAINING))
    df = df.drop("Unnamed: 0" , axis=1)
    return df


def getTestData():
    df = pd.read_csv(getPathToDataFile(Paths.TESTING))
    df = df.drop("Unnamed: 0" , axis=1)
    return df

def testTrainSplit(df , splitRatio = 0.77):
    from sklearn.model_selection import train_test_split
    df.columns
    X = df.drop("target", axis=1)
    y = df["target"]
    #X_train, X_test, y_train, y_test =
    return train_test_split(X, y, test_size=1-splitRatio, random_state=42)

#df = getTrainingData()
#df = df.fillna(df.mean(skipna=True))

def getTrend(x):
    arr = x.values
    updown = []
    mags = []
    for i,elem in enumerate(arr[:-1]):
        diff = arr[i] - arr[i+1]
        #x[f"{x.index[0]}_trend_mag_{i}_{i+1}"] = diff
        #x[f"{x.index[0]}_trend_updown_{i}_{i+1}"] = 1 if diff > 0 else -1

        updown.append(1 if diff > 0 else -1)
        mags.append(diff)

    x[f"{x.index[0]}_trend"] = np.sum(updown)
    x[f"{x.index[0]}_trend_mag"] = np.sum(mags)
    return x

def diffCols(df):
    colsims = defaultdict(list)
    for col in df:
        if "_" in col:
            pref = "_".join(col.split("_")[:-1])
            colsims[pref].append(col)
    for key,val in colsims.items():
        if len(val) > 1:
            #ndf = df.loc[:,val].apply(lambda x: getTrend(x), axis=1)
            #ndf = ndf.drop(val , axis=1)
            df[f"{key}_min"] = df[val].min(axis=1)
            df[f"{key}_max"] = df[val].max(axis=1)
            df[f"{key}_mean"] = df[val].mean(axis=1)
            df = df.drop(val , axis=1)
            #df = df.join(ndf)
    return df
