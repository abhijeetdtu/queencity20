import pandas as pd
import pathlib
import os
from collections import defaultdict

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


def diffCols(df):
    colsims = defaultdict(list)
    for col in df:
        if "_" in col:
            pref = "_".join(col.split("_")[:-1])
            colsims[pref].append(col)

    for key,val in colsims.items():
        if len(val) > 1:
            df[f"{key}_min"] = df[val].min(axis=1)
            df[f"{key}_max"] = df[val].max(axis=1)
            df[f"{key}_mean"] = df[val].mean(axis=1)
            df[f"{key}_std"] = df[val].std(axis=1)
            df = df.drop(val , axis=1)

    len(df.columns)
    return df

df = getTrainingData()
def yoyCols(df):
    colsims = defaultdict(list)
    for col in df:
        if "_" in col:
            pref = "_".join(col.split("_")[:-1])
            colsims[pref].append(col)

    for key,val in colsims.items():
        if len(val) > 1:
            diffs = []
            for i,v in enumerate(val[:-1]):
                (df[v]-df[val[i+1]]).min(axis=1)
                df[f"{key}_max"] = df[val].max(axis=1)
                df[f"{key}_mean"] = df[val].mean(axis=1)
                df = df.drop(val , axis=1)

    len(df.columns)
    return df
