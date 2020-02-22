import pandas as pd
import pathlib
import os

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


def testTrainSplit(df , splitRatio = 0.77):
    from sklearn.model_selection import train_test_split
    df.columns
    X = df.drop("target" , axis=1)
    y = df["target"]
    #X_train, X_test, y_train, y_test =
    return train_test_split(X, y, test_size=1-splitRatio, random_state=42)
