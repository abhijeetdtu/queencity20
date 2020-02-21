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
        path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    except:
        path = pathlib.Path().resolve()
    dataPath = os.path.abspath(os.path.join(path , "data"))
    return dataPath

def getPathToDataFile(filename):
    path = getPathToData()
    return os.path.abspath(os.path.join(path , filename))

def getTrainingData():
    df = pd.read_csv(getPathToDataFile(Paths.TRAINING))
    return df
