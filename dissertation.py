import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample 


class DissertationData(object): 
    def __init__(self, dframe, depvar): 
        self.y = dframe.pop(depvar).values
        self.X = dframe.values 
        self.feature_names = dframe.columns.tolist() 

    def scale(self): 
        return StandardScaler().fit_transform(self.X)


