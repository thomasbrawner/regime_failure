import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample 


class DataFormatter(object): 
    def __init__(self, dframe, depvar):
        self.y = dframe.pop(depvar).values 
        self.feature_names = dframe.columns.tolist() 
        if 'year' in self.feature_names:
            self.years = dframe['year'].values 

    def set_specification(self, lag, factors=None): 
        lag = [lag]
        regimes = ['military', 'personal', 'party', 'institutions']
        controls = ['duration', 'gdppc', 'growth', 'resource']
        dummies = []
        if factors is not None: 
            for factor in factors: 
                dummies += [col for col in self.feature_names if factor in col]
        self.specification = lag + regimes + controls + dummies 

    def format_features(self, scale=False): 
        if not hasattr(self, 'specification'): 
            raise Exception('Need to set specification before formatting features')     
        if scale: 
            self.X = StandardScaler().fit_transform(dframe[self.specification].values)
        else: 
            self.X = dframe[self.specification].values

def optimal_l2(X, y): 
    # Find the optimal level of L2 regularization 
    logit = LogisticRegressionCV(Cs=50, cv=10)
    logit.fit(X, y)
    return logit.C_ 

