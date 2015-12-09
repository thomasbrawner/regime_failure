import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample 


class DataFormatter(object): 
    def __init__(self, dframe, depvar):
        self.data = dframe
        self.y = self.data.pop(depvar).values 
        self.feature_names = self.data.columns.tolist() 
        if 'year' in self.feature_names:
            self.years = self.data['year'].values 

    def set_specification(self, lag, factors=None): 
        lag = [lag]
        regimes = ['duration', 'military', 'personal', 'party', 'institutions']
        controls = ['gdppc', 'growth', 'resource', 'population'] 
        dummies = []
        if factors is not None: 
            for factor in factors: 
                dummies += [col for col in self.feature_names if factor in col]
        self.specification = lag + regimes + controls + dummies 

    def format_features(self, scale=False): 
        if not hasattr(self, 'specification'): 
            raise Exception('Need to set specification before formatting features')     
        if scale: 
            self.X = StandardScaler().fit_transform(self.data[self.specification].values)
        else: 
            self.X = self.data[self.specification].values

def optimal_l2(X, y): 
    # Find the optimal level of L2 regularization 
    logit = LogisticRegressionCV(Cs=50, cv=10)
    logit.fit(X, y)
    return logit.C_ 

def bootstrap_estimates(model, X, y, n_boot): 
    # coefficient estimates for n_boot bootstrap samples 
    coefs = [np.hstack([model.fit(iX, iy).intercept_, model.fit(iX, iy).coef_.ravel()])
             for iX, iy in (resample(X, y) for _ in xrange(n_boot))]
    return np.vstack(coefs)


    
