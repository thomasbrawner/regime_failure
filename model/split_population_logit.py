## -------------------------------------------------------- ##
## split_population_logit
## class and functions for estimating split population logit
## with optional regularization parameters (glmnet)
## tb 2 september 2015
## -------------------------------------------------------- ##

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.base.model import GenericLikelihoodModel

## -------------------------------------------------------- ##

class SplitPopulationLogit(GenericLikelihoodModel):
    
    def __init__(self, endog, exog = None, kZ = 0, lmbda = 0.0, alpha = 0.5, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
        super(SplitPopulationLogit, self).__init__(endog, exog, **kwds)
        self.kZ = kZ
        self.lmbda = lmbda
        self.alpha = alpha
    
    def nloglikeobs(self, params):
        return -split_population_logit_likelihood(y = self.endog, X = self.exog, kZ = self.kZ, 
                                                  lmbda = self.lmbda, alpha = self.alpha, params = params)
    
    def fit(self, start_params = None, maxiter = 10000, maxfun = 10000, **kwds):
        if start_params is None:
            start_params = np.zeros(shape = (self.exog.shape[1]))
        return super(SplitPopulationLogit, self).fit(start_params = start_params, 
                                                     maxiter = maxiter, maxfun = maxfun, **kwds)

## -------------------------------------------------------- ##

def split_population_logit_likelihood(y, X, kZ, params, lmbda = None, alpha = None):
    """
    Evaluate the log likelihood for provided vector of parameter estimates. 
    """
    # split exog to Z & X 
    Z = X[:, :kZ]
    X = X[:, kZ:]
    
    # parameters associated with Z & X
    g = params[:kZ]
    b = params[kZ:]
    
    # parameters to subject to regularization: omit intercepts
    p = np.append(g[1:], b[1:])
    
    # P(y) & P(true detection)
    p = inv_logit(np.dot(X, b))
    r = inv_logit(np.dot(Z, g))
    
    # elastic net penalty
    penalty = 0
    if (lmbda is not None) & (alpha is not None):
        l2 = (1 - alpha) * 0.5 * np.dot(p, p)
        l1 = alpha * np.sum([np.abs(i) for i in p])
        penalty = lmbda * (l2 + l1)
    
    # evaluate likelihood regularization
    loglik = y * np.log(r * p) + (1 - y) * np.log((1 - r) + r * (1 - p)) - penalty
    return loglik

## -------------------------------------------------------- ##

def set_split_population_data(Z, X, normalize = False): 
    """
    Generate the exogenous array to feed to model. Z and X should be 
    pandas data frames. Normalize if using regularization. 
    """
    # get Z dimensionality
    kZ = Z.shape[1]
    
    # variable names
    var_names = Z.columns.tolist() + X.columns.tolist()
    
    # add intercepts to variable names
    var_names.insert(kZ, 'intercept_X')
    var_names.insert(0, 'intercept_Z')
    
    # concatenate the two data frames
    X = np.concatenate((np.array(Z), np.array(X)), axis = 1)
    
    # normalize if using regularization
    if normalize:
        X = StandardScaler().fit_transform(X)
        
    # add intercept terms to Z, X
    X = np.insert(X, kZ, 1, axis = 1)
    X = np.insert(X, 0, 1, axis = 1)
    
    return X, kZ + 1, var_names
    
## -------------------------------------------------------- ##

def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

## -------------------------------------------------------- ##



