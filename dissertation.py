import itertools
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
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


class SequentialFoldsClassifier(object): 
    def __init__(self, model, params, years, X, y): 
        self.model = model
        self.params = params
        self.years = years
        self.X = X
        self.y = y

    def make_split(self, year): 
        train_mask = self.years < year 
        train_x, train_y = self.X[train_mask], self.y[train_mask]
        test_mask = self.years == year
        test_x, test_y = self.X[test_mask], self.y[test_mask]
        return train_x, train_y, test_x, test_y

    def make_param_grid(self): 
        param_combos = [x for x in apply(itertools.product, self.params.values())]
        return [dict(zip(self.params.keys(), p)) for p in param_combos]

    def evaluate_model(self): 
        self.pr_scores = []
        self.roc_scores = []
        self.param_grid = self.make_param_grid()
        for parameters in self.param_grid: 
            self.model.set_params(**parameters)
            param_pr_scores = []
            param_roc_scores = []
            for yr in np.unique(self.years)[np.unique(self.years) > 1960]: 
                x_train, y_train, x_test, y_test = self.make_split(yr)
                self.model.fit(x_train, y_train)
                preds = self.model.predict_proba(x_test)[:, 1]
                param_pr_scores.append(auc_pr_curve(y_test, preds))
                try:
                    param_roc_scores.append(roc_auc_score(y_test, preds))
                except: 
                    pass 
            self.pr_scores.append(np.nanmean(param_pr_scores))
            self.roc_scores.append(np.nanmean(param_roc_scores))
        self.optimal_params = self.param_grid[np.argmax(self.roc_scores)]

        
def bootstrap_estimates(model, X, y, n_boot): 
    # coefficient estimates for n_boot bootstrap samples 
    coefs = [np.hstack([model.fit(iX, iy).intercept_, model.fit(iX, iy).coef_.ravel()])
             for iX, iy in (resample(X, y) for _ in xrange(n_boot))] 
    return np.vstack(coefs) 

def auc_pr_curve(y_true, y_pred): 
    # area under the precision-recall curve 
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision) 
