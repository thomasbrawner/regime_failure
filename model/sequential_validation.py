from __future__ import print_function 
import dissertation_plots as dp
import itertools
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample 


class SequentialValidationPrediction(object): 
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

    def evaluate_model(self, metric=roc_auc_score): 
        self.scores = []
        self.param_grid = self.make_param_grid()
        for parameters in self.param_grid:
            self.model.set_params(**parameters)
            param_scores = []
            for yr in np.unique(self.years)[np.unique(self.years) > 1960]: 
                x_train, y_train, x_test, y_test = self.make_split(yr)
                self.model.fit(x_train, y_train)
                preds = self.model.predict_proba(x_test)[:, 1]
                try:
                    param_scores.append(metric(y_test, preds))
                except: 
                    pass 
            self.scores.append(np.mean(param_scores))
        self.optimal_params = self.param_grid[np.argmax(self.scores)]

    def bootstrap_estimates(self, n_boot=100): 
        self.model.set_params(**self.optimal_params)
        ests = [np.hstack([self.model.fit(iX, iy).coef_.ravel(), self.model.fit(iX, iy).intercept_])
                for iX, iy in (resample(self.X, self.y) for _ in xrange(n_boot))] 
        self.boot_estimates = np.vstack(ests)

    def predict(self):
        self.model.set_params(**self.optimal_params)
        data, probs = [], []
        for yr in np.unique(self.years)[np.unique(self.years) > 1960]:
            x_train, y_train, x_test, y_test = self.make_split(yr)
            self.model.fit(x_train, y_train)
            probs.append(self.model.predict_proba(x_test)[:, 1])
            data.append(y_test)
        self.y_test = np.concatenate(data)
        self.probabilities = np.concatenate(probs)
