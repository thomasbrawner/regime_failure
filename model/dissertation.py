from __future__ import print_function 
import dissertation_plots as dp
import itertools
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import re 
import seaborn as sns 
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample 


class DataFormatter(object): 
    def __init__(self, dframe, depvar):
        self.data = dframe
        self.depvar = depvar
        self.y = self.data.pop(self.depvar).values 
        self.feature_names = self.data.columns.tolist() 
        if 'year' in self.feature_names:
            self.years = self.data['year'].values 

    def set_specification(self, lags=True, regimes=True, controls=True, region=True, decade=True):
        self.specification = []
        if lags:
            self.specification += [col for col in self.feature_names if col.startswith(self.depvar) and 'lag' in col]
        if regimes:
            self.specification += ['duration', 'military', 'monarch', 'personal', 'party', 'institutions']
        if controls:
            self.specification += ['gdppc', 'growth', 'resource', 'population', 'openness', 'inequality', 'kaopen']
        if region:
            self.specification += [col for col in self.feature_names if 'region_' in col] 
        if decade: 
            self.specification += [col for col in self.feature_names if 'decade_' in col]
    
    def format_features(self, scale=False): 
        if not hasattr(self, 'specification'): 
            raise Exception('Need to set specification before formatting features')     
        if scale: 
            self.X = StandardScaler().fit_transform(self.data[self.specification].values)
        else: 
            self.X = self.data[self.specification].values


class KFoldsValidationPrediction(object): 
    def __init__(self, model, params, k, X_train, y_train):
        self.model = model
        self.params = params
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
    
    def evaluate_model(self, metric):
        search = GridSearchCV(self.model, self.params, scoring=metric, n_jobs=-1, cv=self.k)
        search.fit(self.X_train, self.y_train)
        self.optimal_params = search.best_params_
        self.cv_score = search.grid_scores_[1][1]
    
    def predict_test(self, X_test): 
        self.model.set_params(**self.optimal_params)
        self.model.fit(self.X_train, self.y_train)
        self.probabilities = self.model.predict_proba(X_test)[:, 1]

    def predict_in_sample(self):
        self.model.set_params(**self.optimal_params)
        folds = StratifiedKFold(self.y_train, n_folds=self.k)
        test_indices, test_data, test_probs = [], [], []
        for train_index, test_index in folds:
            X_train, X_test, y_train, y_test = self.X_train[train_index], self.X_train[test_index], self.y_train[train_index], self.y_train[test_index]
            train = self.model.fit(X_train, y_train)
            test_probs.append(train.predict_proba(X_test)[:, 1])
            test_indices.append(test_index)
            test_data.append(y_test)
        test_indices = np.argsort(np.concatenate(test_indices))
        self.y_test = np.concatenate(test_data)[test_indices]
        self.probabilities = np.concatenate(test_probs)[test_indices]
    
    def bootstrap_estimates(self, n_boot=100): 
        self.model.set_params(**self.optimal_params)
        ests = [np.hstack([self.model.fit(iX, iy).coef_.ravel(), self.model.fit(iX, iy).intercept_])
                for iX, iy in (resample(self.X_train, self.y_train) for _ in xrange(n_boot))] 
        self.boot_estimates = np.vstack(ests)


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
        if not isinstance(self.model, LogisticRegression): 
            raise Exception('Bootstrap model estimates only available for LogisticRegression')
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


class Melder(object):
    def __init__(self, imputations, model, params, year_threshold):
        self.imputations = imputations
        self.y = self.imputations[0].y
        self.years = self.imputations[0].years
        self.year_threshold = year_threshold
        self.train_mask = self.years <= self.year_threshold
        self.test_mask = ((self.years > self.year_threshold) & (self.years <= self.year_threshold + 10))
        self.y_test = self.y[self.test_mask]
        self.model = model 
        self.params = params
        
    def evaluate_models(self, method='K-fold'): 
        out_models = []
        for df in self.imputations:
            if method == 'K-fold':
                X_train, y_train = df.X[self.train_mask], df.y[self.train_mask]
                m = KFoldsValidationPrediction(self.model, self.params, 10, X_train, y_train)
                m.evaluate_model('log_loss')
            elif method == 'Sequential':
                m = SequentialFoldsClassifier(self.model, self.params, df.years, df.X, df.y)
                m.evaluate_model()
            else:
                raise ValueError('Method {} not supported. Must be "K-fold" or "Sequential"'.format(method))
            out_models.append(m)
        self.model_evaluations = out_models

    def meld_predictions(self, in_sample=True): 
        out_preds = [] 
        out_scores = []
        for result, df in zip(self.model_evaluations, self.imputations):
            X_test = df.X[self.test_mask]
            if in_sample:
                result.predict_in_sample()
            else:
                result.predict_test(X_test)
            out_preds.append(result.probabilities)
            out_scores.append(result.cv_score)
        self.predictions = np.array(out_preds).mean(axis=0)
        self.cv_scores = np.array(out_scores).mean(axis=0)

    def meld_estimates(self): 
        out_ests = []
        for result in self.model_evaluations:
            result.bootstrap_estimates() 
            out_ests.append(result.boot_estimates)
        self.estimates = np.concatenate(out_ests)

    def boxplot_estimates(self, names, ignore=None, fname=None): 
        if not hasattr(self, 'estimates'): 
            raise Exception('No estimates to present. Run meld_estimates first')
        dp.boxplot_estimates(self.estimates, names, ignore, fname)
