from __future__ import print_function 
import dissertation_plots as dp
import itertools
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import progressbar
import re 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression 
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
        if not isinstance(lag, list): 
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

    def plot_metrics(self, fname, x_label=None):
        if len(self.params) > 1: 
            raise Exception('Can only plot performance over a single hyperparameter')
        x = np.array(self.params.values()[0])
        plt.figure()  
        plt.plot(x, np.array(self.scores), c='k', linestyle=':')
        plt.xscale('log')
        if x_label is not None:
            plt.xlabel(x_label, labelpad=11)
        else: 
            plt.xlabel(self.params.keys()[0], labelpad=11)
        plt.ylabel('Area Under Curve', labelpad=11)
        plt.ylim([0.0, 1.0])
        plt.legend(loc='best')
        plt.tight_layout() 
        plt.savefig(fname)
        plt.close()
        return


class Melder(object):
    def __init__(self, imputations, model, params):
        self.imputations = imputations
        self.model = model 
        self.params = params
        self.progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), 
                                                         progressbar.Percentage(), ' '],
                                                         maxval=len(self.imputations)) 
        
    def evaluate_models(self): 
        print('\nEvaluating models for {0} imputed data sets'.format(str(len(self.imputations))))
        out_models = []
        self.progress.currval = 0
        for df in self.progress(self.imputations):
            m = SequentialFoldsClassifier(self.model, self.params, df.years, df.X, df.y)
            m.evaluate_model() 
            out_models.append(m)
        self.model_evaluations = out_models

    def meld_predictions(self): 
        print('\nMelding predicted probabilities')
        out_preds = [] 
        self.progress.currval = 0
        for result in self.progress(self.model_evaluations):
            result.predict()
            out_preds.append(result.probabilities)
        self.predictions = np.array(out_preds).mean(axis=0)
        
    def meld_estimates(self): 
        print('\nConcatenating bootstrap estimates')
        out_ests = []
        self.progress.currval = 0
        for result in self.progress(self.model_evaluations):
            result.bootstrap_estimates() 
            out_ests.append(result.boot_estimates)
        self.estimates = np.concatenate(out_ests)
