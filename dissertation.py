from __future__ import print_function 
import dissertation_plots as dp
import itertools
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import progressbar
import re 
import seaborn as sns 
from sklearn.grid_search import GridSearchCV
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

    def set_specification(self, lags=True, regimes=True, controls=True, cs=True, ts=True):
        self.specification = []
        if lags:
            self.specification += [col for col in self.feature_names if 'lag' in col]
        if regimes:
            self.specification += ['duration', 'military', 'personal', 'party', 'institutions']
        if controls:
            self.specification += ['gdppc', 'growth', 'resource', 'population'] 
        if cs:
            self.specification += [col for col in self.feature_names if 'region' in col] 
        if ts: 
            self.specification += [col for col in self.feature_names if 'decade' in col]

    def format_features(self, scale=False): 
        if not hasattr(self, 'specification'): 
            raise Exception('Need to set specification before formatting features')     
        if scale: 
            self.X = StandardScaler().fit_transform(self.data[self.specification].values)
        else: 
            self.X = self.data[self.specification].values


class KFoldsClassifier(object): 
    def __init__(self, model, params, k, X, y):
        self.model = model
        self.params = params
        self.k = k
        self.X = X
        self.y = y
    
    def make_param_grid(self): 
        param_combos = [x for x in apply(itertools.product, self.params.values())]
        return [dict(zip(self.params.keys(), p)) for p in param_combos]

    def evaluate_model(self, metric):
        self.param_grid = self.make_param_grid() 
        search = GridSearchCV(self.model, self.param_grid, scoring=metric, n_jobs=-1, cv=10)
        search.fit(self.X, self.y)
        self.optimal_params = search.best_params_
    
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
    def __init__(self, imputations, model, params):
        self.imputations = imputations
        self.y = self.imputations[0].y
        self.years = self.imputations[0].years
        self.y_test = self.y[self.years > 1960]
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

    def boxplot_estimates(self, names, ignore=None, fname=None): 
        if not hasattr(self, 'estimates'): 
            raise Exception('No estimates to present. Run meld_estimates first')
        dp.boxplot_estimates(self.estimates, names, ignore, fname)



def prepare_data(file_path, dep_var, lag_var, factors, scale=False): 
    df = pd.read_table(file_path, sep=',', index_col=0)
    df = DataFormatter(df, depvar=dep_var)
    df.set_specification(lag=lag_var, factors=factors)
    df.format_features(scale=scale)
    return df

def make_file_path(number): 
    return 'clean_data/imputation_{0}.csv'.format(str(number))

def specification_details(classifier, dep_var, lag_var): 
    model = classifier.__class__.__name__
    penalty = '' 
    if isinstance(classifier, LogisticRegression): 
        penalty = classifier.get_params()['penalty']
    s = '\n' + model + ' ' + penalty + '\nDependent Variable: ' + dep_var + '\nLag Variable: ' + '_'.join(lag_var)
    m = model + penalty
    return s, m

def run_specification(classifier, params, dep_var, lag_var, factors=None, scale=False, estimates=False):
    data_list = [prepare_data(file_path=make_file_path(i), 
                              dep_var=dep_var, 
                              lag_var=lag_var,
                              factors=factors, 
                              scale=scale)
                 for i in xrange(1, 11)]
    spec, mod = specification_details(classifier, dep_var, lag_var)
    print(spec)
    melder = Melder(data_list, classifier, params)
    melder.evaluate_models()
    melder.meld_predictions()
    if estimates:
        melder.meld_estimates() 
    return melder, mod
