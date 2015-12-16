from __future__ import print_function 
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
        progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), 
                                                    progressbar.Percentage(), ' ']) 
        for parameters in progress(self.param_grid):
            self.model.set_params(**parameters)
            param_pr_scores = []
            param_roc_scores = []
            for yr in np.unique(self.years)[np.unique(self.years) > 1960]: 
                x_train, y_train, x_test, y_test = self.make_split(yr)
                self.model.fit(x_train, y_train)
                preds = self.model.predict_proba(x_test)[:, 1]
                try:
                    param_roc_scores.append(roc_auc_score(y_test, preds))
                    param_pr_scores.append(auc_pr_curve(y_test, preds))
                except: 
                    pass 
            self.pr_scores.append(np.nanmean(param_pr_scores))
            self.roc_scores.append(np.nanmean(param_roc_scores))
        self.optimal_params_pr = self.param_grid[np.argmax(self.pr_scores)]
        self.optimal_params_roc = self.param_grid[np.argmax(self.roc_scores)]

    def bootstrap_estimates(self, n_boot=100, metric='roc'):
        if not isinstance(self.model, LogisticRegression): 
            raise Exception('Bootstrap model estimates only available for LogisticRegression')
        if metric == 'pr':
            self.model.set_params(**self.optimal_params_pr)
        elif metric == 'roc':
            self.model.set_params(**self.optimal_params_roc)
        else:
            raise Exception('Metric {0} not supported'.format(metric))
        ests = [np.hstack([self.model.fit(iX, iy).coef_.ravel(), self.model.fit(iX, iy).intercept_])
                for iX, iy in (resample(self.X, self.y) for _ in xrange(n_boot))] 
        self.boot_estimates = np.vstack(ests)

    def predict(self, metric='roc'):
        if metric == 'pr':
            self.model.set_params(**self.optimal_params_pr)
        elif metric == 'roc':
            self.model.set_params(**self.optimal_params_roc)
        else:
            raise Exception('Metric {0} not supported'.format(metric))
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
        plt.plot(x, np.array(self.pr_scores), c='k', linestyle='--', label='Precision-Recall Curve')
        plt.plot(x, np.array(self.roc_scores), c='k', linestyle=':', label='ROC Curve')
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
    def __init__(self, results):
        self.results = results 

    def meld_predictions(self, metric='roc'): 
        out_preds = [] 
        print('\nMelding predicted probabilities')
        progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), 
                                                    progressbar.Percentage(), ' ']) 
        for result in progress(self.results):
            result.predict(metric)
            out_preds.append(result.probabilities)
        return np.array(out_preds).mean(axis=0)
        
    def meld_estimates(self): 
        out_ests = []
        print('\nMelding coefficient estimates')
        progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), 
                                                    progressbar.Percentage(), ' ']) 
        for result in progress(self.results):
            result.bootstrap_estimates() 
            out_ests.append(result.boot_estimates)
        return np.concatenate(out_ests)


def boxplot_estimates(ests, names, ignore=None, fname=None):
    if ignore is not None: 
        for factor in ignore:
            p = re.compile(factor)
            [names.remove(m) for m in filter(p.match, names)]
        ests = ests[:, :len(names)]
    data = pd.DataFrame(ests, columns=names)
    sns.boxplot(data)
    plt.axhline(y=0, linestyle='--')
    plt.ylabel('Estimate')
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
        plt.close() 
    else: 
        plt.show() 


def auc_pr_curve(y_true, y_pred): 
    # area under the precision-recall curve 
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision) 
