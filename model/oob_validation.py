from __future__ import print_function
import numpy as np
import progressbar
from itertools import product
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.utils import check_random_state
from sklearn.utils.fixes import bincount
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])


def _generate_sample_indices(random_state, n_samples):
    '''Samples in bag'''
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices

def _generate_unsampled_indices(random_state, n_samples):
    '''Samples out of bag'''
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]
    return unsampled_indices


class OOBPerformance(object): 
    '''OOB error for provided ensemble classifier'''
    def __init__(self, classifier, X, y):
        self.classifier = classifier
        self.X = X
        self.y = y
        self.oob_index_array = self._generate_oob_index_array()

    def _generate_oob_index_array(self):
        '''OOB membership for each estimator, return array([n_samples, n_estimators])'''
        n_samples = self.X.shape[0]
        samples = np.arange(n_samples)
        return np.array([np.in1d(samples, _generate_unsampled_indices(estimator.random_state, n_samples)) for estimator in self.classifier.estimators_]).T

    def _oob_predict_proba(self, sample_idx):
        '''OOB predicted probability for a single sample'''
        oob_estimators = np.array(self.classifier.estimators_)[self.oob_index_array[sample_idx]]
        sample = self.X[sample_idx].reshape(1, -1)
        return np.array([estimator.predict_proba(sample)[:, 1] for estimator in oob_estimators]).mean() 

    def oob_score(self, metric):
        '''Evaluate OOB predicted probability for all samples and return loss for given metric'''
        probabilities = np.array([self._oob_predict_proba(i) for i in xrange(self.X.shape[0])])
        return metric(self.y, probabilities) 

    def _oob_permute_predict(self, estimator_idx, metric):
        '''Evaluate change in metric randomly permuting each feature for the estimator'''
        estimator = self.classifier.estimators_[estimator_idx]
        oob_samples_mask = self.oob_index_array[:, estimator_idx]
        oob_samples = self.X[oob_samples_mask]
        oob_y_true = self.y[oob_samples_mask]
        oob_probs_baseline = estimator.predict_proba(oob_samples)[:, 1]
        oob_error_baseline = metric(oob_y_true, oob_probs_baseline)
        oob_errors_permutation = []
        for col in xrange(self.X.shape[1]): 
            oob_samples_c = oob_samples.copy() 
            np.random.shuffle(oob_samples_c[:, col])
            oob_probs_permutation = estimator.predict_proba(oob_samples_c)[:, 1]
            oob_errors_permutation.append(metric(oob_y_true, oob_probs_permutation))
        return np.array(oob_errors_permutation) - oob_error_baseline

    def permutation_importance(self, metric): 
        '''Average change in metric due to random permutation of features for each estimator'''
        return np.vstack([self._oob_permute_predict(i, metric) for i in xrange(len(self.classifier.estimators_))]).mean(axis=0)


class OOBValidation(object):
    '''Out-of-bag validation using the default scoring method'''
    def __init__(self, model, param_dict):
        '''
        Input:
            model : sklearn model instance
            param_dict : dictionary, hyperparameters (keys), arrays (values)
        '''
        self.model = model
        self.param_dict = param_dict
        self.param_grid = self.make_param_grid()

    def make_param_grid(self):
        '''Converts dictionary of hyperparameters and values to grid for searching'''
        param_combos = [x for x in apply(product, self.param_dict.values())]
        return [dict(zip(self.param_dict.keys(), p)) for p in param_combos]

    def fit(self, X, y, metric, minimize=True):
        '''Grid search over hyperparameters, evaluation with given metric in OOB samples'''
        self.oob_scores = []
        progress.currval = 0
        for params in progress(self.param_grid):
            self.model.set_params(**params)
            self.model.fit(X, y)
            oob = OOBPerformance(self.model, X, y) 
            self.oob_scores.append(oob.oob_score(metric))
        best_idx = np.argmin(np.array(self.oob_scores)) if minimize else np.argmax(np.array(self.oob_scores))
        self.best_params = self.param_grid[best_idx]
        self.best_model = self.model.set_params(**self.best_params)
        self.oob_score_ = self.oob_scores[best_idx]

    def predict(self, X):
        '''Predicted classes and probabilities for features array X using best model'''
        self.predicted_probabilities = self.best_model.predict_proba(X)[:, 1]
        self.predicted_values = self.best_model.predict(X)


class OOBMelder(object): 
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

    def validate_models(self): 
        out_models = []
        out_scores = []
        for df in self.imputations:
            X_train, y_train = df.X[self.train_mask], df.y[self.train_mask]
            validator = OOBValidation(self.model, self.params)
            validator.fit(X_train, y_train, metric=log_loss)
            out_models.append(validator.best_model)
            out_scores.append(validator.oob_score_)
        self.best_models = out_models
        self.scores = np.array(out_scores)

    def meld_predictions(self, train=True): 
        out_preds = [] 
        for result, df in zip(self.best_models, self.imputations):
            if train:
                out_preds.append(result.predict_proba(df.X[self.train_mask])[:, 1])
            else:
                out_preds.append(result.predict_proba(df.X[self.test_mask])[:, 1])
        self.predictions = np.array(out_preds).mean(axis=0)

    def meld_feature_importances(self, metric=log_loss): 
        oobs = [OOBPerformance(model, df.X[self.train_mask], df.y[self.train_mask]) for model, df in zip(self.best_models, self.imputations)]
        self.feature_importances_array = np.array([oob.permutation_importance(metric) for oob in oobs]).T
        self.feature_importances = self.feature_importances_array.mean(axis=1)


if __name__ == '__main__': 
    X, y = make_classification(n_samples=500)

    # demo OOBLoss for an ensemble classifier
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    forest.fit(X, y)
    oob = OOBPerformance(forest, X, y)
    loss = oob.oob_score(log_loss)

    # demo grid search over hyperparameters using OOB validation 
    params = {'max_features' : ['sqrt', 0.5],
              'min_samples_split' : [6, 12],
              'min_samples_leaf' : [3, 7]}
    validator = OOBValidation(forest, params)
    validator.fit(X, y, metric=log_loss)
    validator.predict(X)

    # demo permutation importance
    model = validator.best_model
    oob_best = OOBPerformance(model, X, y)
    importance = oob_best.permutation_importance(log_loss)

    # compare permutation importance to scikit-learn feature importance
    model.fit(X, y)
    importance2 = model.feature_importances_
    print(np.corrcoef(importance, importance2))
