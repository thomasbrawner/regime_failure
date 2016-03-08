import numpy as np
import progressbar
from itertools import product
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state
from sklearn.utils.fixes import bincount
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])


def _generate_sample_indices(random_state, n_samples):
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices

def _generate_unsampled_indices(random_state, n_samples):
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]
    return unsampled_indices

def sample_is_oob(estimators, X):
    n_samples = X.shape[0]
    samples = np.arange(n_samples)
    return np.array([np.in1d(samples, _generate_unsampled_indices(tree.random_state, n_samples)) for tree in estimators]).T



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

    def fit(self, X, y):
        self.oob_scores = []
        progress.currval = 0
        for params in progress(self.param_grid):
            self.model.set_params(**params)
            self.model.fit(X, y)
            self.oob_scores.append(self.model.oob_score_)
        best_idx = np.argmax(np.array(self.oob_scores))
        self.best_params = self.param_grid[best_idx]
        self.best_model = self.model.set_params(**self.best_params)

    def predict(self, X):
        self.predicted_probabilities = self.best_model.predict_proba(X)[:, 1]
        self.predicted_values = self.best_model.predict(X)


if __name__ == '__main__': 
    X, y = make_classification(n_samples=500)
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    forest.fit(X, y)
    oob_idx = sample_is_oob(forest.estimators_, X)
