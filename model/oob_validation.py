import numpy as np
import progressbar
from itertools import product
from sklearn.ensemble import RandomForestClassifier
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])


class OOBValidation(object):

    def __init__(self, model, param_dict):
        self.model = model
        self.param_dict = param_dict
        self.param_grid = self.make_param_grid()

    def make_param_grid(self):
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
