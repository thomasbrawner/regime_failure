import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from oob_validation import OOBValidation
sns.set_style('ticks')


def make_pred_prob_plot_data(model, df, column):
    dfc = df.copy() 
    rng = np.linspace(df[column].min(), df[column].max())
    probs = []
    for val in rng:
        dfc[column] = val
        pred_probs = model.predict_proba(dfc)[:, 1]
        probs.append([boot_sample.mean() for boot_sample in (resample(pred_probs) for _ in xrange(1000))])
    return rng, np.array(probs).T

def probability_plot(model, df, column, fname):
    rng, probs = make_pred_prob_plot_data(model, df, column)
    fig, ax1 = plt.subplots()
    prob_means = probs.mean(axis=0)
    upper_bounds = np.percentile(probs, q=95, axis=0)
    lower_bounds = np.percentile(probs, q=5, axis=0)
    ax1.plot(rng, prob_means, color='k')
    ax1.fill_between(rng, lower_bounds, upper_bounds, facecolor='k', alpha=0.25)
    ax1.set_xlabel(column, labelpad=11)
    ax1.set_ylabel('Predicted Probability', labelpad=11)
    ax2 = ax1.twinx()
    ax2.hist(df[column].values, color='k', alpha=0.15)
    ax2.set_ylabel('Frequency', labelpad=11)
    plt.tight_layout() 
    plt.savefig(fname)
    plt.close() 


if __name__ == '__main__':
    X, y = make_classification(n_samples=500)
    df = pd.DataFrame(X)
    df.columns = ['c' + str(i) for i in xrange(X.shape[1])]
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True)
    params = {
            'max_features' : ['sqrt', 0.5],
            'min_samples_split' : [6, 12],
            'min_samples_leaf' : [3, 7]}
    validator = OOBValidation(forest, params)
    validator.fit(X, y)
    probability_plot(validator.best_model, df, 'c0', 'predicted_probabilities.png')
