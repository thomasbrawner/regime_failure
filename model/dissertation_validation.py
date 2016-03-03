from __future__ import print_function
import dissertation as diss
import numpy as np 
import pandas as pd 
import progressbar
import warnings
from pymongo import MongoClient
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
warnings.simplefilter('ignore')
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' ']) 


def prepare_data(file_path, dep_var, spec, scale=True): 
    df = pd.read_table(file_path, sep=',', index_col=0)
    df = diss.DataFormatter(df, depvar=dep_var)
    df.set_specification(
            lags='lags' in spec,
            regimes='regimes' in spec,
            controls='controls' in spec,
            region='region' in spec,
            decade='decade' in spec)
    df.format_features(scale=scale)
    return df

def make_file_path(number): 
    return 'clean_data/imputation_{0}.csv'.format(str(number))

def make_mongo_docs(model, attributes, probabilities_in, probabilities_test, cv_scores, year_threshold):
    model = model.__class__.__name__
    docs = list() 
    for i in xrange(len(attributes)):
        docs.append(
                {'model' : model, 
                 'attributes' : attributes[i],
                 'probabilities_in' : probabilities_in[i],
                 'probabilities_test' : probabilities_test[i],
                 'year_threshold' : year_threshold,
                 'cv_score' : cv_scores[i]})
    return docs


classifiers = [
    SGDClassifier(loss='log', penalty='elasticnet'),
    ExtraTreesClassifier(n_estimators=3000, n_jobs=-1),
    RandomForestClassifier(n_estimators=3000, n_jobs=-1),
    GradientBoostingClassifier(n_estimators=3000)
    ]

hyperparameters = [
    {'alpha' : np.logspace(-3, 0, num=21), 'l1_ratio' : np.linspace(0, 1, num=11)},
    {'max_depth' : [15, None], 'min_samples_split' : [6, 12, 18], 'min_samples_leaf' : [2, 4]}, 
    {'max_depth' : [15, None], 'min_samples_split' : [6, 12, 18], 'min_samples_leaf' : [2, 4]}, 
    {'loss' : ['deviance', 'exponential'], 'max_depth' : [3, 5, 9], 'max_features' : ['sqrt', None], 'subsample' : [0.632, 1.0], 'learning_rate' : [0.001, 0.005, 0.01, 0.1]}
    ]

year_thresholds = [1975, 1985, 1995, 2005]
attributes = [['lags', 'regimes', 'controls', 'region', 'decade'],
              ['regimes', 'controls', 'region', 'decade']]

mongo_docs = []
for model, params in zip(classifiers, hyperparameters):
    for threshold in year_thresholds:
        print('Model: {0}, Year: {1}'.format(model.__class__.__name__, str(threshold)))
        progress.currval = 0
        out_probs_in = []
        out_probs_test = []
        out_cv_scores = [] 
        for specification in progress(attributes):
            data_list = [prepare_data(file_path=make_file_path(i), dep_var='failure', spec=specification) for i in xrange(1, 11)]
            melder = diss.Melder(data_list, model, params, year_threshold=threshold)
            melder.evaluate_models()
            melder.meld_predictions(in_sample=True)
            out_probs_in.append(melder.predictions)
            melder.meld_predictions(in_sample=False)
            out_probs_test.append(melder.predictions)
            out_cv_scores.append(melder.cv_scores)
        mongo_docs += make_mongo_docs(model, attributes, out_probs_in, out_probs_test, out_cv_scores, threshold)

client = MongoClient()
db = client['dissertation']
collection = db['validation']
for document in mongo_docs:
    if not collection.find_one({'model' : document['model'], 'attributes' : document['attributes'], 'year_threshold' : document['year_threshold']}):
        document['probabilities_in'] = document['probabilities_in'].tolist()  # mongo does not like numpy arrays
        document['probabilities_test'] = document['probabilities_test'].tolist()
        collection.insert_one(document)
client.close() 
