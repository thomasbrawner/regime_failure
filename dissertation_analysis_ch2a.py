## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## dissertation_analysis_ch2a.py
## Cross-validation performance for classifiers
## tb 26 feb 2015, last update 2 mar 2015
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##

import dissertation_programs as diss
import numpy as np
import pandas as pd
import random
import sklearn

## --------------------------------------------------------------------- ##
## how many iterations for out-of-sample predictions? 

n_repeats = 100

## --------------------------------------------------------------------- ##
## dependent variables, spatial lags

dep_vars = ['failure','coerce','auttrans','demtrans']
lag_vars = ['lag_failure','lag_coerce','lag_auttrans','lag_demtrans']

## --------------------------------------------------------------------- ##
## control variables

controls0 = ['duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions']
controls1 = ['duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region']
controls2 = ['duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','cowcode']

controls = [controls0, controls1, controls2]

## --------------------------------------------------------------------- ##
## period dummies

periods = [None, 'two_year', 'lustrum']

## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
##
## classifiers
## 
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## logit l1

out_metrics = []
for period in periods:
    for dep_var in dep_vars:
        data = diss.format_data(dep_var = dep_var, lag = 'spatial_lags_' + dep_var + '.txt')
        if period == None:
            rest = [[dep_var] + c for c in controls]
            full = {'lagfail' : [[dep_var] + ['lag_failure'] + c for c in controls],
                    'lagcoer' : [[dep_var] + ['lag_coerce'] + c for c in controls],
                    'lagauto' : [[dep_var] + ['lag_auttrans'] + c for c in controls],
                    'lagdemo' : [[dep_var] + ['lag_demtrans'] + c for c in controls]}
        else:
            rest = [[dep_var] + [period] + c for c in controls]
            full = {'lagfail' : [[dep_var] + [period] + ['lag_failure'] + c for c in controls],
                    'lagcoer' : [[dep_var] + [period] + ['lag_coerce'] + c for c in controls],
                    'lagauto' : [[dep_var] + [period] + ['lag_auttrans'] + c for c in controls],
                    'lagdemo' : [[dep_var] + [period] + ['lag_demtrans'] + c for c in controls]}
        for i in range(n_repeats):
            out_metrics += diss.predictor_performance(data, dep_var, period, rest, full, metric = ['auroc','recall'], method = 'lasso')

pd.DataFrame(out_metrics).to_csv('classification/performance_lasso.csv')

## --------------------------------------------------------------------- ##
## logit l2

out_metrics = []
for period in periods:
    for dep_var in dep_vars:
        data = diss.format_data(dep_var = dep_var, lag = 'spatial_lags_' + dep_var + '.txt')
        if period == None:
            rest = [[dep_var] + c for c in controls]
            full = {'lagfail' : [[dep_var] + ['lag_failure'] + c for c in controls],
                    'lagcoer' : [[dep_var] + ['lag_coerce'] + c for c in controls],
                    'lagauto' : [[dep_var] + ['lag_auttrans'] + c for c in controls],
                    'lagdemo' : [[dep_var] + ['lag_demtrans'] + c for c in controls]}
        else:
            rest = [[dep_var] + [period] + c for c in controls]
            full = {'lagfail' : [[dep_var] + [period] + ['lag_failure'] + c for c in controls],
                    'lagcoer' : [[dep_var] + [period] + ['lag_coerce'] + c for c in controls],
                    'lagauto' : [[dep_var] + [period] + ['lag_auttrans'] + c for c in controls],
                    'lagdemo' : [[dep_var] + [period] + ['lag_demtrans'] + c for c in controls]}
        for i in range(n_repeats):
            out_metrics += diss.predictor_performance(data, dep_var, period, rest, full, metric = ['auroc','recall'])

pd.DataFrame(out_metrics).to_csv('classification/performance_logit.csv')

## --------------------------------------------------------------------- ##
## svm

out_metrics = []
for period in periods:
    for dep_var in dep_vars:
        data = diss.format_data(dep_var = dep_var, lag = 'spatial_lags_' + dep_var + '.txt')
        if period == None:
            rest = [[dep_var] + c for c in controls]
            full = {'lagfail' : [[dep_var] + ['lag_failure'] + c for c in controls],
                    'lagcoer' : [[dep_var] + ['lag_coerce'] + c for c in controls],
                    'lagauto' : [[dep_var] + ['lag_auttrans'] + c for c in controls],
                    'lagdemo' : [[dep_var] + ['lag_demtrans'] + c for c in controls]}
        else:
            rest = [[dep_var] + [period] + c for c in controls]
            full = {'lagfail' : [[dep_var] + [period] + ['lag_failure'] + c for c in controls],
                    'lagcoer' : [[dep_var] + [period] + ['lag_coerce'] + c for c in controls],
                    'lagauto' : [[dep_var] + [period] + ['lag_auttrans'] + c for c in controls],
                    'lagdemo' : [[dep_var] + [period] + ['lag_demtrans'] + c for c in controls]}
        for i in range(n_repeats):
            out_metrics += diss.predictor_performance(data, dep_var, period, rest, full, metric = ['auroc','recall'], method = 'svm')

pd.DataFrame(out_metrics).to_csv('classification/performance_svm2.csv')

## --------------------------------------------------------------------- ##
## random forest

out_metrics = []
for period in periods:
    for dep_var in dep_vars:
        data = diss.format_data(dep_var = dep_var, lag = 'spatial_lags_' + dep_var + '.txt')
        if period == None:
            rest = [[dep_var] + c for c in controls]
            full = {'lagfail' : [[dep_var] + ['lag_failure'] + c for c in controls],
                    'lagcoer' : [[dep_var] + ['lag_coerce'] + c for c in controls],
                    'lagauto' : [[dep_var] + ['lag_auttrans'] + c for c in controls],
                    'lagdemo' : [[dep_var] + ['lag_demtrans'] + c for c in controls]}
        else:
            rest = [[dep_var] + [period] + c for c in controls]
            full = {'lagfail' : [[dep_var] + [period] + ['lag_failure'] + c for c in controls],
                    'lagcoer' : [[dep_var] + [period] + ['lag_coerce'] + c for c in controls],
                    'lagauto' : [[dep_var] + [period] + ['lag_auttrans'] + c for c in controls],
                    'lagdemo' : [[dep_var] + [period] + ['lag_demtrans'] + c for c in controls]}
        for i in range(n_repeats):
            out_metrics += diss.predictor_performance(data, dep_var, period, rest, full, metric = ['auroc','recall'], method = 'random_forest', interaction = False)

pd.DataFrame(out_metrics).to_csv('classification/performance_random_forest.csv')

## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
