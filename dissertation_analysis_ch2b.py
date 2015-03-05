## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## dissertation_analysis_ch2b.py
## build on predictive analyses, simulations for best models with sklearn classifiers
## tb 26 feb 2015
## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##

import dissertation_programs as diss
import numpy as np
import pandas as pd
import random
from sklearn import linear_model, ensemble, metrics

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## Logistic regression with L2 regularization:
## --DV: Failure
## --Lag: Failure
## --Controls: region effects, 5-year effects with interaction

failure_spec = ['failure','lag_failure','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
failure_preds = diss.classifier_predicted_probabilities(spec = failure_spec, dep_var = 'failure', unit = 'lustrum', lag_var = 'lag_failure', factors = ['region'], interaction = True, method = 'logit')
failure_preds.to_csv('classification/bootstrap/logit_failure_failure.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## Logistic regression with L2 regularization:
## --DV: Coerce
## --Lag: Failure
## --Controls: region effects, 5-year effects with interaction

coerce_spec = ['auttrans','lag_failure','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
coerce_preds = diss.classifier_predicted_probabilities(spec = coerce_spec, dep_var = 'auttrans', unit = 'lustrum', lag_var = 'lag_failure', factors = ['region'], interaction = True, method = 'logit')
coerce_preds.to_csv('classification/bootstrap/logit_coerce_failure.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## Logistic regression with L2 regularization:
## --DV: Autocratic
## --Lag: Failure
## --Controls: region effects, 5-year effects with interaction

autocratic_spec = ['auttrans','lag_failure','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
autocratic_preds = diss.classifier_predicted_probabilities(spec = autocratic_spec, dep_var = 'auttrans', unit = 'lustrum', lag_var = 'lag_failure', factors = ['region'], interaction = True, method = 'logit')
autocratic_preds.to_csv('classification/bootstrap/logit_autocratic_failure.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## Logistic regression with L1 regularization:
## --DV: Failure
## --Lag: Failure
## --Controls: region effects, 5-year effects with interaction

failure_spec = ['failure','lag_failure','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
failure_preds = diss.classifier_predicted_probabilities(spec = failure_spec, dep_var = 'failure', unit = 'lustrum', lag_var = 'lag_failure', factors = ['region'], interaction = True, method = 'lasso')
failure_preds.to_csv('classification/bootstrap/lasso_failure_failure.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## Logistic regression with L1 regularization:
## --DV: Coerce
## --Lag: Failure
## --Controls: region effects, 5-year effects with interaction

coerce_spec = ['coerce','lag_failure','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
coerce_preds = diss.classifier_predicted_probabilities(spec = coerce_spec, dep_var = 'coerce', unit = 'lustrum', lag_var = 'lag_failure', factors = ['region'], interaction = True, method = 'lasso')
coerce_preds.to_csv('classification/bootstrap/lasso_coerce_failure.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## Logistic regression with L1 regularization:
## --DV: Autocratic
## --Lag: Failure
## --Controls: region effects, 5-year effects with interaction

autocratic_spec = ['auttrans','lag_failure','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
autocratic_preds = diss.classifier_predicted_probabilities(spec = autocratic_spec, dep_var = 'auttrans', unit = 'lustrum', lag_var = 'lag_failure', factors = ['region'], interaction = True, method = 'lasso')
autocratic_preds.to_csv('classification/bootstrap/lasso_autocratic_failure.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## random forest
## --DV: Failure
## --Lag: Failure
## --Controls: region effects, 5-year effects

failure_spec = ['failure','lag_failure','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
failure_preds = diss.classifier_predicted_probabilities(spec = failure_spec, dep_var = 'failure', unit = 'lustrum', lag_var = 'lag_failure', factors = ['region'], interaction = False, method = 'random_forest')
failure_preds.to_csv('classification/bootstrap/random_forest_failure_failure.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## random forest
## --DV: Failure
## --Lag: Coerce
## --Controls: region effects, 5-year effects

failure_spec = ['failure','lag_coerce','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
failure_preds = diss.classifier_predicted_probabilities(spec = failure_spec, dep_var = 'failure', unit = 'lustrum', lag_var = 'lag_coerce', factors = ['region'], interaction = False, method = 'random_forest')
failure_preds.to_csv('classification/bootstrap/random_forest_failure_coerce.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## random forest
## --DV: Failure
## --Lag: Autocratic
## --Controls: region effects, 5-year effects

failure_spec = ['failure','lag_auttrans','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
failure_preds = diss.classifier_predicted_probabilities(spec = failure_spec, dep_var = 'failure', unit = 'lustrum', lag_var = 'lag_auttrans', factors = ['region'], interaction = False, method = 'random_forest')
failure_preds.to_csv('classification/bootstrap/random_forest_failure_autocratic.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## random forest
## --DV: Coerce
## --Lag: Coerce
## --Controls: region effects, 5-year effects

coerce_spec = ['coerce','lag_coerce','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
coerce_preds = diss.classifier_predicted_probabilities(spec = coerce_spec, dep_var = 'coerce', unit = 'lustrum', lag_var = 'lag_coerce', factors = ['region'], interaction = False, method = 'random_forest')
coerce_preds.to_csv('classification/bootstrap/random_forest_coerce_coerce.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## random forest
## --DV: Autocratic
## --Lag: Coerce
## --Controls: region effects, 5-year effects

autocratic_spec = ['auttrans','lag_coerce','duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region','lustrum']
autocratic_preds = diss.classifier_predicted_probabilities(spec = autocratic_spec, dep_var = 'auttrans', unit = 'lustrum', lag_var = 'lag_coerce', factors = ['region'], interaction = False, method = 'random_forest')
autocratic_preds.to_csv('classification/bootstrap/random_forest_autocratic_coerce.csv')

## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------- ##

