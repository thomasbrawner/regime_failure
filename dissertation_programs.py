## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## dissertation_programs.py
## tb 21 jan 2015, last update 19 feb 2015
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##

import glob
import itertools
import numpy as np
import pandas as pd
from math import floor
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import cross_validation, linear_model
import statsmodels.api as sm
import sys

## --------------------------------------------------------------------- ##
## analysis class

class brada(object):
    '''
    Create an instance of a data analysis object that can be used by the functions
    in statsmodels & sklearn modules.
    Arguments: 
        - data_frame: pandas data frame with columns to include in the analysis
        - dep_var: the column name for the dependent variable, given as string
        - factors: list type of factor variables in the analysis, to be transformed to dummy variables
        - constant: boolean, specify if column of 1s should be generated for estimating an intercept (default = True) 
    Values: 
        - X: pandas data frame of features 
        - y: vector of observations of the dependent variable
        - names: feature names
        - dv: dependent variable name
        - dim: shape of the data frame
    '''
    def __init__(self, data_frame, dep_var, factors = None, constant = True):
        data_frame = data_frame.dropna(axis = 0)
        data_frame.reset_index(drop = True, inplace = True)
        if factors != None:
            for f in factors:
                dummy = pd.get_dummies(data_frame[f], prefix = f)
                data_frame = pd.merge(data_frame, dummy, left_index = True, right_index = True, how = 'left')
                data_frame.drop(f, axis = 1, inplace = True)
        features = [feat for feat in data_frame.columns.tolist() if feat not in [dep_var]]
        self.X = data_frame[features]
        if constant:
            self.X = sm.add_constant(self.X)
        self.y = data_frame[dep_var]
        self.names = features
        self.dv = dep_var
        self.dim = data_frame.shape

    def estimate(self, method = 'logit'):
        '''
        Estimator method for instance of brada. Uses linear regression and logistic regression
        estimators in the statsmodels module.
        Arguments: 
            - method: one of either 'logit' (default) or 'ols' for a linear probability model. The dissertation
                      looks uniquely at binary outcomes, including regime failure and coups d'etat, thus a focus
                      on these two estimators.
        Values (dictionary with following key, value pairs): 
            - table: regression table with coefficient estimates, standard errors, p-values, t values, and 95% confidence 
                     interval, as pandas data frame
            - deviance: deviance (for method == 'logit')
            - rmse: root mean squared error (for method == 'ols')
            - rsq: R squared (for method == 'ols')
            - loglik: log-likelihood
            - aic: Akaike information criterion
            - bic: Bayesian information criterion
            - n: number of observations
        '''
        if method == 'ols':
            model = sm.OLS(self.y, self.X).fit()
            coefs = np.round(model.params, 3)
            ses = np.round(model.bse, 3)
            pval = np.round(model.pvalues, 3)
            tval = np.round(model.tvalues, 3)
            conf = np.round(model.conf_int(), 3)
            table = pd.concat([coefs, ses, pval, tval, conf], axis = 1)
            table.columns = ['estimate','std. error','p-value','t','0.025','0.975']
            out = {'table' : table,
                   'estimator' : method,
                   'rmse' : round(np.sqrt(model.mse_resid), 3),
                   'rsq' : round(model.rsquared, 3),
                   'loglik' : round(model.llf, 3),
                   'aic' : round(model.aic, 3),
                   'bic' : round(model.bic, 3),
                   'n' : int(model.nobs)}
            return out
        elif method == 'logit':
            model = sm.GLM(self.y, self.X, family = sm.families.Binomial()).fit()
            coefs = np.round(model.params, 3)
            ses = np.round(model.bse, 3)
            pval = np.round(model.pvalues, 3)
            tval = np.round(model.tvalues, 3)
            conf = np.round(model.conf_int(), 3)
            table = pd.concat([coefs, ses, pval, tval, conf], axis = 1)
            table.columns = ['estimate','std. error','p-value','z','0.025','0.975']
            out = {'table' : table,
                   'estimator' : method,
                   'deviance' : round(model.deviance, 3),
                   'loglik' : round(model.llf, 3),
                   'aic' : round(model.aic, 3),
                   'bic' : round(model.bic, 3),
                   'n' : int(model.nobs)}
            return out
        else:
            print 'method must be one of ["logit","ols"]'
            return None

    def simulate(self, method = 'logit', nsims = 1000, set_var = None, set_values = None, seed = None):
        '''
        Simulation method for instance of brada. Uses the estimates from a linear probability model or a logistic
        regression to generate simulations of predicted probabilies at counterfactual values of a predictor variable of 
        interest. At present, provide a list of two values for this variable to evaluate a distribution of simulated 
        predicted probabilities along with the first differences in the predicted probabilities at these two values.
        Arguments:
            - method: one of 'logit' (default) or 'ols' for a linear probability model
            - nsims: number of simulated coefficient vectors drawn from multivariate normal distribution with mean 
                     corresponding to the coefficient estimates and variance corresponding to the variance-covariance
                     matrix.
            - set_var: the variable of interest for which we want to set counterfactual values
            - set_values: list of length 2 with values at which the simulated predicted probabilities are evaluated
            - seed: optional, set the random number for drawing simulated coefficient vectors, for replicability
        Values:
            - pandas data frame with three columns: 'val0' = predicted probabilities for first value in set_values; 
              'val1' = predicted probabilities for second value; 'fd' = first difference (val0 - val1).
        '''
        out = []
        if seed:
            np.random.seed(seed)
        if method == 'logit':
            model = sm.GLM(self.y, self.X, family = sm.families.Binomial()).fit()
            coefs = model.params
            vcov = model.cov_params()
            try:
                x = np.random.multivariate_normal(coefs, vcov, nsims)
            except:
                print 'Simulations not executed for provided model'
                return None
            for val in set_values:
                self.X[set_var] = val
                linpreds = np.dot(self.X, x.T)
                probs = 1 / (1 + np.exp(-1 * linpreds))
                out.append(probs)
        elif method == 'ols':
            model = sm.OLS(self.y, self.X).fit()
            coefs = model.params
            vcov = model.cov_params()
            x = np.random.multivariate_normal(coefs, vcov, nsims)
            for val in set_values:
                self.X[set_var] = val
                out.append(np.dot(self.X, x.T))
        else:
            print 'method must be one of ["logit","ols"]'
            return None
        out.append(out[0] - out[1])
        out = [pd.DataFrame(x).mean(axis = 0) for x in out]
        out = pd.concat(out, axis = 1)
        out.columns = ['x0','x1','fd']
        return out

    def classify(self, method = 'logit', folds = 5, seed = None, beta = [0.5]):
        '''
        Classification method for instance of brada. Uses a number of classifiers provided in the sklearn module
        to make out-of-sample forecasts in k-fold cross-validation. 
        Arguments: 
            - method: one of ['logit','random_forest','svm','naive_bayes','boosting']
            - folds: number of folds used in cross-validation (default = 5)
            - seed: optional, seed value for splitting the data into folds, for replicability
            - beta: list, values of beta at which to evaluate the F-beta score
        Values (dictionary with the following key, value pairs):
            - predictions: pandas data frame with observed outcome values, classified values, and predicted probabilities
            - accuracy: accuracy
            - f1: F1 score (F-score)
            - fbeta: pandas data frame of betas and F-beta score(s) with n_rows equal to len(beta)
            - precision: precision
            - recall: recall
            - confusion: 2x2 confusion matrix
            - roc: data frame of false positive rate, true positive rate, and thresholds, for making ROC plots
            - auroc: area under the ROC curve
        '''
        splits = cross_validation.StratifiedKFold(self.y, n_folds = folds, shuffle = True, random_state = seed)
        out = []
        for train, test in splits:
            x_train, x_test, y_train, y_test = self.X.ix[train], self.X.ix[test], self.y.ix[train], self.y.ix[test]
            if method == 'logit':
                model = linear_model.LogisticRegression().fit(x_train, y_train)
            elif method == 'random_forest':
                model = sklearn.ensemble.RandomForestClassifier(n_estimators = 1000, random_state = seed).fit(x_train, y_train)
            elif method == 'svm':
                model = sklearn.svm.SVC(kernel = 'rbf', probability = True, random_state = seed).fit(x_train, y_train)
            elif method == 'naive_bayes':
                model = sklearn.naive_bayes.BernoulliNB().fit(x_train, y_train)
            elif method == 'boosting':
                model = sklearn.ensemble.AdaBoostClassifier(n_estimators = 100, random_state = seed).fit(x_train, y_train)
            else:
                print 'method must be one of ["logit","random_forest","svm","naive_bayes","boosting"]'
                return None
            preds = zip(test.tolist(), y_test.tolist(), model.predict(x_test).tolist(), model.predict_proba(x_test)[:,1:].flatten().tolist())
            out += preds
        out = pd.DataFrame(out)
        out.columns = ['index','observed','classified','probability']
        out.set_index('index', inplace = True)
        out.sort_index(inplace = True)
        acc = sklearn.metrics.accuracy_score(out['observed'], out['classified'], normalize = True)
        confusion = sklearn.metrics.confusion_matrix(out['observed'], out['classified'])
        f1 = sklearn.metrics.f1_score(out['observed'], out['classified'])
        fbeta_list = []
        for b in beta:
            fbeta_list += [(b, sklearn.metrics.fbeta_score(out['observed'], out['classified'], beta = b, average = 'weighted'))]
        fbeta = pd.DataFrame(fbeta_list)
        fbeta.columns = ['beta','score']
        precision = sklearn.metrics.precision_score(out['observed'], out['classified'])
        recall = sklearn.metrics.recall_score(out['observed'], out['classified'])
        roc = sklearn.metrics.roc_curve(out['observed'], out['probability'])
        roc = pd.DataFrame([roc[0], roc[1], roc[2]]).T
        roc.columns = ['fpr','tpr','threshold']
        auroc = sklearn.metrics.roc_auc_score(out['observed'], out['probability'])
        out_dict = {'predictions' : out,
                    'accuracy' : round(acc, 3),
                    'f1' : round(f1, 3),
                    'fbeta' : fbeta,
                    'precision' : round(precision, 3),
                    'recall' : round(recall, 3),
                    'confusion' : confusion,
                    'roc' : roc,
                    'auroc' : round(auroc, 3)}
        return out_dict
            
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##

## complementary functions

## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##

def round_down_any(x, base):
    '''
    Round down to nearest <base>.
    '''
    round_down = np.vectorize(floor)
    x = np.array((x / base), dtype = float)
    out = round_down(x)
    return pd.Series(out * base).astype(int)
    
## --------------------------------------------------------------------- ##

def log_negative(vec):
    '''
    Transform highly kurtotic distributions centered about zero.
    '''
    return np.where(vec < 0, np.log((vec * -1) + 1) * -1, np.log(vec + 1))

## --------------------------------------------------------------------- ##

def format_data(gwf = True, dep_var = 'failure', control = True, lag = None):
    '''
    Given the directory setup of the data files ('data/...'), merge and manipulate files for analysis.
    Notably, this function generates the dependent variable, loads the corresponding spatial lags, and 
    transforms and generates independent variables used in the analysis.
    '''
    gwf = pd.read_table('data/GWFtscs.txt')
    if dep_var == 'failure':
        gwf = gwf[['cowcode','year','gwf_duration','gwf_fail','gwf_party','gwf_personal','gwf_military','gwf_monarch']]
        gwf.columns = ['cowcode','year','duration','failure','party','personal','military','monarchy']
    elif dep_var == 'coerce':
        gwf = gwf[['cowcode','year','gwf_duration','gwf_fail','gwf_fail_type','gwf_party','gwf_personal','gwf_military','gwf_monarch']]
        gwf['coerce'] = ((gwf['gwf_fail_type'] == 4) | (gwf['gwf_fail_type'] == 5) | (gwf['gwf_fail_type'] == 6)).astype(int)
        gwf.drop(['gwf_fail','gwf_fail_type'], axis = 1, inplace = True)
        gwf.columns = ['cowcode','year','duration','party','personal','military','monarchy','coerce']
    elif dep_var == 'auttrans':
        gwf = gwf[['cowcode','year','gwf_duration','gwf_fail','gwf_fail_subsregime','gwf_party','gwf_personal','gwf_military','gwf_monarch']]
        gwf['auttrans'] = (gwf['gwf_fail_subsregime'] == 2).astype(int)
        gwf.drop(['gwf_fail','gwf_fail_subsregime'], axis = 1, inplace = True)
        gwf.columns = ['cowcode','year','duration','party','personal','military','monarchy','auttrans']
    elif dep_var == 'demtrans':
        gwf = gwf[['cowcode','year','gwf_duration','gwf_fail','gwf_fail_subsregime','gwf_party','gwf_personal','gwf_military','gwf_monarch']]
        gwf['demtrans'] = (gwf['gwf_fail_subsregime'] == 1).astype(int)
        gwf.drop(['gwf_fail','gwf_fail_subsregime'], axis = 1, inplace = True)
        gwf.columns = ['cowcode','year','duration','party','personal','military','monarchy','demtrans']
    else:
        print 'Do not recognize this dependent variable choice'
        return None
    if lag:
        lag = pd.read_table('data/' + lag, sep = ',')
        gwf = pd.merge(gwf, lag, on = ['cowcode','year'], how = 'left')
    con = pd.read_table('data/control_variables.txt', sep = ',')
    gwf = pd.merge(gwf, con, on = ['cowcode','year'], how = 'left')
    gwf['duration'] = np.log(gwf['duration'])
    gwf['duration_military'] = gwf['duration'] * gwf['military']
    gwf['duration_personal'] = gwf['duration'] * gwf['personal']
    gwf['duration_party'] = gwf['duration'] * gwf['party']
    gwf['resource'] = np.log(gwf['resource'] + 1)
    gwf['growth'] = log_negative(gwf['growth'])
    gwf['population'] = np.log(gwf['population'])
    gwf['openness'] = np.log(gwf['openness'])
    gwf['two_year'] = round_down_any(gwf['year'], base = 2)
    gwf['three_year'] = round_down_any(gwf['year'], base = 3)
    gwf['lustrum'] = round_down_any(gwf['year'], base = 5)
    gwf['decade'] = round_down_any(gwf['year'], base = 10)
    return gwf
    
## --------------------------------------------------------------------- ##

def reg_table(estimates):
    '''
    Format the ['table'] value returned by estimator method. Zip estimates and standard errors along with 
    feature names and indicators of statistical significance at $p$ <= 0.05. The output is Nx2 data frame
    indexed by feature name. Multiple reg_tables are merged by tex_reg_table().
    '''
    est = estimates['table']['estimate'].tolist()
    filler = [''] * len(est)
    se = estimates['table']['std. error'].tolist()
    star = np.where(np.absolute(np.divide(est, se)) >= 1.96, '*', '').tolist()
    names = estimates['table'].index.tolist()
    names_se = [name + '_StdError' for name in names]
    iters = [iter(names), iter(names_se)]
    names = list(it.next() for it in itertools.cycle(iters))
    iters = [iter(est), iter(se)]
    est = list(it.next() for it in itertools.cycle(iters))
    iters = [iter(star), iter(filler)]
    stars = list(it.next() for it in itertools.cycle(iters))
    out = pd.DataFrame(zip(est, stars))
    out.columns = ['est','stars']
    out.index = names
    return out 

## --------------------------------------------------------------------- ##

def reg_summary(estimates):
    '''
    Format the regression summary values returned by the estimator method, contingent on the method used. A
    data frame is returned and appended to the corresponding table of estimates and standard errors from reg_table()
    using tex_reg_table().
    '''
    if estimates['estimator'] == 'logit':
        names = [r'\hline','$N$','Deviance','Log-likelihood','AIC','BIC']
        values = ['',estimates['n'],estimates['deviance'],estimates['loglik'],estimates['aic'],estimates['bic']]
        return pd.DataFrame(zip(values, [''] * len(names)), index = names)
    elif estimates['estimator'] == 'ols':
        names = [r'\hline','$N$','RMSE','$R^2$','Log-likelihood','AIC','BIC']
        values = ['',estimates['n'],estimates['rmse'],estimates['rsq'],estimates['loglik'],estimates['aic'],estimates['bic']]
        return pd.DataFrame(zip(values, [''] * len(names)), index = names)
    else:
        return None

## --------------------------------------------------------------------- ##

def tex_reg_table(reg_tables, reg_summaries, factors = None, file_name = ''):
    '''
    Provide list of the outputs from reg_table() and reg_summary(). Also provide list of factor names, i.e., 
    the substring corresponding to levels of a factor variable (e.g., 'year' for 'year_XXXX'). Finally, 
    provide the desired prefix for the file name, as the body of the tex table will be written to file
    to put in the final touches.
    '''
    out_table = reg_tables.pop(0)
    for table in reg_tables:
        out_table = pd.merge(out_table, table, left_index = True, right_index = True, how = 'outer')
    index_list = out_table.index.tolist()
    if factors != None:
        for f in factors:
            f = f + '_'
            index_list = [x for x in index_list if f not in x]
    out_table = out_table.ix[index_list]
    index_list = out_table.index.tolist()
    index_list = ['' if '_StdError' in x else x for x in index_list]
    out_table.index = index_list

    out_summary = reg_summaries.pop(0)
    for summary in reg_summaries:
        out_summary = pd.merge(out_summary, summary, left_index = True, right_index = True, how = 'outer')

    out_table.columns = range(out_table.shape[1])
    out_summary.columns = range(out_summary.shape[1])
    out_table = pd.concat([out_table, out_summary])
    
    est_cols = [x for x in out_table.columns if x%2 == 0]   # get estimate columns
    str_cols = [x for x in out_table.columns if x%2 != 0]   # stars
    out_table[est_cols] = out_table[est_cols].fillna('.')
    out_table[str_cols] = out_table[str_cols].replace(np.nan, '')
    out_table.columns = ['c' + str(x) for x in out_table.columns.tolist()]
    for col in out_table.columns:
        idx = out_table.columns.get_loc(col)
        out_table.insert(idx, '&' + str(idx), '&')
    out_table.insert(out_table.columns.get_loc(out_table.columns[-1]) + 1, r'\\', r'\\')
    out_table.ix['\hline'] = ''
    columns = [r'',r'&',r'\multicolumn{2}{c}{Model 1}',r'&',r'\multicolumn{2}{c}{Model 2}',r'&',r'\multicolumn{2}{c}{Model 3}',r'&',r'\multicolumn{2}{c}{Model 4}',r'&',r'\multicolumn{2}{c}{Model 5}',r'\\']
    with open(file_name + '.tex', 'w+') as f:
        f.write(' '.join(columns) + '\n')
        f.write('\\hline \n')
        f.write(out_table.to_string(header = False))
        f.close()

## --------------------------------------------------------------------- ##

def plot_fd(simulations, file_name):
    '''
    Plot the distribution of mean predicted values across observations for each simulation for simulation() method. 
    Plot the first differences between the paired values provided to the set_values argument of that 
    method. Provide the file_name for the pdf output.
    '''
    try:
        fig = plt.figure(figsize = (12,8))
        ax1 = plt.subplot2grid((2,5), (0,0), colspan = 2, rowspan = 1)
        ax2 = plt.subplot2grid((2,5), (1,0), colspan = 2, rowspan = 1, sharex = ax1)
        ax3 = plt.subplot2grid((2,5), (0,2), colspan = 3, rowspan = 2)
        ax1.hist(simulations['x0'], 15, normed = False, facecolor = 'k', alpha = 0.4)
        ax1.set_xlabel(r'Predicted Probability ($x_0$)', labelpad = 10)
        ax2.hist(simulations['x1'], 15, normed = False, facecolor = 'k', alpha = 0.4)
        ax2.set_xlabel(r'Predicted Probability ($x_1$)', labelpad = 10)
        ax3.hist(simulations['fd'], 30, normed = False, facecolor = 'k', alpha = 0.4)
        ax3.set_xlabel(r'First Differences ($x_0 - x_1$)', labelpad = 10)
        plt.tight_layout()
        fig.savefig(file_name + '.pdf')
        plt.close()
    except:
        print 'No simulations passed to plotter'
        pass

## --------------------------------------------------------------------- ##

def period_simulator(X, y, coef, periods, lag_periods, lag_var, set_values):
    '''
    Simulated first differences using for every period, focusing on interaction term between 
    period dummy and the spatial lag. Takes the simulated coefficients, the matrix of predictors, 
    the vector of outcomes, the column names for the period dummies, the column names for the 
    period-lag interaction terms, and counterfactual values for the spatial lag at which to evaluate
    the first differences. Simulation strategy is the same as described in the simulation method above.
    '''
    simulations = {}
    for period, lag_period in zip(periods, lag_periods):
        non_periods = [x for x in periods if x != period]
        non_lag_periods = [x for x in lag_periods if x != lag_period]
        for non_period in non_periods:
            X[non_period] = 0
        for non_lag_period in non_lag_periods:
            X[non_lag_period] = 0
        X[period] = 1
        out = []
        for value in set_values:
            X[lag_var] = value
            X[lag_period] = X[lag_var].multiply(X[period], axis = 'index')
            linpreds = np.dot(X, coef.T)
            probs = 1 / (1 + np.exp(-1 * linpreds))
            out.append(probs)
        fd = (out[0] - out[1]).mean(axis = 0) # average fd across obs for each simulated beta
        simulations[period] = fd
    return pd.DataFrame(simulations)

## --------------------------------------------------------------------- ##

def plot_period_simulations(simulations, periods, file_name, id_vars, var_name):
    '''
    Plot the simulated first differences for each five-year period. Takes the matrix of predicted 
    probabilities from the period_simulator(), the column names for the period dummies, and 
    the file name to be given to pdf output, which is a seaborn boxplot of probabilities for 
    every time period.
    '''
    sns.set_style('whitegrid')
    melted_sims = pd.melt(simulations, id_vars = id_vars, var_name = var_name, value_name = 'First Differences')
    melted_sims[var_name] = melted_sims[var_name].str.split('_').str.get(-1)    # years only, omit prefix
    melted_sims = melted_sims.replace({'Lag' : {'demtrans' : 'Democratic',
                                                'auttrans' : 'Autocratic',
                                                'coerce' : 'Coerce',
                                                'failure' : 'Failure'}})
    labs = melted_sims[var_name].unique().tolist()
    labs = ['' if i%2 == 0 else x for i, x in enumerate(labs)]    # use every other period label for aesthetic purposes
    plt.figure(figsize = (12, 8))
    sns.set_style('whitegrid')
    f = sns.factorplot('Lustrum', col = 'Lag', y = 'First Differences', data = melted_sims, col_wrap = 2, kind = 'box')
    f.set_xlabels('')
    g = f.facet_axis(1,1)
    g.set_xticklabels(labels = labs)
    plt.savefig(file_name + '.pdf')
    plt.close()

## --------------------------------------------------------------------- ##

def evaluate_classifier(n, da_instance, spec_no):
    '''
    Perform 5-fold cross-validation and extract performance metrics for specifications.
    '''
    output_list = [da_instance.classify() for i in range(n)]
    auroc = {}
    auroc[spec_no] = [x['auroc'] for x in output_list]
    f = {}
    f[spec_no] = [x['f1'] for x in output_list]
    return {'auroc' : auroc,
            'f' : f}

## --------------------------------------------------------------------- ##

def output_classifier_metrics(file_name_root, base, test, test2 = None, metrics = ['auroc']):
    '''
    Merge restricted and unrestricted specifications, melt the merged data frame, and 
    write to file for plotting. The plotter should be in this function, but seaborn functions
    do not work well with Macs, so the heavy lifting is done on Mac, the plotting on Windows.
    Optionally take a second test set, where time-varying effect of spatial lag is tested. 
    '''
    for metric in metrics:
        base_metrics = []
        for model in base:
            base_metrics.append(pd.DataFrame(model[metric]))
        base_metrics = pd.concat(base_metrics, axis = 1)
        base_metrics['type'] = 'restricted'

        test_metrics = []
        for model in test:
            test_metrics.append(pd.DataFrame(model[metric]))
        test_metrics = pd.concat(test_metrics, axis = 1)
        test_metrics['type'] = 'full'

        if test2:
            test2_metrics = []
            for model in test2:
                test2_metrics.append(pd.DataFrame(model[metric]))
            test2_metrics = pd.concat(test2_metrics, axis = 1)
            test2_metrics['type'] = 'interaction'

            metric_data = pd.concat([base_metrics, test_metrics, test2_metrics], axis = 0)
        else:
            metric_data = pd.concat([base_metrics, test_metrics], axis = 0)

        metric_data = pd.melt(metric_data, id_vars = 'type', var_name = 'model', value_name = metric)
        metric_data.rename(columns = {'type' : ''}, axis = 1, inplace = True)
        metric_data.to_csv('figures/cross_val/' + metric + '/' + file_name_root + '.txt', sep = ',')

## --------------------------------------------------------------------- ##
        
def plot_classifier_metrics(path, extension, metric):
    '''
    Extract all the data files with classification metrics for given path, produce the 
    plots comparing restricted and unrestricted specifications. 
    '''
    file_names = glob.glob(path + '*' + extension)
    for name in file_names:
        data = pd.read_csv(name, sep = ',', index_col = 0)
        data.columns = ['','model',metric]

        name = name[:name.find(extension)]
        name = name + '.pdf'

        if 'lustrum' in name:
            plt.figure(figsize = (12,8))
            sns.set_style('whitegrid')
            sns.set_palette(['0.30','0.50','0.70'])
            f = sns.factorplot('model', hue = '', y = metric, data = data, kind = 'box', hue_order = ['restricted','full','interaction'])
            f.set_xlabels('')
            f.set_ylabels('')
            plt.savefig(name)
            plt.close()
        else:
            plt.figure(figsize = (12,8))
            sns.set_style('whitegrid')
            sns.set_palette(['0.35','0.65'])
            f = sns.factorplot('model', hue = '', y = metric, data = data, kind = 'box', hue_order = ['restricted','full'])
            f.set_xlabels('')
            f.set_ylabels('')
            plt.savefig(name)
            plt.close()
        
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
