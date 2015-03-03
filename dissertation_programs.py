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
from sklearn import cross_validation, linear_model, ensemble, svm, metrics
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
            - method: one of ['logit','random_forest','svm','lasso','boosting']
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
                model = ensemble.RandomForestClassifier(n_estimators = 1000, random_state = seed).fit(x_train, y_train)
            elif method == 'svm':
                model = svm.SVC(kernel = 'rbf', probability = True, random_state = seed).fit(x_train, y_train)
            elif method == 'lasso':
                model = linear_model.LogisticRegression(penalty = 'l1').fit(x_train, y_train)
            elif method == 'boosting':
                model = ensemble.AdaBoostClassifier(n_estimators = 100, random_state = seed).fit(x_train, y_train)
            else:
                print 'method must be one of ["logit","random_forest","svm","lasso","boosting"]'
                return None
            preds = zip(test.tolist(), y_test.tolist(), model.predict(x_test).tolist(), model.predict_proba(x_test)[:,1:].flatten().tolist())
            out += preds
        out = pd.DataFrame(out)
        out.columns = ['index','observed','classified','probability']
        out.set_index('index', inplace = True)
        out.sort_index(inplace = True)
        fbeta_list = []
        for b in beta:
            fbeta_list += [(b, metrics.fbeta_score(out['observed'], out['classified'], beta = b, average = 'weighted'))]
        fbeta = pd.DataFrame(fbeta_list)
        fbeta.columns = ['beta','score']
        roc = metrics.roc_curve(out['observed'], out['probability'])
        roc = pd.DataFrame([roc[0], roc[1], roc[2]]).T
        roc.columns = ['fpr','tpr','threshold']
        out_dict = {'predictions' : out,
                    'accuracy' : metrics.accuracy_score(out['observed'], out['classified'], normalize = True),
                    'f1' : metrics.f1_score(out['observed'], out['classified']),
                    'fbeta' : fbeta,
                    'precision' : metrics.precision_score(out['observed'], out['classified']),
                    'recall' : metrics.recall_score(out['observed'], out['classified']),
                    'confusion' : metrics.confusion_matrix(out['observed'], out['classified']),
                    'roc' : roc,
                    'auroc' : metrics.roc_auc_score(out['observed'], out['probability'])}
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

def add_period_interaction(brada, period, lag_var):
    '''
    Create dummy matrix and generate interaction terms between provided variable and each dummy variable in the matrix.
    Returns a modified instance of the brada object for data analysis. For use with the predictor_performance() function
    for specifying predictor matrix for interactions between 5-year periods and spatial lags.
    '''
    lustrum_dummies = pd.get_dummies(brada.X[period], prefix = period)
    brada.X = pd.merge(brada.X, lustrum_dummies, left_index = True, right_index = True, how = 'left')
    brada.X = brada.X.drop(period, axis = 1)
    periods = [x for x in brada.X.columns.tolist() if period in x]
    for col in periods:
        brada.X['lag_' + col] = brada.X[lag_var].multiply(brada.X[col], axis = 'index')
    return brada
    
## --------------------------------------------------------------------- ##

def predictor_performance(dframe, dep_var, period, restricted, full = {}, interaction = True, metric = ['auroc'], method = 'logit', folds = 3):
    '''
    Evaluate the performance of classifiers, comparing restricted to unrestricted specifications. Provide list of metrics and the classification
    method. Returns list of dictionaries, where each dictionary has model information and performance output. 
    '''
    performance_out = []
    for idx in range(len(restricted)):
        factor_list = [x for x in ['two_year','lustrum','region','cowcode'] if x in restricted[idx]]
        
        # create data objects for analysis
        dat_rest = diss.brada(dframe[restricted[idx]], dep_var = dep_var, factors = factor_list, constant = False)
        dat_alt = {spec : diss.brada(dframe[full[spec][idx]], dep_var = dep_var, factors = factor_list, constant = False) for spec in full}

        # run classifier on each specification
        seed = random.randint(1, 1000000)
        classifiers = {'restricted' : dat_rest.classify(method = method, folds = folds, seed = seed),
                       'failure' : dat_alt['lagfail'].classify(method = method, folds = folds, seed = seed),
                       'coerce' : dat_alt['lagcoer'].classify(method = method, folds = folds, seed = seed),
                       'autocratic' : dat_alt['lagauto'].classify(method = method, folds = folds, seed = seed),
                       'democratic' : dat_alt['lagdemo'].classify(method = method, folds = folds, seed = seed)}

        # model information
        performance = {'dependent' : dep_var,
                       'model' : idx,
                       'period' : period}

        # add performance metrics for each model
        for m in metric:
            for key, value in classifiers.iteritems():
                out_name = m + '_' + key
                performance[out_name] = value[m]
        performance_out.append(performance)

    # if 5-year period and interaction needed, also check performance with lag-period interaction
    if period == 'lustrum' and interaction:
        for idx in range(len(restricted)):
            factor_list = [x for x in ['region','cowcode'] if x in restricted[idx]]

            # create data objects for analysis, new factor list
            dat_rest = diss.brada(dframe[restricted[idx]], dep_var = dep_var, factors = factor_list, constant = False)
            dat_alt = {spec : diss.brada(dframe[full[spec][idx]], dep_var = dep_var, factors = factor_list, constant = False) for spec in full}
            
            # run classifier on each specification, with interactions
            seed = random.randint(1, 1000000)
            classifiers = {'restricted' : dat_rest.classify(method = method, folds = folds, seed = seed),
                           'failure' : add_period_interaction(dat_alt['lagfail'], period, 'lag_failure').classify(method = method, folds = folds, seed = seed),
                           'coerce' : add_period_interaction(dat_alt['lagcoer'], period, 'lag_coerce').classify(method = method, folds = folds, seed = seed),
                           'autocratic' : add_period_interaction(dat_alt['lagauto'], period, 'lag_auttrans').classify(method = method, folds = folds, seed = seed),
                           'democratic' : add_period_interaction(dat_alt['lagdemo'], period, 'lag_demtrans').classify(method = method, folds = folds, seed = seed)}

            # model information
            performance = {'dependent' : dep_var,
                           'model' : idx,
                           'period' : 'lag_lustrum'}

            # add performance metrics for each model
            for m in metric:
                for key, value in classifiers.iteritems():
                    out_name = m + '_' + key
                    performance[out_name] = value[m]
            performance_out.append(performance)
                
    return performance_out

## --------------------------------------------------------------------- ##

def resample_data(df):
    '''
    Sample observations in the data frame with replacement. The default returns a 
    data frame of the same dimensions. This function is used to resample the indices of 
    the design matrix in classifier_predicted_probabilities().
    '''
    indices = np.random.choice(df.index, size = df.shape[0], replace = True)
    return df.ix[indices]

## --------------------------------------------------------------------- ##

def generate_dummies(df, unit, interaction = False, conditioning_var = None, conditioning_prefix = None):
    '''
    Generate dummy matrix for given factor, optionally generating an interaction term for each level of the factor
    and a given conditioning variable. Used by classifier_predicted_probabilities().
    '''
    dummies = pd.get_dummies(df[unit], prefix = unit)
    df2 = pd.merge(df, dummies, left_index = True, right_index = True, how = 'left')
    df2 = df2.drop(unit, axis = 1)
    dummy_names = [x for x in df2.columns.tolist() if unit in x]
    if interaction:
        if not conditioning_var:
            print 'Please supply a conditioning variable to generate interaction with the unit indicator'
            return None
        if not conditioning_prefix:
            print 'Please supply a prefix for the interaction term'
            return None
        for col in dummy_names:
            df2[conditioning_prefix + col] = df2[conditioning_var].multiply(df2[col], axis = 'index')
    return df2

## --------------------------------------------------------------------- ##

def period_classifier_probs_v1(X, classifier, periods, lag_var, set_values):
    '''
    This version evaluates predicted probabilities for the random forest classifier. It does not
    require specifying a multiplicative interaction between the spatial lag and the period indicator.
    This function is executed in classifier_predicted_probabilities() when interaction is False.
    '''
    out = []
    for period in periods:
        non_periods = [x for x in periods if x != period]
        for non_period in non_periods:
            X[non_period] = 0
        X[period] = 1
        for value in set_values:
            X[lag_var] = value
            probs = classifier.predict_proba(X)[:,1:]
            out.append({'period' : period[(period.find('_') + 1):],
                        'lag_value' : value,
                        'mean_prob' : np.mean(probs),
                        'median_prob' : np.median(probs)})
    return out

## --------------------------------------------------------------------- ##

def period_classifier_probs_v2(X, classifier, periods, lag_periods, lag_var, set_values):
    '''
    This version evaluates predicted probabilities for the classifiers. It specifies the multiplicative 
    interaction between the sptail lag and the period indicator. This function is executed in 
    classifier_predicted_probabilities() when interaction is True.
    '''
    out = []
    for period, lag_period in zip(periods, lag_periods):
        non_periods = [x for x in periods if x != period]
        non_lag_periods = [x for x in lag_periods if x != lag_period]
        for non_period in non_periods:
            X[non_period] = 0
        for non_lag_period in non_lag_periods:
            X[non_lag_period] = 0
        X[period] = 1
        for value in set_values:
            X[lag_var] = value
            X[lag_period] = X[lag_var].multiply(X[period], axis = 'index')
            probs = classifier.predict_proba(X)[:,1:]
            out.append({'period' : period[(period.find('_') + 1):],
                        'lag_value' : value,
                        'mean_prob' : np.mean(probs),
                        'median_prob' : np.median(probs)})
    return out

## --------------------------------------------------------------------- ##

def classifier_predicted_probabilities(spec, dep_var, unit, lag_var, factors, interaction = True, method = 'logit'):
    '''
    Generate bootstrap predicted probabilities for sklearn classifiers. Return data frame with indicators for period and 100 mean and median 
    predicted probabilities for each period across counterfactual values of the provided spatiotemporal lag.
    '''
    data = format_data(dep_var = dep_var, lag = 'spatial_lags_' + dep_var + '.txt')
    data = data[spec]
    if interaction:
        data = generate_dummies(data, unit, interaction, conditioning_var = lag_var, conditioning_prefix = 'lag_')
    else:
        data = generate_dummies(data, unit)

    lag_range = data[lag_var].describe().ix[['25%','75%']]
    lag_range = np.linspace(lag_range[0], lag_range[1], num = 10)

    periods = [x for x in data.columns.tolist() if unit in x]
    if interaction:
        lag_periods = [x for x in data.columns.tolist() if 'lag_' + unit in x]

    dd = brada(data, dep_var = dep_var, factors = factors, constant = False)

    if method == 'logit':
        classifier = linear_model.LogisticRegression().fit(dd.X, dd.y)
    elif method == 'lasso':
        classifier = linear_model.LogisticRegression(penalty = 'l1').fit(dd.X, dd.y)
    elif method == 'random_forest':
        classifier = ensemble.RandomForestClassifier(n_estimators = 1000).fit(dd.X, dd.y)
    else:
        print 'classifier method not supported'
        return None

    output = []
    for i in range(100):
        X = resample_data(dd.X)
        if interaction:
            output += period_classifier_probs_v2(X, classifier, periods, lag_periods, lag_var, lag_range)
        else:
            output += period_classifier_probs_v1(X, classifier, periods, lag_var, lag_range)

    return pd.DataFrame(output)
          
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
