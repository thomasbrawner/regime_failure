


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
        data_frame.dropna(axis = 0, inplace = True)
        data_frame.reset_index(drop = True, inplace = True)
        import statsmodels.api as sm
        if factors != None:
            from pandas import get_dummies, merge
            for f in factors:
                dummy = get_dummies(data_frame[f], prefix = f)
                data_frame = merge(data_frame, dummy, left_index = True, right_index = True, how = 'left')
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
        import statsmodels.api as sm
        import numpy as np
        from pandas import DataFrame
        if method == 'ols':
            model = sm.OLS(self.y, self.X).fit()
            coefs = np.round(model.params, 3)
            ses = np.round(model.bse, 3)
            pval = np.round(model.pvalues, 3)
            tval = np.round(model.tvalues, 3)
            conf = np.round(model.conf_int(), 3)
            table = pd.concat([coefs, ses, pval, tval, conf], axis = 1)
            table.columns = ['estimate','std. error','p-value','t','0.025','0.975']
            aic = round(model.aic, 3)
            bic = round(model.bic, 3)
            rsq = round(model.rsquared, 3)
            lik = round(model.llf, 3)
            rmse = round(np.sqrt(model.mse_resid), 3)
            obs = int(model.nobs)
            out = {'table' : table,
                   'rmse' : rmse,
                   'rsq' : rsq,
                   'loglik' : lik,
                   'aic' : aic,
                   'bic' : bic,
                   'n' : obs}
            return out
        elif method == 'logit':
            model = sm.GLM(self.y, self.X, family = sm.families.Binomial()).fit()
            coefs = np.round(model.params, 3)
            ses = np.round(model.bse, 3)
            pval = np.round(model.pvalues, 3)
            tval = np.round(model.tvalues, 3)
            conf = np.round(model.conf_int(), 3)
            table = pd.concat([coefs, ses, pval, tval, conf], axis = 1)
            table.columns = ['estimate','std. error','p-value','t','0.025','0.975']
            aic = round(model.aic, 3)
            bic = round(model.bic, 3)
            lik = round(model.llf, 3)
            obs = int(model.nobs)
            dev = round(model.deviance, 3)
            out = {'table' : table,
                   'deviance' : dev,
                   'loglik' : lik,
                   'aic' : aic,
                   'bic' : bic,
                   'n' : obs}
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
        import statsmodels.api as sm
        import numpy as np
        import pandas as pd
        if seed:
            np.random.seed(seed)
        if method == 'logit':
            model = sm.GLM(self.y, self.X, family = sm.families.Binomial()).fit()
            coefs = model.params
            vcov = model.cov_params()
            x = np.random.multivariate_normal(coefs, vcov, nsims)
            out = []
            for val in set_values:
                self.X[set_var] = val
                linpreds = np.dot(self.X, x.T)
                probs = 1 / (1 + np.exp(-1 * linpreds))
                out.append(pd.DataFrame(probs).describe(percentiles = [.50]).ix[['50%']])
            out = pd.concat(out, axis = 0).T
            out.columns = ['val0','val1']
            out['fd'] = out['val0'] - out['val1']
            return out
        elif method == 'ols':
            model = sm.OLS(self.y, self.X).fit()
            coefs = model.params
            vcov = model.cov_params()
            x = np.random.multivariate_normal(coefs, vcov, nsims)
            out = []
            for val in set_values:
                self.X[set_var] = val
                linpreds = np.dot(self.X, x.T)
                out.append(pd.DataFrame(linpreds).describe(percentiles = [.50]).ix[['50%']])
            out = pd.concat(out, axis = 0).T
            out.columns = ['val0','val1']
            out['fd'] = out['val0'] - out['val1']
            return out
        else:
            print 'method must be one of ["logit","ols"]'
            return None

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
        from sklearn.cross_validation import StratifiedKFold
        from sklearn import metrics
        from pandas import DataFrame
        splits = StratifiedKFold(self.y, n_folds = folds, shuffle = True, random_state = seed)
        out = []
        for train, test in splits:
            x_train, x_test, y_train, y_test = self.X.ix[train], self.X.ix[test], self.y.ix[train], self.y.ix[test]
            if method == 'logit':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression().fit(x_train, y_train)
            elif method == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators = 1000, random_state = seed).fit(x_train, y_train)
            elif method == 'svm':
                from sklearn.svm import SVC
                model = SVC(kernel = 'rbf', probability = True, random_state = seed).fit(x_train, y_train)
            elif method == 'naive_bayes':
                from sklearn.naive_bayes import BernoulliNB
                model = BernoulliNB().fit(x_train, y_train)
            elif method == 'boosting':
                from sklearn.ensemble import AdaBoostClassifier
                model = AdaBoostClassifier(n_estimators = 100, random_state = seed).fit(x_train, y_train)
            else:
                print 'method must be one of ["logit","random_forest","svm","naive_bayes","boosting"]'
                return None
            preds = zip(test.tolist(), y_test.tolist(), model.predict(x_test).tolist(), model.predict_proba(x_test)[:,1:].flatten().tolist())
            out += preds
        out = DataFrame(out)
        out.columns = ['index','observed','classified','probability']
        out.set_index('index', inplace = True)
        out.sort_index(inplace = True)
        acc = metrics.accuracy_score(out['observed'], out['classified'], normalize = True)
        confusion = metrics.confusion_matrix(out['observed'], out['classified'])
        f1 = metrics.f1_score(out['observed'], out['classified'])
        fbeta_list = []
        for b in beta:
            fbeta_list += [(b, metrics.fbeta_score(out['observed'], out['classified'], beta = b, average = 'weighted'))]
        fbeta = DataFrame(fbeta_list)
        fbeta.columns = ['beta','score']
        precision = metrics.precision_score(out['observed'], out['classified'])
        recall = metrics.recall_score(out['observed'], out['classified'])
        roc = metrics.roc_curve(out['observed'], out['probability'])
        roc = DataFrame([roc[0], roc[1], roc[2]]).T
        roc.columns = ['fpr','tpr','threshold']
        auroc = metrics.roc_auc_score(out['observed'], out['probability'])
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
            
