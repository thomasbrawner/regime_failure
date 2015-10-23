## ------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------- ##
## dissertation_programs.py
## tb 15 jun 2015, last update 13 aug 2015
## ------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------- ##

from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler

## ------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------- ##

def round_down_any(x, base):
    """
    Round down to nearest <base>.
    """
    out = np.array(x) / base * base
    return pd.Series(out)

## ------------------------------------------------------------------------- ##

class DataAnalysis(object):

    def __init__(self, dframe, depvar, factors = None, scale = False):
        """
        Create formatted data frame for analysis, return X and y. 
    
        Arguments: 
            - dframe : pandas data frame subset to columns in the analysis
            - depvar : column name of dependent variable
            - factors : list of factor variables in the analysis, which are transformed to dummy variables
        """ 
        dframe = dframe.dropna()
        dframe = dframe.reset_index(drop = True)
        
        if factors is not None:
            for f in factors:
                dummy = pd.get_dummies(dframe[f], prefix = f)
                dframe = pd.merge(dframe, dummy, left_index = True, right_index = True, how = 'left')
                dframe = dframe.drop(f, axis = 1)

        self.y = dframe.pop(depvar).values
        self.X = dframe.values
        if scale:
            self.X = StandardScaler().fit_transform(self.X) 
        self.features = dframe.columns 
        self.factors = factors

        '''
        if 'year' in self.features.tolist():
            var_idx = self.features.tolist().index('year')
            self.X[:, var_idx] = dframe['year'].values 
        '''

        if 'year' in self.features.tolist():
        	self.year = dframe['year'].values 

    def logit_optimal_c(self):
        """
        Find optimal regularization strength, L2 penalty, in 5-fold cross-validation.
        Sets attributes
            self.Cs : range of 100 values for regularization parameter 
            self.C_optimal : value of regularization with best CV performance 
        """
        auroc = make_scorer(roc_auc_score)
        model = LogisticRegressionCV(Cs = 100, cv = 5, fit_intercept = True, 
                                     scoring = auroc, penalty = 'l2', solver = 'lbfgs')
        fit = model.fit(self.X, self.y)
        self.Cs = fit.Cs_ 
        self.C_optimal = fit.C_
    

    def plot_regularization_path(self):
        """
        Plot coefficient magnitudes over regularization strength. Parameters taken 
        from logit_optimal_c(). Plots the paths. 
        """
        logit = LogisticRegression(fit_intercept = True)

        coefs = []
        for c in self.Cs:
            logit.set_params(C = c)
            logit.fit(self.X, self.y)
            coefs.append(logit.coef_)
        coefs = np.concatenate(coefs)    
    
        plt.figure(figsize = (14, 10))
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 14
        ax = plt.gca()
        ax.set_color_cycle(np.repeat('0.40', self.X.shape[1]))
        ax.plot(self.Cs, coefs, '--')
        ax.axvline(x = self.C_optimal, color = '0.40')  
        ax.set_xscale('log')
        plt.xlabel('$C$', fontsize = 18, labelpad = 12)
        plt.ylabel(r'$\beta$', fontsize = 18, rotation = 0, labelpad = 12)
        plt.axis('tight')
        

    def logit_bootstrap_estimates(self, nboot = 1000, var_names = None):
        """
        Box plots, distribution of bootstrap coefficient estimates for each variable in model. 
        
        Arguments:
            - nboot : number of bootstrap samples
            - var_names : dict, column names to appear in the plot
        """
        logit = LogisticRegression(fit_intercept = True)

        coefs = []
        for i in range(nboot):    
            iX, iy = resample(self.X, self.y)
            logit.set_params(C = self.C_optimal)
            logit.fit(iX, iy)
            coefs.append(logit.coef_)
        coefs = np.concatenate(coefs)
    
        coefs_out = coefs.copy()
        coefs = pd.DataFrame(coefs)
        coefs.columns = self.features 

        if self.factors is not None:
            remove_columns = [x for x in coefs.columns.tolist() for y in self.factors if y in x]                
            coefs = coefs.drop(remove_columns, axis = 1)
            
        if var_names is not None:
            coefs = coefs.rename(columns = var_names)
         
        plt.figure(figsize = (14, 5))
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 14
        ax = plt.gca()
        aspect = self.X.shape[1] * 0.2
        sns.boxplot(coefs, vert = False, color = '0.60')
        ax.axvline(x = 0, color = '0.60', linestyle = '--')  
        plt.xlabel(r'$\beta$', fontsize = 18, labelpad = 12)
        
        self.bootstrap_coefs = coefs_out
        

    def logit_first_differences(self, variable):
        """
        Plot histogram of differences in predicted probability of event given 
        shift from 25th to 75th percentile in specified variable, all other 
        features kept at their observed values, using the boostrap coefficient
        estimates.  

        Arguments:
            - variable : variable 
        """
        
        edata = self.X.copy()
        
        # set counterfactual values, 25th and 75th percentiles
        var_idx = self.features.tolist().index(variable)
        x0 = np.percentile(edata[:, var_idx], 25)
        x1 = np.percentile(edata[:, var_idx], 75)

        # evaluate predicted probs for counterfactuals and bootstrap estimates
        out = []
        for val in [x0, x1]:
            edata[:, var_idx] = val
            linpreds = np.dot(edata, self.bootstrap_coefs.T)
            out.append(1 / (1 + np.exp(-1 * linpreds)))
        
        # first differences
        fd = out[1] - out[0]
        fd = np.median(fd, axis = 0)
        fd_mu = np.round(np.mean(fd), 3)
        
        # plot distribution of first differences by regime type
        plt.figure(figsize = (14,8))
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 14
        plt.hist(fd, 40, normed = True, facecolor = '0.60')
        plt.annotate(r'$\mu =$ %s' % fd_mu, xy = (0.8, 0.9), xycoords = 'axes fraction', size = 18)
        plt.xlabel(r'$\Delta$ Pr$(Y = 1) ~|~ x_0 \rightarrow x_1$', fontsize = 18, labelpad = 15)
        plt.ylabel('')


    def sequential_forecaster(self, classifier, year):
        """
        Train classifier up to year, then predict for year.

        Arguments:
            - classifier: optimized classifier from grid search 
            - year: dict, variable : value
            
        Values:
            - array [
                test_y: observed values for forecasted year,
                preds: forecast probabilities of event 
                ]
        """
        # train, test split on year
        mask = self.year < year.values()[0]
        train_x, train_y = self.X[mask], self.y[mask]
        mask2 = self.year == year.values()[0]
        test_x, test_y = self.X[mask2], self.y[mask2]
        
        # remove year
        var_idx = self.features.tolist().index(year.keys()[0])
        train_x = np.delete(train_x, var_idx, 1)
        test_x = np.delete(test_x, var_idx, 1)

        # generate predicted probs
        classifier.fit(train_x, train_y)
        preds = classifier.predict_proba(test_x)[:,1:].flatten()
        years = np.repeat(year.values()[0], len(preds))
        return np.array([years, test_y, preds]).T 


    def train_test_split_year(self, year):

        mask = self.year < year.values()[0]
        train_x, train_y = self.X[mask], self.y[mask]
        mask2 = self.year == year.values()[0]
        test_x, test_y = self.X[mask2], self.y[mask2]

        # remove year
        var_idx = self.features.tolist().index(year.keys()[0])
        train_x = np.delete(train_x, var_idx, 1)
        test_x = np.delete(test_x, var_idx, 1)

        return train_x, train_y, test_x, test_y

## ------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------- ##
