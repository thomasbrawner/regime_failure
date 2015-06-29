## ------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------- ##
## dissertation_programs.py
## tb 15 jun 2015
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

## ------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------- ##

def round_down_any(x, base):
    """
    Round down to nearest <base>.
    """
    round_down = np.vectorize(floor)
    x = np.array((x / base), dtype = float)
    out = round_down(x)
    return pd.Series(out * base).astype(int)

## ------------------------------------------------------------------------- ##

def gelman_rescale(x):
    """
    Rescale numeric attributes as Gelman recommends.
    """
    mu = np.mean(x)
    sd = np.std(x)
    return (x - mu) / (2 * sd)

## ------------------------------------------------------------------------- ##

class DataAnalysis(object):

    def __init__(self, dframe, depvar, factors = None):
        """
        Create formatted data frame for analysis, return X and y. 
    
        Arguments: 
            - dframe : pandas data frame subset to columns in the analysis
            - depvar : column name of dependent variable
            - factors : list of factor variables in the analysis, which are transformed to dummy variables
            - rescale : list of numeric variables to rescale
        Values: 
            - X : design matrix
            - y : dependent variable
        """ 
        # list-wise delete and reset the index
        dframe = dframe.dropna()
        dframe = dframe.reset_index(drop = True)
        
        # create dummy variables from provided factor list
        if factors is not None:
            for f in factors:
                dummy = pd.get_dummies(dframe[f], prefix = f)
                dframe = pd.merge(dframe, dummy, left_index = True, right_index = True, how = 'left')
                dframe = dframe.drop(f, axis = 1)
        
        # set y and X
        features = [x for x in dframe.columns.tolist() if x not in [depvar]]
        self.X = dframe[features]
        self.y = dframe[depvar]
    
    ## --------------------------------------------------------------------- ##
    
    def logit_optimal_c(self):
        """
        Find optimal regularization strength, L2 penalty, in 5-fold cross-validation
        
        Arguments: 
            - Cs : integer, number of values to search over
        Values: 
            - Cs_ : the values of C used
            - C_ : the optimal C
        """
        auroc = make_scorer(roc_auc_score)
        model = LogisticRegressionCV(Cs = 100, cv = 5, fit_intercept = True, 
                                     scoring = auroc, penalty = 'l2', solver = 'lbfgs')
        fit = model.fit(self.X, self.y)
        return fit.Cs_, fit.C_
    
    ## --------------------------------------------------------------------- ##

    def plot_regularization_path(self, Cs = None, optimal_c = None):
        """
        Plot coefficient magnitudes over regularization strength. Parameters taken 
        from logit_optimal_c(). Prints the paths. 
        
        Arguments:
            - Cs : array of regularization strengths
            - optimal_c : value with best performance in CV
        """
        # logit
        logit = LogisticRegression(fit_intercept = True)

        # obtain coefficients over penalty values
        coefs = []
        for c in Cs:
            logit.set_params(C = c)
            logit.fit(self.X, self.y)
            coefs.append(logit.coef_)
        coefs = np.concatenate(coefs)    
    
        # plot
        plt.figure(figsize = (14, 10))
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        ax = plt.gca()
        ax.set_color_cycle(np.repeat('0.40', self.X.shape[1]))
        ax.plot(Cs, coefs, '--')
        ax.axvline(x = optimal_c, color = '0.40')  
        ax.set_xscale('log')
        plt.xlabel('$C$', fontsize = 18)
        plt.ylabel(r'$\beta$', fontsize = 18, rotation = 0)
        plt.axis('tight')
        
    ## --------------------------------------------------------------------- ##

    def logit_bootstrap_estimates(self, nboot = 1000, optimal_c = None, factors = None, var_names = None):
        """
        Box plots, distribution of bootstrap coefficient estimates for each variable in model. 
        
        Arguments:
            - nboot : number of bootstrap samples
            - optimal_c : value of regularization parameter with best CV performance
            - factors : list of factor variables in the analysis, to be omitted from this plot
            - var_names : dict, column names to appear in the plot
        """
        # logit
        logit = LogisticRegression(fit_intercept = True)

        # get the estimates from resampled data
        coefs = []
        for i in range(nboot):    
            iX, iy = resample(self.X, self.y)
            logit.set_params(C = optimal_c)
            logit.fit(iX, iy)
            coefs.append(logit.coef_)
        coefs = np.concatenate(coefs)
    
        # make copy of full results to send to first differences
        coefs_out = coefs.copy()
        
        # convert to data frame, use col names from X
        coefs = pd.DataFrame(coefs)
        coefs.columns = self.X.columns
        
        # remove factors from output
        if factors is not None:
            remove_columns = [x for x in coefs.columns.tolist() for y in factors if y in x]                
            coefs = coefs.drop(remove_columns, axis = 1)
            
        if var_names is not None:
            coefs = coefs.rename(columns = var_names)
        
        # boxplots 
        plt.figure(figsize = (14, 5))
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        ax = plt.gca()
        aspect = self.X.shape[1] * 0.2
        sns.boxplot(coefs, vert = False, color = '0.60')
        ax.axvline(x = 0, color = '0.60', linestyle = '--')  
        plt.xlabel(r'$\beta$', fontsize = 18)
        
        return coefs_out
        
    ## --------------------------------------------------------------------- ##

    def logit_first_differences(self, boot_estimates, variable):
        
        edata = self.X.copy()
        
        # set counterfactual values, 25th and 75th percentiles
        x0, x1 = edata[variable].describe().ix[['25%','75%']]
        
        # evaluate predicted probs for counterfactuals and bootstrap estimates
        out = []
        for val in [x0, x1]:
            edata[variable] = val
            linpreds = np.dot(edata, boot_estimates.T)
            out.append(1 / (1 + np.exp(-1 * linpreds)))
        
        # first differences
        fd = out[1] - out[0]
        
        # median fd by observation
        fd = np.median(fd, axis = 0)
        
        # mean fd
        fd_mu = np.round(np.mean(fd), 3)
        
        # plot distribution of first differences by regime type
        plt.figure(figsize = (14,8))
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.hist(fd, 40, normed = False, facecolor = '0.60')
        plt.annotate(r'$\mu =$ %s' % fd_mu, xy = (0.1, 0.9), xycoords = 'axes fraction', size = 16)
        plt.xlabel(r'$\Delta$ Pr$(Y = 1) ~|~ x_0 \rightarrow x_1$', fontsize = 18, labelpad = 15)
        plt.ylabel('')
    
## ------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------- ##
