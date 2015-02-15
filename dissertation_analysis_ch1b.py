## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## dissertation_analysis_ch1b.py
## plot simulated first differences for change in spatial lag conditional
## on five-year periods over time. 
## tb 9 feb 2015, last update 10 feb 2015
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##

import dissertation_programs as diss
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import statsmodels.api as sm
import sys

## ------------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------------- ##
## dependent variables, spatial lags, control variables, and set the period

dep_vars = ['failure','coerce','auttrans','demtrans']
lag_vars = ['lag_failure','lag_coerce','lag_auttrans','lag_demtrans']
period = 'lustrum'
controls = ['duration','party','personal','military','duration_party','duration_personal','duration_military'] + [period]

## ------------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------------- ##
## loop over specifications, spatial lag with geographic weights


for dep_var in dep_vars:
    data = diss.format_data(dep_var = dep_var, lag = 'spatial_lags_' + dep_var + '.txt')
    simulations = []
    for lag_var in lag_vars:
        dat = data[[dep_var] + [lag_var] + controls]
        dat = dat[dat[period] < 2010]

        period_dummies = pd.get_dummies(dat[period], prefix = period)
        dat = pd.merge(dat, period_dummies, left_index = True, right_index = True, how = 'left')
        dat = dat.drop([period], axis = 1)
        dat = dat.dropna(axis = 0)

        periods = [x for x in dat.columns.tolist() if period in x]
        for col in periods:
            dat['lag_' + col] = dat[lag_var].multiply(dat[col], axis = 'index')
        lag_periods = [x for x in dat.columns.tolist() if 'lag_' + period in x]
        
        dd = diss.brada(dat, dep_var = dep_var, constant = False)
        x0, x1 = np.array(dd.X[lag_var].describe().ix[['25%','75%']]).flatten()
        m = sm.GLM(dd.y, dd.X, family = sm.families.Binomial()).fit()
        est = m.params
        vcov = m.cov_params()
        betas = np.random.multivariate_normal(est, vcov, 1000)
        sims = diss.period_simulator(dd.X, dd.y, betas, periods, lag_periods, lag_var, [x0, x1])
        sims['Lag'] = lag_var.split('_')[1]
        simulations.append(sims)
    simulations = pd.concat(simulations)
    filename = 'figures/time_varying/periods_geographic_' + dep_var
    x = diss.plot_period_simulations(simulations, periods, filename, id_vars = 'Lag', var_name = 'Lustrum')

## ------------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------------- ##
## loop over specifications, spatial lag with trade weights


for dep_var in dep_vars:
    data = diss.format_data(dep_var = dep_var, lag = 'spatial_lags_' + dep_var + '_trade.txt')
    simulations = []
    for lag_var in lag_vars:
        dat = data[[dep_var] + [lag_var] + controls]
        dat = dat[dat[period] < 2010]

        period_dummies = pd.get_dummies(dat[period], prefix = period)
        dat = pd.merge(dat, period_dummies, left_index = True, right_index = True, how = 'left')
        dat = dat.drop([period], axis = 1)
        dat = dat.dropna(axis = 0)

        periods = [x for x in dat.columns.tolist() if period in x]
        for col in periods:
            dat['lag_' + col] = dat[lag_var].multiply(dat[col], axis = 'index')
        lag_periods = [x for x in dat.columns.tolist() if 'lag_' + period in x]
        
        dd = diss.brada(dat, dep_var = dep_var, constant = False)
        x0, x1 = np.array(dd.X[lag_var].describe().ix[['25%','75%']]).flatten()
        m = sm.GLM(dd.y, dd.X, family = sm.families.Binomial()).fit()
        est = m.params
        vcov = m.cov_params()
        betas = np.random.multivariate_normal(est, vcov, 1000)
        sims = diss.period_simulator(dd.X, dd.y, betas, periods, lag_periods, lag_var, [x0, x1])
        sims['Lag'] = lag_var.split('_')[1]
        simulations.append(sims)
    simulations = pd.concat(simulations)
    filename = 'figures/time_varying/periods_trade_' + dep_var
    x = diss.plot_period_simulations(simulations, periods, filename, id_vars = 'Lag', var_name = 'Lustrum')
    
## ------------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------------- ##
## loop over specifications, spatial lag with linguistic weights


for dep_var in dep_vars:
    data = diss.format_data(dep_var = dep_var, lag = 'spatial_lags_' + dep_var + '_linguistic.txt')
    simulations = []
    for lag_var in lag_vars:
        dat = data[[dep_var] + [lag_var] + controls]
        dat = dat[dat[period] < 2010]

        period_dummies = pd.get_dummies(dat[period], prefix = period)
        dat = pd.merge(dat, period_dummies, left_index = True, right_index = True, how = 'left')
        dat = dat.drop([period], axis = 1)
        dat = dat.dropna(axis = 0)

        periods = [x for x in dat.columns.tolist() if period in x]
        for col in periods:
            dat['lag_' + col] = dat[lag_var].multiply(dat[col], axis = 'index')
        lag_periods = [x for x in dat.columns.tolist() if 'lag_' + period in x]
        
        dd = diss.brada(dat, dep_var = dep_var, constant = False)
        x0, x1 = np.array(dd.X[lag_var].describe().ix[['25%','75%']]).flatten()
        m = sm.GLM(dd.y, dd.X, family = sm.families.Binomial()).fit()
        est = m.params
        vcov = m.cov_params()
        betas = np.random.multivariate_normal(est, vcov, 1000)
        sims = diss.period_simulator(dd.X, dd.y, betas, periods, lag_periods, lag_var, [x0, x1])
        sims['Lag'] = lag_var.split('_')[1]
        simulations.append(sims)
    simulations = pd.concat(simulations)
    filename = 'figures/time_varying/periods_linguistic_' + dep_var
    x = diss.plot_period_simulations(simulations, periods, filename, id_vars = 'Lag', var_name = 'Lustrum')

## ------------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------------- ##
