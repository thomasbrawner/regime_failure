## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## dissertation_analysis_ch1a.py
## produce regression tables and first differences plots for chapter 1
## tb 24 jan 2015, last update 6 feb 2015
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##

import dissertation_programs as diss
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys

## --------------------------------------------------------------------- ##
## dependent variables, spatial lags, and control variables

dep_vars = ['failure','coerce','auttrans','demtrans']
lag_vars = ['lag_failure','lag_coerce','lag_auttrans','lag_demtrans',['lag_auttrans','lag_demtrans']]

controls0 = ['duration','two_year']
controls1 = ['duration','party','personal','military','two_year']
controls2 = ['duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','two_year']
controls3 = ['duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','two_year','region']
controls4 = ['duration','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','two_year','cowcode']

controls = [controls0, controls1, controls2, controls3, controls4]

## --------------------------------------------------------------------- ##
## loop over specifications, spatial lag with geographic weights

for dv in dep_vars:
    data = diss.format_data(dep_var = dv, lag = 'spatial_lags_' + dv + '.txt')
    for lag in lag_vars:
        if type(lag) == list:
            specs = [[dv] + lag + c for c in controls]
        else:
            specs = [[dv] + [lag] + c for c in controls]
        models = []
        for spec in specs:
            factor_list = [x for x in ['two_year','region','cowcode'] if x in spec]
            dat = diss.brada(data[spec], dep_var = dv, factors = factor_list, constant = False)
            models.append(dat.estimate())
            if specs.index(spec) == 2:
                if type(lag) == list:
                    for l in lag:
                        x0, x1 = dat.X[l].describe().ix[['25%','75%']]
                        diss.plot_fd(dat.simulate(set_var = lag, set_values = [x0, x1], seed = 1234), 'figures/geographic_' + dv + '_' + l + '_aut_dem')
                else:
                    x0, x1 = dat.X[lag].describe().ix[['25%','75%']]
                    diss.plot_fd(dat.simulate(set_var = lag, set_values = [x0, x1], seed = 1234), 'figures/geographic_' + dv + '_' + lag)
        tables = [diss.reg_table(x) for x in models]
        summaries = [diss.reg_summary(x) for x in models]
        if type(lag) == list:
            diss.tex_reg_table(tables, summaries, factors = ['two_year','region','cowcode'], file_name = 'tables/geographic_' + dv + '_' + '_'.join(lag))
        else:
            diss.tex_reg_table(tables, summaries, factors = ['two_year','region','cowcode'], file_name = 'tables/geographic_' + dv + '_' + lag)

## --------------------------------------------------------------------- ##
## loop over specifications, spatial lag with trade weights

for dv in dep_vars:
    data = diss.format_data(dep_var = dv, lag = 'spatial_lags_' + dv + '_trade.txt')
    for lag in lag_vars:
        if type(lag) == list:
            specs = [[dv] + lag + c for c in controls]
        else:
            specs = [[dv] + [lag] + c for c in controls]
        models = []
        for spec in specs:
            factor_list = [x for x in ['two_year','region','cowcode'] if x in spec]
            dat = diss.brada(data[spec], dep_var = dv, factors = factor_list, constant = False)
            models.append(dat.estimate())
            if specs.index(spec) == 2:
                if type(lag) == list:
                    for l in lag:
                        x0, x1 = dat.X[l].describe().ix[['25%','75%']]
                        diss.plot_fd(dat.simulate(set_var = lag, set_values = [x0, x1], seed = 1234), 'figures/trade_' + dv + '_' + l + '_aut_dem')
                else:
                    x0, x1 = dat.X[lag].describe().ix[['25%','75%']]
                    diss.plot_fd(dat.simulate(set_var = lag, set_values = [x0, x1], seed = 1234), 'figures/trade_' + dv + '_' + lag)
        tables = [diss.reg_table(x) for x in models]
        summaries = [diss.reg_summary(x) for x in models]
        if type(lag) == list:
            diss.tex_reg_table(tables, summaries, factors = ['two_year','region','cowcode'], file_name = 'tables/trade_' + dv + '_' + '_'.join(lag))
        else:
            diss.tex_reg_table(tables, summaries, factors = ['two_year','region','cowcode'], file_name = 'tables/trade_' + dv + '_' + lag)

## --------------------------------------------------------------------- ##
## loop over specifications, spatial lag with linguistic weights

for dv in dep_vars:
    data = diss.format_data(dep_var = dv, lag = 'spatial_lags_' + dv + '_linguistic.txt')
    for lag in lag_vars:
        if type(lag) == list:
            specs = [[dv] + lag + c for c in controls]
        else:
            specs = [[dv] + [lag] + c for c in controls]
        models = []
        for spec in specs:
            factor_list = [x for x in ['two_year','region','cowcode'] if x in spec]
            dat = diss.brada(data[spec], dep_var = dv, factors = factor_list, constant = False)
            models.append(dat.estimate())
            if specs.index(spec) == 2:
                if type(lag) == list:
                    for l in lag:
                        x0, x1 = dat.X[l].describe().ix[['25%','75%']]
                        diss.plot_fd(dat.simulate(set_var = lag, set_values = [x0, x1], seed = 1234), 'figures/linguistic_' + dv + '_' + l + '_aut_dem')
                else:
                    x0, x1 = dat.X[lag].describe().ix[['25%','75%']]
                    diss.plot_fd(dat.simulate(set_var = lag, set_values = [x0, x1], seed = 1234), 'figures/linguistic_' + dv + '_' + lag)
        tables = [diss.reg_table(x) for x in models]
        summaries = [diss.reg_summary(x) for x in models]
        if type(lag) == list:
            diss.tex_reg_table(tables, summaries, factors = ['two_year','region','cowcode'], file_name = 'tables/linguistic_' + dv + '_' + '_'.join(lag))
        else:
            diss.tex_reg_table(tables, summaries, factors = ['two_year','region','cowcode'], file_name = 'tables/linguistic_' + dv + '_' + lag)

## --------------------------------------------------------------------- ##
## fix variable names in tex tables and insert dummy specification

variables = {r'party' : r'Party',
             r'personal' : r'Personal',
             r'military' : r'Military',
             r'duration_party' : r'Duration $\\times$ Party',
             r'duration_military' : r'Duration $\\times$ Military',
             r'duration_personal' : r'Duration $\\times$ Personal',
             r'duration' : r'Duration',
             r'gdppc' : r'GDP per capita',
             r'growth' : r'Growth',
             r'resource' : r'Resources',
             r'population' : r'Population',
             r'institutions' : r'Institutions',
             r'lag_failure' : r'Lag Failures',
             r'lag_coerce' : r'Lag Coercions',
             r'lag_auttrans' : r'Lag Autocratic Transitions',
             r'lag_demtrans' : r'Lag Democratic Transitions'}

tables = glob.glob('tables/*.tex')

dummy = 'Two-year period dummies & $\\checkmark$ & & $\\checkmark$ & & $\\checkmark$ & & $\\checkmark$ & & $\\checkmark$ & \\\\ \nRegion dummies & & & & & & & $\\checkmark$ & & & \\\\ \nCountry dummies & & & & & & & & & $\\checkmark$ & \\\\ \n\\hline\n'

for table in tables:
    with open(table, 'r+') as tab:
        f = tab.read()
        marker = f.find('$N$')
        f = f[:marker] + dummy + f[marker:]
        for key, value in variables.iteritems():
            p = re.compile(key, re.IGNORECASE)
            f = p.sub(value, f)
        with open(table, 'w') as tab:
            tab.write(f)

## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
