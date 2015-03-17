## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## dissertation_analysis_ch1c.py
## democratic diffusion
## 16 mar 2015
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##

import dissertation_programs as diss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn import linear_model

## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## read the data and set up the model specifications

data = diss.format_data(dep_var = 'demtrans', lag = 'spatial_lags_demtrans.txt')

controls0 = ['duration','lag_demtrans']
controls1 = ['duration','lag_demtrans','party','personal','military']
controls2 = ['duration','lag_demtrans','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions',]
controls3 = ['duration','lag_demtrans','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','region']
controls4 = ['duration','lag_demtrans','party','personal','military','duration_party','duration_personal','duration_military','gdppc','growth','resource','population','institutions','cowcode']
controls = [controls0, controls1, controls2, controls3, controls4]

specs = [['demtrans'] + c for c in controls]

## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## run the models

models = []
for spec in specs:
	factor_list = [x for x in ['region','cowcode'] if x in spec]
	dat = diss.brada(data[spec], dep_var = 'demtrans', factors = factor_list, constant = False)
	models.append(dat.estimate())
	x0, x1 = dat.X['lag_demtrans'].describe().ix[['25%','75%']]
   	diss.plot_fd(dat.simulate(set_var = 'lag_demtrans', set_values = [x0, x1], seed = 1234), 'diffusion/figures/fd_diffusion' + str(specs.index(spec)))

tables = [diss.reg_table(x) for x in models]
summaries = [diss.reg_summary(x) for x in models]

diss.tex_reg_table(tables, summaries, factors = ['region','cowcode'], file_name = 'diffusion/diffusion_table')
	
## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## fix variable names in table and insert dummy specification

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
             r'lag_demtrans' : r'Lag Democratic Transitions'}

dummy = 'Region dummies & & & & & & & $\\checkmark$ & & & \\\\ \nCountry dummies & & & & & & & & & $\\checkmark$ & \\\\ \n\\hline\n'

with open('diffusion/diffusion_table.tex', 'r+') as tab:
	f = tab.read()
	marker = f.find('$N$')
	f = f[:marker] + dummy + f[marker:]
	for key, value in variables.iteritems():
		p = re.compile(key, re.IGNORECASE)
		f = p.sub(value, f)
        with open('diffusion/diffusion_table.tex', 'w') as tab:
            tab.write(f)

## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## get predicted probabilities for the specification with region dummies

spec = ['demtrans'] + controls3
dd = data[spec + ['year']]
dd['lustrum'] = diss.round_down_any(dd['year'], base = 5)
dd = dd.dropna(axis = 0)
dd.reset_index(drop = True, inplace = True)

dat = diss.brada(data[spec], dep_var = 'demtrans', factors = ['region'], constant = False)
logit = linear_model.LogisticRegression().fit(dat.X, dat.y)

probs = logit.predict_proba(dat.X)[:,1:].flatten()
dd['probs'] = probs

## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
## get first differences for the same specification

lag_values = dd['lag_demtrans'].describe().ix[['25%','75%']]

dat.X['lag_demtrans'] = lag_values[0]
dd['low'] = logit.predict_proba(dat.X)[:,1:].flatten()

dat.X['lag_demtrans'] = lag_values[1]
dd['high'] = logit.predict_proba(dat.X)[:,1:].flatten()

dd['fd'] = dd['low'].subtract(dd['high'], axis = 'index')

dd[['lustrum','year','demtrans','probs','low','high','fd']].to_csv('diffusion/year_probabilities.csv')

## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
