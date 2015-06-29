
# coding: utf-8

# I think what should happen is that we should evaluate the lags for a series of historical windows, then we should see how they correlate for different measures of connectivity, then seeing that they have similar associations, we should find the first principal component for each window. Then we have a number of values of the lag that is equal to the number of historical windows being considered. In the end, hopefully this yields a more representative and all-encompassing measure of regional political instability prevailing for the units of observation in the study.

#### Preliminary stuff

# In[1]:

get_ipython().magic(u'matplotlib inline')

import requests, zipfile, StringIO
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA

print 'Python version ' + sys.version
print 'Pandas version ' + pd.__version__
print 'Numpy version ' + np.__version__

def dates_to_integer(x):
    """
    Convert pandas timestamp series to integers.
    """
    date_list = x.tolist()
    date_list = [d.to_datetime() for d in date_list]
    date_list = [d.date().toordinal() for d in date_list]
    return pd.Series(date_list)

def expand_grid(x, y):
    """
    All possible combinations of values from two supplied vectors.
    """
    xg, yg = np.meshgrid(x, y, copy = False)
    xg = xg.flatten()
    yg = yg.flatten() 
    return pd.DataFrame({'x' : xg, 'y' : yg})


#### Spatial weights

# In[2]:

## ---------------------------------------------------------------------------------------------- ##
## load and clean geographic distance data, generate unity-based normalized distance weight

dist = pd.read_csv('http://privatewww.essex.ac.uk/~ksg/data/capdist.csv')
dist = dist[['numa','numb','kmdist']]
dist['kmdist_norm'] = (dist['kmdist'] - np.min(dist['kmdist'])) / (np.max(dist['kmdist']) - np.min(dist['kmdist']))
dist['kmdist_norm'] = dist['kmdist_norm'] * -1 + np.max(dist['kmdist_norm'])
dist.drop('kmdist', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## load and clean trade distance data, generate unity-based normalized distance weight

trade = pd.read_table('dyadic_trade_imputed.txt', sep = ',')
trade['trade'] = trade['flow1'] + trade['flow2']
trade['trade'] = np.log(trade['trade'] + 1)
trade['trade_norm'] = (trade['trade'] - np.min(trade['trade'])) / (np.max(trade['trade']) - np.min(trade['trade']))
trade.drop(['flow1','flow2','trade'], axis = 1, inplace = True)
trade.dropna(inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## load linguistic distance data, binary weights

lang = pd.read_table('linguistic_distance.txt', sep = ',')


#### Load the GWF and manipulate

# In[3]:

gwf_data_url = 'http://sites.psu.edu/dictators/wp-content/uploads/sites/12570/2015/04/GWF-Autocratic-Regimes-1.2.zip'
r = requests.get(gwf_data_url)
z = zipfile.ZipFile(StringIO.StringIO(r.content))
z.extract('GWF Autocratic Regimes 1.2/GWFtscs.txt');
gwf = pd.read_table('GWF Autocratic Regimes 1.2/GWFtscs.txt')

## ---------------------------------------------------------------------------------------------- ##
## subset gwf to relevant columns

gwf = gwf[['cowcode','year','gwf_fail','gwf_enddate','gwf_fail_subsregime','gwf_fail_type']]

## ---------------------------------------------------------------------------------------------- ##
## apply specific date to observations

gwf['end_date'] = pd.to_datetime(gwf['gwf_enddate'], format = '%d%b%Y')
gwf['end_year'] = pd.DatetimeIndex(gwf['end_date']).year
gwf['date_year'] = gwf['end_year'] != gwf['year']
gwf['date'] = np.where(gwf['date_year'] == True, pd.to_datetime('12/31/' + gwf['year'].astype(str)), gwf['end_date'])
gwf.drop(['gwf_enddate','end_date','end_year','date_year'], axis = 1, inplace = True)
gwf['dateInt'] = dates_to_integer(gwf['date'])


#### Create the *coercion*, *democratic transition*, and *autocratic transition* indicators

# In[4]:

gwf['coerce'] = ((gwf['gwf_fail_type'] == 4) | (gwf['gwf_fail_type'] == 5) | (gwf['gwf_fail_type'] == 6)).astype(int)
gwf['demtrans'] = (gwf['gwf_fail_subsregime'] == 1).astype(int)
gwf['auttrans'] = (gwf['gwf_fail_subsregime'] == 2).astype(int)

gwf.rename(columns = {'gwf_fail' : 'failure'}, inplace = True)
gwf.drop(['gwf_fail_type','gwf_fail_subsregime'], axis = 1, inplace = True)
gwf = gwf[['cowcode','year','date','dateInt','failure','coerce','demtrans','auttrans']]
gwf.head()


#### Expand data frame and find all pairs within 3-year window

# In[5]:

days = expand_grid(gwf['dateInt'], gwf['dateInt'])
days.columns = ['date1','date2']
days = days[days['date2'] < days['date1']]
days = days[days['date2'] >= (days['date1'] - 1096)]


#### Expand the other variables

# In[6]:

## ---------------------------------------------------------------------------------------------- ##
## expand cowcodes and merge

days = pd.merge(days, expand_grid(gwf['cowcode'], gwf['cowcode']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'cowcode1', 'y' : 'cowcode2'}, inplace = True)
days = days[days['cowcode1'] != days['cowcode2']]

## ---------------------------------------------------------------------------------------------- ##
## expand failures indicator and merge, drop target state

days = pd.merge(days, expand_grid(gwf['failure'], gwf['failure']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'failure1', 'y' : 'failure'}, inplace = True)
days.drop('failure1', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand coerced failure types and merge, drop target state

days = pd.merge(days, expand_grid(gwf['coerce'], gwf['coerce']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'coerce1', 'y' : 'coerce'}, inplace = True)
days.drop('coerce1', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand years and merge

days = pd.merge(days, expand_grid(gwf['year'], gwf['year']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'year1', 'y' : 'year2'}, inplace = True)
days.sort(['cowcode1','cowcode2','year1','year2'], inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## drop duplicate country-window observations for sending states

duplicate_check_cols = ['cowcode1','cowcode2','year1']
days.drop_duplicates(duplicate_check_cols, take_last = True, inplace = True)


#### Construct the lags for each weight type

# In[7]:

## ---------------------------------------------------------------------------------------------- ##
## construct the spatial lags for each fail variable == sum of weighted failures

## geographic distance

dist2 = pd.merge(days, dist, left_on = ['cowcode1','cowcode2'], right_on = ['numa','numb'], how = 'left')

out_list = []
for var in ['failure','coerce']:
    lags = dist2[['cowcode1','year1','kmdist_norm',var]]
    lags['weighted_lag'] = lags['kmdist_norm'].multiply(lags[var], axis = 'index')
    lags = lags[['cowcode1','year1','weighted_lag']]
    lags = lags.groupby(['cowcode1','year1']).sum()
    lags.columns = ['lag_' + var + '_geog']
    out_list.append(lags)

dist_out = pd.concat(out_list, axis = 1)
dist_out.reset_index(inplace = True)
dist_out.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)

## trade distance

trade2 = pd.merge(days, trade, left_on = ['cowcode1','cowcode2','year1'], right_on = ['ccode1','ccode2','year'], how = 'left')

out_list = []
for var in ['failure','coerce']:
    lags = trade2[['cowcode1','year1','trade_norm',var]]
    lags['weighted_lag'] = lags['trade_norm'].multiply(lags[var], axis = 'index')
    lags = lags[['cowcode1','year1','weighted_lag']]
    lags = lags.groupby(['cowcode1','year1']).sum()
    lags.columns = ['lag_' + var + '_trade']
    out_list.append(lags)

trade_out = pd.concat(out_list, axis = 1)
trade_out.reset_index(inplace = True)
trade_out.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)

## linguistic distance

lang2 = pd.merge(days, lang, on = ['cowcode1','cowcode2'], how = 'left')

out_list = []
for var in ['failure','coerce']:
    lags = lang2[['cowcode1','year1','connection',var]]
    lags['weighted_lag'] = lags['connection'].multiply(lags[var], axis = 'index')
    lags = lags[['cowcode1','year1','weighted_lag']]
    lags = lags.groupby(['cowcode1','year1']).sum()
    lags.columns = ['lag_' + var + '_lang']
    out_list.append(lags)

lang_out = pd.concat(out_list, axis = 1)
lang_out.reset_index(inplace = True)
lang_out.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)


#### Merge the lags

# In[8]:

lags_out = pd.merge(dist_out, trade_out, on = ['cowcode','year'], how = 'left')
lags_out = pd.merge(lags_out, lang_out, on = ['cowcode','year'], how = 'left')


#### Plot the raw distributions of the lags

# In[9]:

## ---------------------------------------------------------------------------------------------- ##
## plot distributions

plt.figure(figsize = (14,8))
 
plt.subplot(231)
plt.hist(lags_out['lag_failure_geog'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (Geographic)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(232)
plt.hist(lags_out['lag_failure_trade'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (Trade)')
plt.ylabel('')
plt.grid(True)

plt.subplot(233)
plt.hist(lags_out['lag_failure_lang'].dropna(), 16, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (Linguistic)')
plt.ylabel('')
plt.grid(True)

plt.subplot(234)
plt.hist(lags_out['lag_coerce_geog'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Coerced Failures (Geographic)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(235)
plt.hist(lags_out['lag_coerce_trade'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Coerced Failures (Trade)')
plt.ylabel('')
plt.grid(True)

plt.subplot(236)
plt.hist(lags_out['lag_coerce_lang'].dropna(), 16, normed = False, facecolor = 'k')
plt.xlabel('Lagged Coerced Failures (Linguistic)')
plt.ylabel('')
plt.grid(True)

plt.show()


#### Plot the logged distributions of the lags

# In[10]:

## ---------------------------------------------------------------------------------------------- ##
## plot distributions

plt.figure(figsize = (14,8))
 
plt.subplot(231)
plt.hist(np.log(lags_out['lag_failure_geog'] + 1).dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('ln Lagged Failures (Geographic)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(232)
plt.hist(np.log(lags_out['lag_failure_trade'] + 1).dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('ln Lagged Failures (Trade)')
plt.ylabel('')
plt.grid(True)

plt.subplot(233)
plt.hist(np.log(lags_out['lag_failure_lang'] + 1).dropna(), 16, normed = False, facecolor = 'k')
plt.xlabel('ln Lagged Failures (Linguistic)')
plt.ylabel('')
plt.grid(True)

plt.subplot(234)
plt.hist(np.log(lags_out['lag_coerce_geog'] + 1).dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('ln Lagged Coerced Failures (Geographic)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(235)
plt.hist(np.log(lags_out['lag_coerce_trade'] + 1).dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('ln Lagged Coerced Failures (Trade)')
plt.ylabel('')
plt.grid(True)

plt.subplot(236)
plt.hist(np.log(lags_out['lag_coerce_lang'] + 1).dropna(), 16, normed = False, facecolor = 'k')
plt.xlabel('ln Lagged Coerced Failures (Linguistic)')
plt.ylabel('')
plt.grid(True)

plt.show()


#### Principal components

# In[11]:

fails = lags_out[['cowcode','year','lag_failure_geog','lag_failure_trade','lag_failure_lang']].dropna().reset_index(drop = True)
coercions = lags_out[['cowcode','year','lag_coerce_geog','lag_coerce_trade','lag_coerce_lang']].dropna().reset_index(drop = True)


##### Failure, variance explained

# In[13]:

pca = PCA()

## ---------------------------------------------------------------------------------------------- ##
## failures, natural logs

x0 = fails[['lag_failure_geog','lag_failure_trade','lag_failure_lang']]
x0['lag_failure_geog'] = np.log(x0['lag_failure_geog'] + 1)
x0['lag_failure_trade'] = np.log(x0['lag_failure_trade'] + 1)
x0['lag_failure_lang'] = np.log(x0['lag_failure_lang'] + 1)

## ---------------------------------------------------------------------------------------------- ##
## fit pca, variance explained, transform

pca.fit(x0)
x1 = pd.DataFrame(pca.fit_transform(x0))
x1.columns = ['lag_failure_pc1','lag_failure_pc2','lag_failure_pc3']
print 'variance explained by component:', pca.explained_variance_ratio_


##### Loadings

# In[14]:

loadings = pd.DataFrame(pca.components_.T)
loadings.columns = ['pc1','pc2','pc3']
loadings['variable'] = pd.Series(['lag_failure_geog','lag_failure_trade','lag_failure_lang'])
loadings = loadings.set_index('variable')
loadings


##### Clean & format components

# In[15]:

fails = pd.merge(fails, x1, left_index = True, right_index = True, how = 'left')
fails = fails.set_index(['cowcode','year'])


##### Coercions, variance explained

# In[16]:

pca = PCA()

## ---------------------------------------------------------------------------------------------- ##
## coercions, natural logs

x0 = coercions[['lag_coerce_geog','lag_coerce_trade','lag_coerce_lang']]
x0['lag_coerce_geog'] = np.log(x0['lag_coerce_geog'] + 1)
x0['lag_coerce_trade'] = np.log(x0['lag_coerce_trade'] + 1)
x0['lag_coerce_lang'] = np.log(x0['lag_coerce_lang'] + 1)

## ---------------------------------------------------------------------------------------------- ##
## fit pca, variance explained, transform

pca.fit(x0)
x1 = pd.DataFrame(pca.fit_transform(x0))
x1.columns = ['lag_coerce_pc1','lag_coerce_pc2','lag_coerce_pc3']
print 'variance explained by component:', pca.explained_variance_ratio_


##### Loadings

# In[17]:

loadings = pd.DataFrame(pca.components_.T)
loadings.columns = ['pc1','pc2','pc3']
loadings['variable'] = pd.Series(['lag_coerce_geog','lag_coerce_trade','lag_coerce_lang'])
loadings = loadings.set_index('variable')
loadings


##### Clean, format, & merge

# In[18]:

coercions = pd.merge(coercions, x1, left_index = True, right_index = True, how = 'left')
coercions = coercions.set_index(['cowcode','year'])

fails = fails[['lag_failure_pc1','lag_failure_pc2','lag_failure_pc3']].reset_index()
coercions = coercions[['lag_coerce_pc1','lag_coerce_pc2','lag_coerce_pc3']].reset_index()

lags_out = pd.merge(lags_out, fails, on = ['cowcode','year'], how = 'left')
lags_out = pd.merge(lags_out, coercions, on = ['cowcode','year'], how = 'left')


##### Distributions of principal components

# In[19]:

## ---------------------------------------------------------------------------------------------- ##
## plot distributions

plt.figure(figsize = (14,8))
 
plt.subplot(231)
plt.hist(lags_out['lag_failure_pc1'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (PC1)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(232)
plt.hist(lags_out['lag_failure_pc2'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (PC2)')
plt.ylabel('')
plt.grid(True)

plt.subplot(233)
plt.hist(lags_out['lag_failure_pc3'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (PC3)')
plt.ylabel('')
plt.grid(True)

plt.subplot(234)
plt.hist(lags_out['lag_coerce_pc1'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Coerced Failures (PC1)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(235)
plt.hist(lags_out['lag_coerce_pc2'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Coerced Failures (PC2)')
plt.ylabel('')
plt.grid(True)

plt.subplot(236)
plt.hist(lags_out['lag_coerce_pc3'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Coerced Failures (PC3)')
plt.ylabel('')
plt.grid(True)


### Democratic transitions

# In[20]:

## ---------------------------------------------------------------------------------------------- ##
## apply specific date to democratic transitions

gwf['date_demtrans'] = np.where(gwf['demtrans'] == 0, pd.to_datetime('12/31/' + gwf['year'].astype(str)), gwf['date'])
gwf['dateInt_demtrans'] = dates_to_integer(gwf['date_demtrans'])


#### Expand data frame and find all pairs within 3-year window

# In[21]:

days = expand_grid(gwf['dateInt_demtrans'], gwf['dateInt'])
days.columns = ['date1','date2']
days = days[days['date2'] < days['date1']]
days = days[days['date2'] >= (days['date1'] - 1096)]


#### Expand the other variables

# In[22]:

## ---------------------------------------------------------------------------------------------- ##
## expand cowcodes and merge

days = pd.merge(days, expand_grid(gwf['cowcode'], gwf['cowcode']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'cowcode1', 'y' : 'cowcode2'}, inplace = True)
days = days[days['cowcode1'] != days['cowcode2']]

## ---------------------------------------------------------------------------------------------- ##
## expand failures indicator and merge, drop target state

days = pd.merge(days, expand_grid(gwf['failure'], gwf['failure']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'failure1', 'y' : 'failure'}, inplace = True)
days.drop('failure1', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand democratic transitions, drop target state

days = pd.merge(days, expand_grid(gwf['demtrans'], gwf['demtrans']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'demtrans1', 'y' : 'demtrans'}, inplace = True)
days.drop('demtrans1', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand authoritarian transitions, drop target state

days = pd.merge(days, expand_grid(gwf['auttrans'], gwf['auttrans']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'auttrans1', 'y' : 'auttrans'}, inplace = True)
days.drop('auttrans1', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand years and merge

days = pd.merge(days, expand_grid(gwf['year'], gwf['year']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'year1', 'y' : 'year2'}, inplace = True)
days.sort(['cowcode1','cowcode2','year1','year2'], inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## drop duplicate country-window observations for sending states

duplicate_check_cols = ['cowcode1','cowcode2','year1']
days.drop_duplicates(duplicate_check_cols, take_last = True, inplace = True)


#### Construct the lags for each weight type

# In[23]:

## ---------------------------------------------------------------------------------------------- ##
## construct the spatial lags for each fail variable == sum of weighted failures

## geographic distance

dist2 = pd.merge(days, dist, left_on = ['cowcode1','cowcode2'], right_on = ['numa','numb'], how = 'left')

out_list = []
for var in ['failure','demtrans','auttrans']:
    lags = dist2[['cowcode1','year1','kmdist_norm',var]]
    lags['weighted_lag'] = lags['kmdist_norm'].multiply(lags[var], axis = 'index')
    lags = lags[['cowcode1','year1','weighted_lag']]
    lags = lags.groupby(['cowcode1','year1']).sum()
    lags.columns = ['lag_demtrans_' + var + '_geog']
    out_list.append(lags)

dist_out = pd.concat(out_list, axis = 1)
dist_out.reset_index(inplace = True)
dist_out.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)

## trade distance

trade2 = pd.merge(days, trade, left_on = ['cowcode1','cowcode2','year1'], right_on = ['ccode1','ccode2','year'], how = 'left')

out_list = []
for var in ['failure','demtrans','auttrans']:
    lags = trade2[['cowcode1','year1','trade_norm',var]]
    lags['weighted_lag'] = lags['trade_norm'].multiply(lags[var], axis = 'index')
    lags = lags[['cowcode1','year1','weighted_lag']]
    lags = lags.groupby(['cowcode1','year1']).sum()
    lags.columns = ['lag_demtrans_' + var + '_trade']
    out_list.append(lags)

trade_out = pd.concat(out_list, axis = 1)
trade_out.reset_index(inplace = True)
trade_out.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)

## linguistic distance

lang2 = pd.merge(days, lang, on = ['cowcode1','cowcode2'], how = 'left')

out_list = []
for var in ['failure','demtrans','auttrans']:
    lags = lang2[['cowcode1','year1','connection',var]]
    lags['weighted_lag'] = lags['connection'].multiply(lags[var], axis = 'index')
    lags = lags[['cowcode1','year1','weighted_lag']]
    lags = lags.groupby(['cowcode1','year1']).sum()
    lags.columns = ['lag_demtrans_' + var + '_lang']
    out_list.append(lags)

lang_out = pd.concat(out_list, axis = 1)
lang_out.reset_index(inplace = True)
lang_out.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)


#### Merge the lags

# In[24]:

lags_out = pd.merge(lags_out, dist_out, on = ['cowcode','year'], how = 'left')
lags_out = pd.merge(lags_out, trade_out, on = ['cowcode','year'], how = 'left')
lags_out = pd.merge(lags_out, lang_out, on = ['cowcode','year'], how = 'left')


#### Plot the raw distributions of the lags

# In[25]:

plt.figure(figsize = (14, 12))
 
plt.subplot(331)
plt.hist(lags_out['lag_demtrans_failure_geog'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (Geographic)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(332)
plt.hist(lags_out['lag_demtrans_failure_trade'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (Trade)')
plt.ylabel('')
plt.grid(True)

plt.subplot(333)
plt.hist(lags_out['lag_demtrans_failure_lang'].dropna(), 16, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (Linguistic)')
plt.ylabel('')
plt.grid(True)

plt.subplot(334)
plt.hist(lags_out['lag_demtrans_demtrans_geog'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Democratic Transitions (Geographic)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(335)
plt.hist(lags_out['lag_demtrans_demtrans_trade'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Democratic Transitions (Trade)')
plt.ylabel('')
plt.grid(True)

plt.subplot(336)
plt.hist(lags_out['lag_demtrans_demtrans_lang'].dropna(), 16, normed = False, facecolor = 'k')
plt.xlabel('Lagged Democratic Transitions (Linguistic)')
plt.ylabel('')
plt.grid(True)

plt.subplot(337)
plt.hist(lags_out['lag_demtrans_auttrans_geog'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Autocratic Transitions (Geographic)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(338)
plt.hist(lags_out['lag_demtrans_auttrans_trade'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Autocratic Transitions (Trade)')
plt.ylabel('')
plt.grid(True)

plt.subplot(339)
plt.hist(lags_out['lag_demtrans_auttrans_lang'].dropna(), 16, normed = False, facecolor = 'k')
plt.xlabel('Lagged Autocratic Transitions (Linguistic)')
plt.ylabel('')
plt.grid(True)


#### Principal components

# In[26]:

fails = lags_out[['cowcode','year','lag_demtrans_failure_geog','lag_demtrans_failure_trade','lag_demtrans_failure_lang']].dropna().reset_index(drop = True)
dtrans = lags_out[['cowcode','year','lag_demtrans_demtrans_geog','lag_demtrans_demtrans_trade','lag_demtrans_demtrans_lang']].dropna().reset_index(drop = True)
atrans = lags_out[['cowcode','year','lag_demtrans_auttrans_geog','lag_demtrans_auttrans_trade','lag_demtrans_auttrans_lang']].dropna().reset_index(drop = True)


##### Lagged failures, variance explained

# In[27]:

pca = PCA()

## ---------------------------------------------------------------------------------------------- ##
## failures, natural logs

x0 = fails[['lag_demtrans_failure_geog','lag_demtrans_failure_trade','lag_demtrans_failure_lang']]
x0['lag_demtrans_failure_geog'] = np.log(x0['lag_demtrans_failure_geog'] + 1)
x0['lag_demtrans_failure_trade'] = np.log(x0['lag_demtrans_failure_trade'] + 1)
x0['lag_demtrans_failure_lang'] = np.log(x0['lag_demtrans_failure_lang'] + 1)

## ---------------------------------------------------------------------------------------------- ##
## fit pca, variance explained, transform

pca.fit(x0)
x1 = pd.DataFrame(pca.fit_transform(x0))
x1.columns = ['lag_demtrans_failure_pc1','lag_demtrans_failure_pc2','lag_demtrans_failure_pc3']
print 'variance explained by component:', pca.explained_variance_ratio_


##### Loadings

# In[28]:

loadings = pd.DataFrame(pca.components_.T)
loadings.columns = ['pc1','pc2','pc3']
loadings['variable'] = pd.Series(['lag_demtrans_failure_geog','lag_demtrans_failure_trade','lag_demtrans_failure_lang'])
loadings = loadings.set_index('variable')
loadings


##### Clean & format components

# In[29]:

fails = pd.merge(fails, x1, left_index = True, right_index = True, how = 'left')
fails = fails.set_index(['cowcode','year'])


##### Lagged democratic transitions, variance explained

# In[30]:

pca = PCA()

## ---------------------------------------------------------------------------------------------- ##
## failures, natural logs

x0 = dtrans[['lag_demtrans_demtrans_geog','lag_demtrans_demtrans_trade','lag_demtrans_demtrans_lang']]
x0['lag_demtrans_demtrans_geog'] = np.log(x0['lag_demtrans_demtrans_geog'] + 1)
x0['lag_demtrans_demtrans_trade'] = np.log(x0['lag_demtrans_demtrans_trade'] + 1)
x0['lag_demtrans_demtrans_lang'] = np.log(x0['lag_demtrans_demtrans_lang'] + 1)

## ---------------------------------------------------------------------------------------------- ##
## fit pca, variance explained, transform

pca.fit(x0)
x1 = pd.DataFrame(pca.fit_transform(x0))
x1.columns = ['lag_demtrans_demtrans_pc1','lag_demtrans_demtrans_pc2','lag_demtrans_demtrans_pc3']
print 'variance explained by component:', pca.explained_variance_ratio_


##### Loadings

# In[31]:

loadings = pd.DataFrame(pca.components_.T)
loadings.columns = ['pc1','pc2','pc3']
loadings['variable'] = pd.Series(['lag_demtrans_demtrans_geog','lag_demtrans_demtrans_trade','lag_demtrans_demtrans_lang'])
loadings = loadings.set_index('variable')
loadings


##### Clean & format components

# In[32]:

dtrans = pd.merge(dtrans, x1, left_index = True, right_index = True, how = 'left')
dtrans = dtrans.set_index(['cowcode','year'])


##### Lagged autocratic transitions, variance explained

# In[33]:

pca = PCA()

## ---------------------------------------------------------------------------------------------- ##
## failures, natural logs

x0 = atrans[['lag_demtrans_auttrans_geog','lag_demtrans_auttrans_trade','lag_demtrans_auttrans_lang']]
x0['lag_demtrans_auttrans_geog'] = np.log(x0['lag_demtrans_auttrans_geog'] + 1)
x0['lag_demtrans_auttrans_trade'] = np.log(x0['lag_demtrans_auttrans_trade'] + 1)
x0['lag_demtrans_auttrans_lang'] = np.log(x0['lag_demtrans_auttrans_lang'] + 1)

## ---------------------------------------------------------------------------------------------- ##
## fit pca, variance explained, transform

pca.fit(x0)
x1 = pd.DataFrame(pca.fit_transform(x0))
x1.columns = ['lag_demtrans_auttrans_pc1','lag_demtrans_auttrans_pc2','lag_demtrans_auttrans_pc3']
print 'variance explained by component:', pca.explained_variance_ratio_


##### Loadings

# In[34]:

loadings = pd.DataFrame(pca.components_.T)
loadings.columns = ['pc1','pc2','pc3']
loadings['variable'] = pd.Series(['lag_demtrans_auttrans_geog','lag_demtrans_auttrans_trade','lag_demtrans_auttrans_lang'])
loadings = loadings.set_index('variable')
loadings


##### Clean, format, & merge

# In[35]:

atrans = pd.merge(atrans, x1, left_index = True, right_index = True, how = 'left')
atrans = atrans.set_index(['cowcode','year'])

fails = fails[['lag_demtrans_failure_pc1','lag_demtrans_failure_pc2','lag_demtrans_failure_pc3']].reset_index()
dtrans = dtrans[['lag_demtrans_demtrans_pc1','lag_demtrans_demtrans_pc2','lag_demtrans_demtrans_pc3']].reset_index()
atrans = atrans[['lag_demtrans_auttrans_pc1','lag_demtrans_auttrans_pc2','lag_demtrans_auttrans_pc3']].reset_index()

lags_out = pd.merge(lags_out, fails, on = ['cowcode','year'], how = 'left')
lags_out = pd.merge(lags_out, dtrans, on = ['cowcode','year'], how = 'left')
lags_out = pd.merge(lags_out, atrans, on = ['cowcode','year'], how = 'left')


##### Plot the distributions of the principal components

# In[36]:

plt.figure(figsize = (14, 12))
 
plt.subplot(331)
plt.hist(lags_out['lag_demtrans_failure_pc1'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (PC1)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(332)
plt.hist(lags_out['lag_demtrans_failure_pc2'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (PC2)')
plt.ylabel('')
plt.grid(True)

plt.subplot(333)
plt.hist(lags_out['lag_demtrans_failure_pc3'].dropna(), 16, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures (PC3)')
plt.ylabel('')
plt.grid(True)

plt.subplot(334)
plt.hist(lags_out['lag_demtrans_demtrans_pc1'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Democratic Transitions (PC1)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(335)
plt.hist(lags_out['lag_demtrans_demtrans_pc2'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Democratic Transitions (PC2)')
plt.ylabel('')
plt.grid(True)

plt.subplot(336)
plt.hist(lags_out['lag_demtrans_demtrans_pc3'].dropna(), 16, normed = False, facecolor = 'k')
plt.xlabel('Lagged Democratic Transitions (PC3)')
plt.ylabel('')
plt.grid(True)

plt.subplot(337)
plt.hist(lags_out['lag_demtrans_auttrans_pc1'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Autocratic Transitions (PC1)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(338)
plt.hist(lags_out['lag_demtrans_auttrans_pc2'].dropna(), 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Autocratic Transitions (PC2)')
plt.ylabel('')
plt.grid(True)

plt.subplot(339)
plt.hist(lags_out['lag_demtrans_auttrans_pc3'].dropna(), 16, normed = False, facecolor = 'k')
plt.xlabel('Lagged Autocratic Transitions (PC3)')
plt.ylabel('')
plt.grid(True)


#### Write the lags to file

# In[37]:

lags_out.to_csv('dissertation_spatial_lags.csv')

