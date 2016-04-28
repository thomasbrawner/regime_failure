import requests, zipfile, StringIO
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------- ##
# load and clean geographic distance data, scale between 0 and 1, invert

dist = pd.read_csv('http://privatewww.essex.ac.uk/~ksg/data/capdist.csv')
dist = dist[['numa', 'numb', 'kmdist']]
dist['kmdist_norm'] = dist['kmdist'] / dist['kmdist'].max()  
dist['weight'] = dist['kmdist_norm'] * -1 + np.max(dist['kmdist_norm'])
dist.drop(['kmdist', 'kmdist_norm'], axis=1, inplace=True)
dist.rename(columns = {'numa' : 'cowcode1', 'numb' : 'cowcode2'}, inplace=True)
dist.to_csv('clean_data/geographic_distance.txt', sep=',', index=False)

# -------------------------------------------------------------------------- ##
# load and clean trade distance data, scale between 0 and 1

trade = pd.read_table('clean_data/dyadic_trade_imputed.txt', sep=',')
trade['trade'] = trade['flow1'] + trade['flow2']
trade['trade'] = np.log(trade['trade'] + 1)
trade['weight'] = trade['trade'] / trade['trade'].max()
trade.rename(columns={'ccode1' : 'cowcode1', 'ccode2' : 'cowcode2'}, inplace=True)
trade.dropna(inplace=True)
trade['cowcode1'] = trade['cowcode1'].astype(int)
trade['cowcode2'] = trade['cowcode2'].astype(int)
trade[['cowcode1','cowcode2','weight','year']].to_csv('clean_data/trade_distance.txt', sep=',', index=False)

# -------------------------------------------------------------------------- ##
# load and clean linguistic distance data, fix weight column name

ling = pd.read_table('clean_data/linguistic_distance.txt', sep=',')
if 'connection' in ling.columns.tolist():
	ling.rename(columns={'connection' : 'weight'}, inplace=True)
ling.to_csv('clean_data/linguistic_distance.txt', sep=',', index=False)

# -------------------------------------------------------------------------- ##
# gwf data 

# access the data 
gwf_data_url = 'http://sites.psu.edu/dictators/wp-content/uploads/sites/12570/2015/04/GWF-Autocratic-Regimes-1.2.zip'
r = requests.get(gwf_data_url)
z = zipfile.ZipFile(StringIO.StringIO(r.content))
z.extract('GWF Autocratic Regimes 1.2/GWFtscs.txt');
gwf = pd.read_table('GWF Autocratic Regimes 1.2/GWFtscs.txt')

# subset to relevant columns
gwf = gwf[['cowcode', 'year', 'gwf_fail', 'gwf_enddate', 'gwf_fail_subsregime', 'gwf_fail_type', 
           'gwf_party', 'gwf_personal', 'gwf_military', 'gwf_monarch', 'gwf_duration']]

# apply specific date to observations
gwf['end_date'] = pd.to_datetime(gwf['gwf_enddate'], format='%d%b%Y')
gwf['end_year'] = pd.DatetimeIndex(gwf['end_date']).year
gwf['date_year'] = gwf['end_year'] != gwf['year']
gwf['date'] = np.where(gwf['date_year'] == True, pd.to_datetime('12/31/' + gwf['year'].astype(str)), gwf['end_date'])
gwf['dateInt'] = gwf['date'].apply(lambda x: x.to_datetime().date().toordinal())

# create coercion, democratic transition, and autocratic transition indicators
gwf['coerce'] = (gwf['gwf_fail_type'].isin([4, 5, 6])).astype(int)
gwf['demtrans'] = (gwf['gwf_fail_subsregime'] == 1).astype(int)
gwf['auttrans'] = (gwf['gwf_fail_subsregime'] == 2).astype(int)

# clean up 
gwf.rename(columns = {'gwf_fail' : 'failure',
                      'gwf_party' : 'party',
                      'gwf_personal' : 'personal', 
                      'gwf_monarch' : 'monarch',
                      'gwf_military' : 'military',
                      'gwf_duration' : 'duration'}, inplace=True)
gwf = gwf[['cowcode', 'year', 'date', 'dateInt', 'failure', 'coerce', 'demtrans', 'auttrans', 
           'party', 'personal', 'military', 'monarch', 'duration']]
gwf.to_csv('clean_data/gwf.txt', sep=',', index=False)

# -------------------------------------------------------------------------- ##
# powell & thyne coup data 

def to_date(x):
    try:
        return datetime.strptime(x, '%d%b%Y')
    except:
        return pd.NaT


coups = pd.read_table('http://www.uky.edu/~clthyn2/coup_data/powell_thyne_ccode_year.txt', header=0, sep='\t')
coups.rename(columns={'ccode' : 'cowcode'}, inplace=True)
coups.drop('version', axis=1, inplace=True)

c = coups[['cowcode', 'year'] + [col for col in coups.columns if 'coup' in col]]
c = pd.melt(c, id_vars=['cowcode', 'year']).set_index(['cowcode', 'year'])
c.columns = ['cvar', 'coup']
d = coups[['cowcode', 'year'] + [col for col in coups.columns if 'date' in col]]
d = pd.melt(d, id_vars=['cowcode', 'year']).set_index(['cowcode', 'year'])
d.columns = ['dvar', 'date']

coups = pd.concat([c, d], axis=1)
coups['date_coup'] = coups['date'].apply(lambda x: to_date(x))
coups['success'] = (coups['coup'] > 1).astype(int)
coups['attempt'] = (coups['coup'] > 0).astype(int)
coups.drop(['coup', 'cvar', 'dvar', 'date'], axis=1, inplace=True)
coups.reset_index(inplace=True)

coups = coups.groupby(['cowcode', 'year']).first().reset_index()
coups['date_coup'] = np.where(pd.isnull(coups['date_coup']), pd.to_datetime('12/31/' + coups['year'].astype(str)), coups['date_coup'])
coups['dateInt_coup'] = coups['date_coup'].apply(lambda x: x.to_datetime().date().toordinal())
coups.to_csv('clean_data/coups.txt', sep=',', index=False)
