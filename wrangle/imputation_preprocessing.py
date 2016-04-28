import itertools 
import pandas as pd
import numpy as np 
import space_time_lags as st 


def dummify(df, variable, drop=False): 
    dummies = pd.get_dummies(df[variable], prefix=variable)
    output = df.join(dummies)
    if drop: 
        output.drop(variable, axis=1, inplace=True)
    return output

def generate_lag_combos(gwf=True): 
    if gwf:
        dv = ['failure', 'demtrans']
        lags = ['failure', 'demtrans', 'auttrans', 'coerce']
    else: 
        dv = ['success', 'attempt']
        lags = ['success', 'attempt']
    distance = ['geographic', 'linguistic', 'trade']
    return itertools.product(dv, lags, distance)

def generate_lag(df, dv, lag, distance, gwf=True): 
    distance_data = pd.read_table('clean_data/' + distance + '_distance.txt', sep=',')
    lagger = st.SpaceTimeLagger(window=366, data=df)
    if gwf:
        lagger.evaluate_event_window_gwf(dep_var=dv, lag_var=lag)
    else:
        lagger.evaluate_event_window_coups(dep_var=dv, lag_var=lag)
    lag_name = '_'.join([dv, lag, distance]) 
    return lagger.generate_lags(distance_weights=distance_data, prefix=lag_name)

def data_transformations(data): 
    data['growth'] /= 10. 
    data['resource'] = np.log(data['resource'] + 1)
    data['population'] = np.log(data['population'])
    data['openness'] = np.log(data['openness'])
    data['duration'] = np.log(data['duration'])
    return data 

def generate_data():
    gwf = pd.read_table('clean_data/gwf.txt', sep=',')
    coups = pd.read_table('clean_data/coups.txt', sep=',')
    control = pd.read_table('clean_data/control_variables.txt', sep=',')
    data = pd.merge(gwf, coups, on=['cowcode', 'year'], how='left')
    data = pd.merge(data, control, on=['cowcode', 'year'], how='left')
    data = data_transformations(data) 

    for lag in generate_lag_combos(gwf=True): 
        lag_data = generate_lag(data, lag[0], lag[1], lag[2])
        data = pd.merge(data, lag_data, on=['cowcode', 'year'], how='left')

    for lag in generate_lag_combos(gwf=False):
        lag_data = generate_lag(data, lag[0], lag[1], lag[2])
        data = pd.merge(data, lag_data, on=['cowcode', 'year'], how='left')
    
    data['period'] = data['year'] // 2 * 2
    data['decade'] = data['year'] // 10 * 10
    data = dummify(data, 'decade')
    data = dummify(data, 'period')
    data = dummify(data, 'cowcode') 
    data = dummify(data, 'region')
    return data

if __name__ == '__main__': 
    data = generate_data() 
    data.to_csv('clean_data/full_regime_data_to_impute.txt', index=False)
