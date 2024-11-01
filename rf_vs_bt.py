#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from itertools import product
import pickle
import os
cwd = os.getcwd()

SEED = 7531
np.random.seed(SEED)

plt.rcParams.update({
    "text.usetex": True,
    "font.size" : 12,
    #"pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": True,
    "lines.antialiased": True,
    "patch.antialiased": True,
    'axes.linewidth': 0.1
})
%matplotlib inline
%config InlineBackend.figure_format='retina'

from plotting_helpers import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from RF import RandomForestWeight
from data_preprocessor import load_energy, prep_energy

def simple_dm(l1, l2):
    d = l1 - l2
    mod = sm.OLS(d, np.ones(len(d)))
    res = mod.fit().get_robustcov_results(cov_type='HAC',maxlags=1)
    return res

def calc_r2(y_true, y_pred, y_train=None):

    res_mean = ((y_true - y_pred)**2).mean()

    if y_train is None:
        y_true_var = y_true.var()
    else:
        y_train_mean = np.mean(y_train)
        y_true_var = ((y_true - y_train_mean)**2).mean()

    r2 = 1 - (res_mean / y_true_var)
    
    return r2

def quantile_score(y_true, y_pred, alpha):
    diff = y_true - y_pred
    indicator = (diff >= 0).astype(diff.dtype)
    loss = indicator * alpha * diff + (1 - indicator) * (1 - alpha) * (-diff)
    
    return 2*loss

def se(y_true, y_pred):
    return (y_true - y_pred)**2

def ae(y_true, y_pred):
    return np.abs(y_true - y_pred)

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

def crps_sample(y, dat, w, return_mean=True):

    y = y.astype(np.float32)
    dat = dat.astype(np.float32)

    order = np.argsort(dat)
    x = dat[order]

    score_arr = np.zeros((len(y)))

    for i in range(w.shape[0]):
        wi = w[i][order]
        yi = y[i]
        p = np.cumsum(wi)
        P = p[-1]
        a = (p - 0.5 * wi) / P

        # score = 2 / P * np.sum(wi * (np.where(yi < x, 1. , 0.) - a) * (x - yi))
        indicator = (yi < x).astype(x.dtype)
        score = 2 / P * (wi * (indicator - a) * (x - yi)).sum()

        score_arr[i] = score

    if return_mean:
        return score_arr.mean()

    return score_arr
        

#%%
winter_time = ['2018-10-28 02:00:00',
               '2019-10-27 02:00:00',
               '2020-10-25 02:00:00',
               '2021-10-31 02:00:00',
               '2022-10-30 02:00:00',
               '2023-10-29 02:00:00']

df_orig = load_energy('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/Data/rf_data_1823_clean.csv')

# remove samples with 0 load. these arise due to daylight saving
df_orig = df_orig[df_orig.load > 0]
df_orig = df_orig[df_orig.load < 82000]
df_orig = df_orig[~df_orig.date_time.isin(winter_time)]

data_encoding = dict(time_trend=False,
                     time_trend_sq=False,
                     cat_features=False,
                     fine_resolution=False,
                     sin_cos_features=False,
                     last_obs=False)
    
base_fml = ['hour_int', 'weekday_int', 'holiday']

fmls = []
fmls.append(base_fml + ['month_int'])
fmls.append(base_fml + ['yearday'])
fmls.append(base_fml + ['time_trend', 'month_int'])
fmls.append(base_fml + ['time_trend', 'yearday'])
#%%
N_TREES = 1000

crps_arr = []

for fml in fmls:
    print(fml)

    df, _, _ = prep_energy(df=df_orig, **data_encoding)
    
    COMBINED_TEST_PERIOD = True

    if COMBINED_TEST_PERIOD:
        tp_start = "2022-01-01 00:00:00"
        
        dat_train = df.set_index("date_time")[:tp_start]
        dat_test = df.set_index("date_time")[tp_start:]
        
        y_train = dat_train['load'].values
        y_test = dat_test['load'].values
        
        dat_train.drop(columns=['load'], inplace=True)
        dat_test.drop(columns=['load'], inplace=True)
        
        X_train = dat_train[fml]
        X_test = dat_test[fml]
    else:
        tp_start1 = "2022-01-01 00:00:00"
        tp_start2 = "2023-01-01 00:00:00"

        # test_period = 2022
        dat_train = df.set_index("date_time")[:tp_start1]
        dat_test1 = df.set_index("date_time")[tp_start1:tp_start2][:-1]
        dat_test2 = df.set_index("date_time")[tp_start2:]

        y_train = dat_train['load'].values
        y_test = dat_test1['load'].values
        y_test2 = dat_test2['load'].values

        dat_train.drop(columns=['load'], inplace=True)
        dat_test1.drop(columns=['load'], inplace=True)
        dat_test2.drop(columns=['load'], inplace=True)

        X_train = dat_train[fml]
        X_test = dat_test1[fml]
        X_test2 = dat_test2[fml]
    
    MTRY_RF = int(len(fml) / 3)

    # keep this in the loop so each config gets same seed
    hyperparams = dict(n_estimators=N_TREES,
                    random_state=SEED,
                    n_jobs=-1,
                    max_features=MTRY_RF,
                    min_samples_split=2,
                    )

    hyperparams['random_state'] = hyperparams['random_state'] 
    rf = RandomForestWeight(hyperparams=hyperparams)
    rf.fit(X_train, y_train)

    hyperparams_bt = hyperparams.copy()
    hyperparams_bt["max_features"] = len(fml)

    bt = RandomForestWeight(hyperparams=hyperparams_bt)
    bt.fit(X_train, y_train)
    
    _, w_hat_rf = rf.weight_predict(X_test)
    _, w_hat_bt = bt.weight_predict(X_test)
    
    crps_rf = crps_sample(y_test, y_train, w_hat_rf)
    crps_bt = crps_sample(y_test, y_train, w_hat_bt)
    
    crps_arr.append({'time_trend': 'yes' if 'time_trend' in fml else 'no',
                     'day_of_year': 'yes' if 'yearday' in fml else 'no',
                     'CRPS_RF': crps_rf, 'CRPS_BT': crps_bt})
#%%
df_crps = pd.DataFrame(crps_arr)
df_crps
#%%
PLOT = False

if PLOT:
    fig, (ax0, ax1) = plt.subplots(nrows=1, 
                        ncols=2, 
                        figsize=set_size(fraction=1.9,
                                            subplots=(1,2)))

    xticks = [df[df.year == 2018].values[0][0],
            df[df.year == 2019].values[0][0],
            df[df.year == 2020].values[0][0],
            df[df.year == 2021].values[0][0],
            df[df.year == 2022].values[0][0],
            df[df.year == 2023].values[0][0]]

    xlabels = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']


    ax0.plot(df['date'], df['load'], color=blue)
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Energy Demand')
    ax0.set_ylim(29000,86050)
    ax0.set_title("Full Data Set")

    fill_yl = 30000
    fill_yu = 90000

    ax0.axvline(df['date'][0], 
            ls='--', 
            color='black', 
            alpha=.7, 
            zorder=-1)
    ax0.text(df['date'][0], 
            82050, 
            s='Training')
    ax0.axvline(df[(df.date_time >= "2022")&(df.date_time < "2023")]['date'].values[0], 
            ls='--', 
            color='black', 
            alpha=.7,
            zorder=-1)
    ax0.text(df[(df.date_time >= "2022")&(df.date_time < "2023")]['date'].values[0], 
            82050, 
            s='Test 1')
    ax0.axvline(df[df.date_time >= "2023"]['date'].values[0], 
            ls='--', 
            color='black', 
            alpha=.7,
            zorder=-1)
    ax0.text(df[df.date_time >= "2023"]['date'].values[0], 
            82000, 
            s='Test 2')

    df[fml+['load', 'hour_week_int']].groupby(['hour_week_int']).mean()['load'].plot(ax=ax1, color=blue)
    ax1.set_xlabel("Weekly Hour")
    ax1.set_ylabel("Energy Demand")
    ax1.set_title("Average Week")
    ax1.set_ylim(29000,86000)

