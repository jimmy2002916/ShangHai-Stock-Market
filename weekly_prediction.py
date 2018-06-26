# coding: utf-8

import numpy as np
import pandas as pd
from time import time
import datetime
#import pymysql
from sqlalchemy import create_engine
#import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from collections import OrderedDict
from xgboost import XGBClassifier, XGBRegressor
from WQ_alpha101 import Alphas
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from itertools import product
from datetime import timedelta
import statsmodels.api as sm
import datetime as dt

#%%
def to_series(alpha, start_date=None, name=None):
    index = pd.MultiIndex.from_product([alpha.index, alpha.columns])
    array = np.array(alpha).ravel()
    df_alpha = pd.DataFrame(array, index=index, columns=[name])
    df_alpha.index.levels[0].name = 'date'
    df_alpha = df_alpha.replace([-np.inf, np.inf], 0).fillna(value=0)
    if start_date:
        return df_alpha[name].loc[start_date:]
    return df_alpha[name]

#%%
# 去除極端值(5倍中位數法)
def MAD_winsorize_series(series, n=5):
    median = np.median(series)
    MAD = np.median([abs(x-median) for x in series])
    upper_bound = median + n*MAD
    lower_bound = median - n*MAD
    return np.clip(series, lower_bound, upper_bound)
# 標準化
def standardize_series(series):
    std = series.std()
    mean = series.mean()
    return (series-mean)/std

#%%
def df_sliced_index(df):
    new_index = []
    rows = []
    for ind, row in df.iterrows():
        new_index.append(ind)
        rows.append(row)
    return pd.DataFrame(data=rows, index=pd.MultiIndex.from_tuples(new_index))

#%%
def date_split_for_validation(date_list, num_week_lookback=5):
    val_list=[]
    gen=(i for i in range(len(date_list)) if i>=num_week_lookback)
    for i in gen:
        train = date_list[i-num_week_lookback:i]
        test = date_list[i]
        val_list.append([train, test])  
    return val_list

#%%
# 需要传入单个因子值和总市值
def neutralization(df, mkt_cap = True, sector = True):
    y = df[df.columns[0]]
    if mkt_cap:
        if sector: #行业、市值
            dummy_sector = pd.get_dummies(df['sector_code'])
            x = pd.concat([df['mktcap'], dummy_sector], axis = 1).values
        else: #仅市值
            x = df['mktcap'].values
    elif sector: #仅行业
        x = pd.get_dummies(df['sector_code']).values
    result = sm.OLS(y.astype(float), x.astype(float)).fit()
    return result.resid.reset_index(level=0,drop=True)

#%%
## 數據準備
start_date = '2014-04-01'
# 定義要從 SQL 抓取之資料年分
year_list = ['2014','2015','2016']
data_origin = pd.DataFrame()
stock_list = pd.read_excel('stock_list.xlsx')

#%%
for year in year_list:
    print('Caculating the data of %s '% year)
    data = pd.read_csv('china_stock/full_%s.csv'% year)
    data['code'] = data['code'].apply(lambda s:s[:6])
    data['date'] = data['date'].apply(lambda s:datetime.strptime(s, '%Y-%m-%d'))
    data = data.set_index(['date','code'])
    data.sort_index(inplace=True)
    data_origin = pd.concat([data_origin, data])
    print('Caculation finished!')
    
#%%
#去除上市1個月內之股票樣本
data_origin['at_least_1_month'] = data_origin['today'].apply(lambda s:datetime.strptime(s, '%Y-%m-%d')) >= data_origin['listed_date'].apply(lambda s:datetime.strptime(s, '%Y-%m-%d')) + timedelta(days=20)
data_clean = data_origin[data_origin['at_least_1_month'] & data_origin['not_st']]
at_least_1_month = to_series(data_clean['at_least_1_month'].unstack(), start_date=start_date, name='at_least_1_month')
not_st = to_series(data_clean['not_st'].unstack(), start_date=start_date, name='not_st')

#%%
#中性化用
mkt_cap_series  = to_series(data_clean['mkt_cap'].unstack(), start_date=start_date, name='mktcap')
mkt_cap_series = mkt_cap_series[(at_least_1_month==True) & (not_st==True)]
mkt_cap_series = mkt_cap_series.groupby(level = 'date').apply(lambda x:np.log10(x))
mkt_cap_series = mkt_cap_series.groupby(level = 'date').apply(standardize_series)

sector_code = to_series(data_clean['sector_code'].unstack(), start_date=start_date, name='sector_code')
sector_code = sector_code[(at_least_1_month==True) & (not_st==True)]
#%%
#創建一個物件用來計算特徵
alpha_ob = Alphas(data_clean)

# 特徵列表(可自行修改)

factor_list = ['alpha001',
               'alpha002',
               'alpha003',
               'alpha004',
               'alpha006',
               'alpha008',
               'alpha009',
               'alpha012',
               'alpha013',
               'alpha014',
               'alpha015',
               'alpha016',
               'alpha022',
               'alpha024',
               'alpha026',
               'alpha029',
               'alpha038',
               'alpha040',
               'alpha043',
               'alpha044',
               'alpha045',
               'alpha053',
               'alpha054',
               'alpha055',
               'alpha060',
               'alpha102',
               'alpha103',
               'alpha104',
               'alpha105',
               'alpha106',
               'alpha107',
               'alpha108',
               ]

#%%
#特徵計算
data_alpha_neutralized = pd.DataFrame()
data_alpha = pd.DataFrame()

#%%
#因子計算(無中性)
for alpha in factor_list:
    print('Caculating %s'% alpha)
    start = time()
    factor = to_series(getattr(alpha_ob, alpha)(), start_date=start_date, name=alpha)
    factor = factor[(at_least_1_month==True) & (not_st==True)]
    #中位數法去極值(5倍)
    factor = factor.groupby(level = 'date').apply(MAD_winsorize_series)
    #標準化
    factor = factor.groupby(level = 'date').apply(standardize_series)
    data_alpha = pd.concat([data_alpha, pd.DataFrame(factor)], axis=1)
    print('%s over! Time spent: %.2f s' % (alpha, time()-start))
    
#%%
#因子計算(中性)
for alpha in factor_list:
    print('Caculating %s'% alpha)
    start = time()
    factor = to_series(getattr(alpha_ob, alpha)(), start_date=start_date, name=alpha)
    factor = factor[(at_least_1_month==True) & (not_st==True)]
    #中位數法去極值(5倍)
    factor = factor.groupby(level = 'date').apply(MAD_winsorize_series)
    #標準化
    factor = factor.groupby(level = 'date').apply(standardize_series)
    df = pd.concat([factor, mkt_cap_series, sector_code], axis=1)
    factor_neutralized = df.groupby('date').apply(neutralization)
    factor_neutralized.name = alpha
    data_alpha_neutralized = pd.concat([data_alpha_neutralized, pd.DataFrame(factor_neutralized)], axis=1)
    print('%s over! Time spent: %.2f s' % (alpha, time()-start))
    
#%%
#weekly label
label = to_series(data_clean['close'].unstack().pct_change(5).shift(-5), start_date=start_date, name='return_5')
label = label[(at_least_1_month==True) & (not_st==True)]
data_alpha_neutralized = pd.concat([data_alpha_neutralized, pd.DataFrame(label)], axis=1)
data_alpha = pd.concat([data_alpha, pd.DataFrame(label)], axis=1)
date_list_weekly = data_alpha_neutralized.index.levels[0][data_alpha_neutralized.index.levels[0].weekday==4]
data_alpha_neutralized['weekly'] = [data_alpha_neutralized.index[i][0] in date_list_weekly for i in range(len(data_alpha_neutralized.index))]
data_weekly = data_alpha_neutralized[data_alpha_neutralized['weekly']]
del data_weekly['weekly']
data_alpha_neutralized_clean = df_sliced_index(data_weekly)

data_alpha['weekly'] = [data_alpha.index[i][0] in date_list_weekly for i in range(len(data_alpha.index))]
data_weekly = data_alpha[data_alpha['weekly']]
del data_weekly['weekly']
data_alpha_clean = df_sliced_index(data_weekly)

date_list_weekly[-5:]
close = data_clean['close'].unstack()

#%%
#模型訓練(回歸)

algo_list = {'XGBRegressor':XGBRegressor,
             'LinearRegression':LinearRegression,
             'Lasso':Lasso,
             'Ridge':Ridge,
             'RandomForestRegressor':RandomForestRegressor,
             'DecisionTreeRegressor':DecisionTreeRegressor,
             'SVR':SVR
             }


#%%
def ML_train_validate(algo_name, data, factor_list, max_depth=3, max_features=None, 
                   subsample=1.0, n_estimators=100, C=None, 
                   num_lookback_periods=4, num_backtest_periods=50, daily_log=False,
                   reserve_rate=False, threshold='median', stock_nums_check=(10,30),
                   predicted_label='return_5'):
    
    date_list = data.index.levels[0]
    date_splitted = date_split_for_validation(date_list, num_week_lookback=num_lookback_periods)
    returns_list = []
    date_plot = []
    def sample_reserve(df, p=reserve_rate):
                    upper_mask = df[predicted_label]>np.percentile(df[predicted_label], 50+(100-p)/2) 
                    lower_mask = df[predicted_label]<np.percentile(df[predicted_label], 50-(100-p)/2)
                    return df[upper_mask | lower_mask]    
                
    for train_days, valid_day in date_splitted[-num_backtest_periods:-1]:
        try:
            train_set = data.loc[train_days[0]:train_days[-1]]
            valiation_set = data.loc[valid_day]                
            if reserve_rate:
                train_set.index.levels[0].name = 'date'
                train_set = train_set.groupby(level = 'date').apply(sample_reserve)
            # training data
            y_train = train_set.pop(predicted_label)
            X_train = train_set[factor_list]
            
            # validation data
            y_test = valiation_set.pop(predicted_label)
            X_test = valiation_set[factor_list]
            
            # To CSV File
            df.to_csv("test.csv", sep='\t')

            
             # train
            if algo_name in ['XGBRegressor', 'RandomForestRegressor']:
                regressor = algo_list[algo_name](n_estimators=n_estimators, max_depth=max_depth, 
                                                        subsample=subsample)
                #regressor_origin = algo_list[algo_name](n_estimators=n_estimators, max_depth=max_depth, 
                                                        #subsample=subsample)
                #sfm = SelectFromModel(regressor_origin, threshold=threshold)
                #regressor = make_pipeline(sfm, regressor_origin)
                
            elif algo_name in ['DecisionTreeRegressor']:
                regressor = algo_list[algo_name](max_depth=max_depth, max_features=max_features)
                
            elif algo_name in ['SVR']:
                regressor = algo_list[algo_name](C=C)
                
            else:
                regressor = algo_list[algo_name]()
                
                
            #print('訓練且預測日期: %s' % valid_day)    
            regressor.fit(X_train, y_train)
            Y_pred = regressor.predict(X_train)
            train_accuracy = metrics.accuracy_score(y_train>0, Y_pred>0) * 100

            Y_pred_test = regressor.predict(X_test)
            test_accuracy = metrics.accuracy_score(y_test>0, Y_pred_test>0) * 100
            # 將實際結果按照預測值大小排序 
            Y_pred_test_sorted = Y_pred_test[Y_pred_test.argsort()][::-1]
            y_test_sorted = y_test[Y_pred_test.argsort()][::-1]
            
         
            if daily_log:
                print('The prediction date of training set: %s' % valid_day)
                print('train_accuracy: %.2f' % train_accuracy)
                print('test_accuracy: %.2f' % test_accuracy)
                for stock_num in stock_nums_check:
                    print('The average rate of return of first %d stocks: %.2f' % (stock_num, y_test_sorted[:stock_num].mean()))
                    print('The average rate of return of last %d stocks: %.2f' % (stock_num, y_test_sorted[-stock_num:].mean()))
            date_plot.append(valid_day)
            returns_list.append(y_test_sorted)
            
        except Exception as e:
            print(e)
            
    for stock_num in stock_nums_check:
        
        first_return_list = [returns[:stock_num].mean() for returns in returns_list]
        back_return_list = [returns[-stock_num:].mean() for returns in returns_list] 
        first_acc_list= [(returns[:stock_num]>0).sum()/stock_num for returns in returns_list]
        back_acc_list= [(returns[-stock_num:]>0).sum()/stock_num for returns in returns_list]
        mean_return = np.mean(first_return_list)*100
        std_return = np.std(first_return_list)*100
        rate = np.mean(first_acc_list)*100
        print('The Sum of average return of first %d stocks: %.2f%% Deviation: %.2f%% Winning Rate: %.2f%%' % (stock_num,  mean_return, std_return, rate))
        mean_return = np.mean(back_return_list)*100
        std_return = np.std(back_return_list)*100
        rate = np.mean(back_acc_list)*100
        print('The Sum of average return of last %d stocks: %.2f%% Deviation: %.2f%% Winning Rate: %.2f%%' % (stock_num,  mean_return, std_return, rate))
        #plt.plot(date_plot, first_return_list, label='排名前 %d 支股票均報酬' % stock_num)
        #plt.plot(date_plot, back_return_list, label='排名後 %d 支股票均報酬' % stock_num)
        #plt.legend(fontsize=15)
        #plt.hlines(0, xmin=date_plot[0], xmax=date_plot[-1])
        #plt.show()
    

#%%
#特徵列表
# 特徵列表(可自行修改)

factor_no_neutralized_list = ['alpha001',
                               'alpha002',
                               'alpha003',
                               'alpha004',
                               'alpha006',
                               'alpha008',
                               'alpha009',
                               'alpha012',
                               'alpha013',
                               'alpha014',
                               'alpha015',
                               'alpha016',
                               'alpha022',
                               'alpha024',
                               'alpha026',
                               'alpha029',
                               'alpha038',
                               'alpha040',
                               'alpha043',
                               'alpha044',
                               'alpha045',
                               'alpha053',
                               'alpha054',
                               'alpha055',
                               'alpha060',
                               'alpha102',
                               'alpha103',
                               'alpha104',
                               'alpha105',
                               'alpha106',
                               'alpha107',
                               'alpha108',
                               ]


#%%
# 特徵列表(可自行修改)

factor_neutralized_list = ['alpha001',
                           'alpha002',
                           'alpha003',
                           'alpha004',
                           'alpha006',
                           'alpha008',
                           'alpha009',
                           'alpha012',
                           'alpha013',
                           'alpha014',
                           'alpha015',
                           'alpha016',
                           'alpha022',
                           'alpha024',
                           'alpha026',
                           'alpha029',
                           'alpha038',
                           'alpha040',
                           'alpha043',
                           'alpha044',
                           'alpha045',
                           'alpha053',
                           'alpha054',
                           'alpha055',
                           'alpha060',
                           'alpha102',
                           'alpha103',
                           'alpha104',
                           'alpha105',
                           'alpha106',
                           'alpha107',
                           'alpha108',
                           ]


#%%
#預測單周報酬
"""

max_depth_list = [3, 4]
subsample_list= [1.0, 0.9, 0.8]
num_lookback_periods_list = [4, 6, 8]
threshold_list = ['mean', 'median']

for max_depth, subsample, num_lookback_periods, threshold in product(max_depth_list, subsample_list, 
                                                                     num_lookback_periods_list, threshold_list):
    start = time()
    print('max_depth: %d  subsample: %.2f  num_lookback_periods: %d  threshold: %s' % (max_depth, subsample,
                                                                                       num_lookback_periods, threshold))
    ML_train_validate('XGBRegressor', data_alpha_clean, factor_no_neutralized_list, max_depth=max_depth, max_features=None, 
                   subsample=subsample, n_estimators=100, C=None, num_lookback_periods=num_lookback_periods, 
                   num_backtest_periods=50, reserve_rate = False, threshold=threshold)
    print('over! 耗時 %.2f s' % (time()-start))
    
for max_depth, subsample, num_lookback_periods, threshold in product(max_depth_list, subsample_list, 
                                                                     num_lookback_periods_list, threshold_list):
    start = time()
    print('max_depth: %d  subsample: %.2f  num_lookback_periods: %d  threshold: %s' % (max_depth, subsample,
                                                                                       num_lookback_periods, threshold))
    ML_train_validate('XGBRegressor', data_alpha_neutralized_clean, factor_neutralized_list, 
                   max_depth=max_depth, max_features=None, subsample=subsample, n_estimators=100, 
                   C=None, num_lookback_periods=num_lookback_periods, 
                   num_backtest_periods=50, reserve_rate = False, threshold=threshold)
    print('over! 耗時 %.2f s' % (time()-start))
"""

#%%

"""
ML_train_validate('XGBRegressor', data_alpha_clean, factor_no_neutralized_list, max_depth=3,  
                  subsample=1, n_estimators=100, num_lookback_periods=4, num_backtest_periods=12, 
                  daily_log=True, reserve_rate=False, threshold='mean', stock_nums_check=(10,100))
"""

#%%


ML_train_validate('XGBRegressor', data_alpha_neutralized_clean, factor_neutralized_list, max_depth=3,  
                  subsample=1, n_estimators=100, num_lookback_periods=4, num_backtest_periods=12, 
                  daily_log=True, reserve_rate=False, threshold='mean', stock_nums_check=(10,100))


#%%
"""
ML_train_validate('XGBRegressor', data_alpha_clean, factor_no_neutralized_list, max_depth=3,  
                  subsample=1, n_estimators=100, num_lookback_periods=4, num_backtest_periods=24, 
                  daily_log=True, reserve_rate=False, threshold=0.05)
"""
#%%
#以下程式碼檢驗近一年所有可能之參數組合表現

def ML_train_validate(algo_name, data, factor_list, max_depth=3, max_features=None, 
                   subsample=1.0, n_estimators=100, C=None, 
                   num_lookback_periods=4, num_backtest_periods=50, daily_log=False,
                   reserve_rate=False, threshold='median', stock_nums_check=(10,30),
                   predicted_label='return_5'):
    
    date_list = data.index.levels[0]
    date_splitted = date_split_for_validation(date_list, num_week_lookback=num_lookback_periods)
    returns_list = []
    date_plot = []
    def sample_reserve(df, p=reserve_rate):
                    upper_mask = df[predicted_label]>np.percentile(df[predicted_label], 50+(100-p)/2) 
                    lower_mask = df[predicted_label]<np.percentile(df[predicted_label], 50-(100-p)/2)
                    return df[upper_mask | lower_mask]    
                
    for train_days, valid_day in date_splitted[-num_backtest_periods:-1]:
        try:
            train_set = data.loc[train_days[0]:train_days[-1]]
            valiation_set = data.loc[valid_day]                
            if reserve_rate:
                train_set.index.levels[0].name = 'date'
                train_set = train_set.groupby(level = 'date').apply(sample_reserve)
            # training data
            y_train = train_set[predicted_label].apply[lambda x:1 if x>=0 else -1]
            X_train = train_set[factor_list]
            
            # validation data
            y_test = valiation_set[predicted_label].apply[lambda x:1 if x>=0 else -1]
            X_test = valiation_set[factor_list]
            
             # train
            if algo_name in ['XGBClassifier', 'RandomForestRegressor']:
                regressor_origin = algo_list[algo_name](n_estimators=n_estimators, max_depth=max_depth, 
                                                        subsample=subsample)
                sfm = SelectFromModel(regressor_origin, threshold=threshold)
                regressor = make_pipeline(sfm, regressor_origin)
                
            elif algo_name in ['DecisionTreeRegressor']:
                regressor = algo_list[algo_name](max_depth=max_depth, max_features=max_features)
                
            elif algo_name in ['SVR']:
                regressor = algo_list[algo_name](C=C)
                
            else:
                regressor = algo_list[algo_name]()
                
                
            #print('訓練且預測日期: %s' % valid_day)    
            regressor.fit(X_train, y_train)
            Y_pred = regressor.predict(X_train)
            train_accuracy = metrics.accuracy_score(y_train>0, Y_pred>0) * 100

            Y_pred_test = regressor.predict(X_test)
            test_accuracy = metrics.accuracy_score(y_test>0, Y_pred_test>0) * 100
            # 將實際結果按照預測值大小排序 
            Y_pred_test_sorted = Y_pred_test[Y_pred_test.argsort()][::-1]
            y_test_sorted = y_test[Y_pred_test.argsort()][::-1]
            
         
            if daily_log:
                print('valid_day: %s' % valid_day)
                print('train_accuracy: %.2f' % train_accuracy)
                print('test_accuracy: %.2f' % test_accuracy)
                print('The avg return of first 10 stocks: %.2f' % y_test_sorted[:10].mean())
                print('The avg return of first 30 stocks: %.2f' % y_test_sorted[:30].mean())
                print('The avg return of last 10 stocks: %.2f' % y_test_sorted[-10:].mean())
                print('The avg return of first 30 stocks: %.2f' % y_test_sorted[-30:].mean())
            date_plot.append(valid_day)
            returns_list.append(y_test_sorted)
            
        except Exception as e:
            print(e)
            
    for stock_num in stock_nums_check:
        
        first_return_list = [returns[:stock_num].mean() for returns in returns_list]
        back_return_list = [returns[-stock_num:].mean() for returns in returns_list] 
        first_acc_list= [(returns[:stock_num]>0).sum()/stock_num for returns in returns_list]
        back_acc_list= [(returns[-stock_num:]>0).sum()/stock_num for returns in returns_list]
        mean_return = np.mean(first_return_list)*100
        std_return = np.std(first_return_list)*100
        rate = np.mean(first_acc_list)*100
        print('The sum of avg return of first %d sotcks: %.2f%% Deviation: %.2f%% Winning Rate: %.2f%%' % (stock_num,  mean_return, std_return, rate))
        mean_return = np.mean(back_return_list)*100
        std_return = np.std(back_return_list)*100
        rate = np.mean(back_acc_list)*100
        print('The Sum of avg return of last %d stocks: %.2f%% Deviation: %.2f%% Winning Rate: %.2f%%' % (stock_num,  mean_return, std_return, rate))
        #plt.plot(date_plot, first_return_list, label='排名前 %d 支股票均報酬' % stock_num)
        #plt.plot(date_plot, back_return_list, label='排名後 %d 支股票均報酬' % stock_num)
        #plt.legend(fontsize=15)
        #plt.hlines(0, xmin=date_plot[0], xmax=date_plot[-1])
        #plt.show()
        
        
        
#%%
## 每周五更新預測
# 定義要從 SQL 抓取之資料年分
"""
year_list_predict = ['2017', '2018']
data_predict = pd.DataFrame()

for year in year_list_predict:
    data = pd.read_sql(year, engine('nick_china_rq_stock_daily'), index_col='index')
    data.columns = ['date', 'code']+list(data.columns[2:])
    data['code'] = data['code'].apply(lambda s:s[:6])
    data = data.set_index(['date','code'])
    data.sort_index(inplace=True)
    data_predict = pd.concat([data_predict, data])
    
"""

#%%
year_list_predict = ['2017', '2018']
data_predict = pd.DataFrame()

for year in year_list_predict:
    print('Caculating %s'% year)
    data = pd.read_csv('china_stock/full_%s.csv'% year)
    data['code'] = data['code'].apply(lambda s:s[:6])
    data['date'] = data['date'].apply(lambda s:datetime.strptime(s, '%Y-%m-%d'))
    data = data.set_index(['date','code'])
    data.sort_index(inplace=True)
    data_predict = pd.concat([data_predict, data])
    print('Caculation Finished!')
    
#%%
alpha_ob = Alphas(data_predict)
# 特徵列表(可自行修改)



factor_list = ['alpha001',
               'alpha002',
               'alpha003',
               'alpha004',
               'alpha006',
               'alpha008',
               'alpha009',
               'alpha012',
               'alpha013',
               'alpha014',
               'alpha015',
               'alpha016',
               'alpha022',
               'alpha024',
               'alpha026',
               'alpha029',
               'alpha038',
               'alpha040',
               'alpha043',
               'alpha044',
               'alpha045',
               'alpha053',
               'alpha054',
               'alpha055',
               'alpha060',
               'alpha102',
               'alpha103',
               'alpha104',
               'alpha105',
               'alpha106',
               'alpha107',
               'alpha108',
               ]

#%%
data = pd.DataFrame()
for alpha in factor_list:
    print('Caculating %s'% alpha)
    start = time()
    factor = to_series(getattr(alpha_ob, alpha)(), start_date='2017-11-01')
    #中位數法去極值(5倍)
    factor = factor.groupby(level = 'date').apply(MAD_winsorize_series)
    #標準化
    factor = factor.groupby(level = 'date').apply(standardize_series)
    
    factor.name = alpha
    data = pd.concat([data, pd.DataFrame(factor)], axis=1)
    print('%s over! Time Spent: %.2f s' % (alpha, time()-start))
    

#%%
label = to_series(data_predict['close'].unstack().pct_change(5).shift(-5), start_date='2017-11-01')
label.name = 'return_5'
data = pd.concat([data, pd.DataFrame(label)], axis=1)

date_list_weekly_predict = data.index.levels[0][data.index.levels[0].weekday==4]
data['weekly'] = [data.index[i][0] in date_list_weekly_predict for i in range(len(data.index))]
data_weekly = data[data['weekly']]
del data_weekly['weekly']
data_clean_predict = df_sliced_index(data_weekly)
data_clean_predict.index.levels[0]
#%%
max_depth = 4
subsample = 0.9

#多少周訓練樣本
num_week_lookback = 8
#%%
date_splitted_predict = date_split_for_validation(date_list_weekly_predict, num_week_lookback=num_week_lookback)
[train_days, valid_day] = date_splitted_predict[-1]# 取最後一個元素

#%%
train_set = data_clean_predict.loc[train_days[0]:train_days[-1]]
valiation_set = data_clean_predict.loc[valid_day]
# clean data with no label
train_set = train_set[train_set['return_5'] !=0]
# training data
y_train = train_set.pop('return_5')
X_train = train_set
# validation data
stock_list = data_predict['close'].unstack().loc[valid_day]
stock_able = ~stock_list.isnull()
valiation_set = valiation_set[stock_able]
y_test = valiation_set.pop('return_5')
X_test = valiation_set
#%%
assets = X_test.index # 取得股票列表

#%%
regressor = DecisionTreeRegressor()
sfm = SelectFromModel(estimator = regressor, threshold='median')
anova_reg = make_pipeline(sfm, regressor)
anova_reg.fit(X_train, y_train)
Y_pred_test = anova_reg.predict(X_test)
assets_sort = assets[Y_pred_test.argsort()][::-1]
fisrt_ten_stocks = assets_sort[:10]
fisrt_thirty_stocks = assets_sort[:30]
back_ten_stocks = assets_sort[-10:]
back_thirty_stocks = assets_sort[-30:]
fisrt_ten_stocks_df = pd.DataFrame(fisrt_ten_stocks, index=range(1, len(fisrt_ten_stocks)+1), columns=['code'])
fisrt_thirty_stocks_df = pd.DataFrame(fisrt_thirty_stocks, index=range(1, len(fisrt_thirty_stocks)+1), columns=['code'])
back_ten_stocks_df = pd.DataFrame(back_ten_stocks, index=range(1, len(back_ten_stocks)+1), columns=['code'])
back_thirty_stocks_df = pd.DataFrame(back_thirty_stocks, index=range(1, len(back_thirty_stocks)+1), columns=['code'])

#%%
fisrt_ten_stocks_df.to_excel('Weekly Prediction of first 10 stocks.xlsx')
fisrt_thirty_stocks_df.to_excel('Weekly Prediction of first 30 stocks.xlsx' )
back_ten_stocks_df.to_excel('Weekly Prediction of last 10 stocks.xlsx')
back_thirty_stocks_df.to_excel('Weekly Prediction of last 30 stocks.xlsx')

"""
fisrt_ten_stocks_df.to_excel('預測名單/%s/前10名股票.xlsx' % valid_day.strftime('%Y-%m-%d'))
fisrt_thirty_stocks_df.to_excel('預測名單/%s/前30名股票.xlsx' % valid_day.strftime('%Y-%m-%d'))
back_ten_stocks_df.to_excel('預測名單/%s/後10名股票.xlsx' % valid_day.strftime('%Y-%m-%d'))
back_thirty_stocks_df.to_excel('預測名單/%s/後30名股票.xlsx' % valid_day.strftime('%Y-%m-%d'))
"""
