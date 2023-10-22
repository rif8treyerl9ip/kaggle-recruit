#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 2016 until April 2017 traindata
# 2017最終週 testdata


# In[3]:


# import os
# os.listdir('../input')
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import glob, re
# pd.set_option('display.max_columns',10000); pd.set_option('display.max_rows', 50); np.set_printoptions(threshold=90000)

import sys, os
sys.path.append('../../src/') #モジュールが入っているディレクトリのパスを指定

# import eda
# import config
# import maprepro as mpre
# import maprepro2 as mpre2


# In[5]:


path='../../input/'
ar = pd.read_csv(f'{path}air_reserve.csv',)
asi = pd.read_csv(f'{path}air_store_info.csv',)
avd = pd.read_csv(f'{path}air_visit_data.csv',)
di = pd.read_csv(f'{path}date_info.csv',)
# a = pd.read_csv(f'{path}hpg_reserve.csv',)
hsi = pd.read_csv(f'{path}hpg_store_info.csv',)
sa = pd.read_csv(f'{path}sample_submission.csv',)
sir = pd.read_csv(f'{path}store_id_relation.csv',)


# In[6]:


print(asi.shape)
print(ar.shape)
print(avd.shape)
print(di.shape)
print(hsi.shape)
print(sa.shape)
print(sir.shape)
print(avd.shape,sa.shape,avd.shape[0]+sa.shape[0])


# In[7]:


avd


# In[8]:


avd.query('air_store_id=="air_db4b38ebe7a7ceff"')[:5]


# In[9]:


# air_visit_data
def air_visit_data_maesyori(df):
    df.index = pd.to_datetime(df['visit_date'])
    # 要はデータがない日をNanにしてからfillnaで来店者0にしたいだけ
    # 集約関数を適用する必要があるのでmean()している
    df = df.groupby('air_store_id').apply(lambda g: g['visitors'].resample('1d').mean()).reset_index()
    df['visit_date'] = df['visit_date'].dt.strftime('%Y-%m-%d')
    df['was_nil'] = df['visitors'].isnull()
    df['visitors'].fillna(0, inplace=True)
    return df
    
avd = air_visit_data_maesyori(avd)


# In[10]:


avd[:5]


# In[11]:


def creating_reserve_visitor_sum(ar):
    # air_reserve.csvで予約から来訪までの差異を取得できるのでこれを取得して予測に使用する
    # reserve_visitorsとtimedelta_Intを使用する
    from datetime import timedelta 
    ar['visit_datetime'] = pd.to_datetime(ar['visit_datetime'])
    ar['reserve_datetime'] = pd.to_datetime(ar['reserve_datetime'])
    ar['timedelta'] = ar['visit_datetime'] - ar['reserve_datetime'] 
    # ar.dtypes
    ar["timedelta_Int"] = (ar["timedelta"] / timedelta(days=1))
    # 必要なのは予約から来訪までの差異だけなので思い切って予約した時間や来訪した時間は落としてしまう
    ar = ar.drop(['reserve_datetime','timedelta'], axis='columns')
    
    # arのvisit_datetimeを日単位にして、groupby['visit_datetime']してreserve_visitorsのsumをとる
    # timedeltaの周期性を保つ方法が思い浮かばなかったので
    # 単に店ごとにreserve_visitorを足す
    ar.query('air_store_id=="air_db4b38ebe7a7ceff"')[:5]
    ar['visit_date'] = ar['visit_datetime'].dt.strftime('%Y-%m-%d')
    ar = ar.groupby(['air_store_id','visit_date'])['reserve_visitors'].sum().reset_index()
    return ar

ar = creating_reserve_visitor_sum(ar)


# In[12]:


ar


# In[13]:


# date_info
date_info = pd.read_csv(f'{path}/date_info.csv')
def date_info_maesyori(df):
    df.rename(columns={'calendar_date': 'visit_date','holiday_flg': 'is_holiday'}, inplace=True)
    df['prev_day_is_holiday'] = df['is_holiday'].shift().fillna(0)
    df['next_day_is_holiday'] = df['is_holiday'].shift(-1).fillna(0)
    return df

date_info = date_info_maesyori(date_info)


# In[14]:


# 店の売り上げは正規標本っていうよりポアソン…?まあ正規標本か…
store_list = pd.unique(avd.air_store_id)
for store in store_list[:3]:
    avd.query('air_store_id=="air_fff68b929994bfbd"').hist()


# In[15]:


def find_outliers(series):
    # True,falseが返る
    # 2.4は恣意的なのでナンセンスかも
    # print(series.mean())
    return (series - series.mean()) / series.std() > 2.4

def cap_values(series):
    outliers = find_outliers(series)
    max_val = series[~outliers].max()
    series[outliers] = max_val
    return series

def is_outlier_and_transformation_to_clipped_value(avd):
    stores = avd.groupby('air_store_id')
    avd['is_outlier'] = stores.apply(lambda g: find_outliers(g['visitors'])).values
    avd['visitors_capped'] = stores.apply(lambda g: cap_values(g['visitors'])).values
    return avd
avd = is_outlier_and_transformation_to_clipped_value(avd)


# In[16]:


def transformation_of_sample_submission(sa):
    # sample_submissionファイルを前処理
    # store_idとvisit_dateに分割
    # テストのフラグを立てる
    # 目的変数をnanに
    # 連番を降る
    sa['air_store_id'] = sa['id'].str.slice(0, 20)
    sa['visit_date'] = sa['id'].str.slice(21)
    sa['is_test'] = True
    sa['visitors'] = np.nan
    sa['test_number'] = range(len(sa))
    return sa
sa = transformation_of_sample_submission(sa)


# In[17]:


sa.dtypes


# In[18]:


'''air_visit+sample_submission+date_info+air_store_info'''
def all_integration():
    '''今まで作成した特徴量を１つのデータフレームに統合'''
    data = pd.concat((avd, sa.drop('id', axis='columns')))
    data = pd.merge(data,ar,on=["air_store_id","visit_date"],how='left')
    # # reserve_visitorsがない=レジにデータがない→予約数は0としてもよいと考えた
    data['reserve_visitors'].fillna(0, inplace=True)
    data['is_test'].fillna(False, inplace=True)
    data = pd.merge(data, date_info, how='left', left_on=['visit_date'], right_on=['visit_date'])
    data = pd.merge(left=data, right=asi, on='air_store_id', how='left')
    # data = pd.merge(left=data, right=sir, on='air_store_id', how='left')
    # data = pd.merge(left=data, right=hsi, on='hpg_store_id', how='left')
    data['is_test'].fillna(False, inplace=True)
    print(data.shape)
    return data

data = all_integration()
data


# In[19]:


data.isnull().sum()


# In[20]:


tmp = data.air_area_name.str.split(' ',expand=True)
tmp = tmp.rename({0:'Prefecture',1:'municipalities'},axis=1)  
tmp = tmp[['Prefecture','municipalities']]
data = pd.concat([data,tmp], axis=1).drop('air_area_name', axis=1)
data


# In[21]:


data.dtypes
print(data.isnull().sum())


# In[22]:


def transformation_of_data(data):
    data['visit_date'] = pd.to_datetime(data['visit_date'])
    data.index = data['visit_date']
    # astypeでtrue,false→1,0に変換できる
    data['is_weekend'] = data['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    data['day_of_month'] = data['visit_date'].dt.day
    data['dow'] = data['visit_date'].dt.dayofweek
    data['visitors_capped_log1p'] = np.log1p(data['visitors_capped'])
    data = pd.get_dummies(data, columns=['air_genre_name'])
    data = pd.get_dummies(data, columns=['Prefecture','municipalities'])
    
    print(data.shape)
    
    # target_encoding かなり有効。精度が0.2くらい上がる
    tmp = data.groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
    data = pd.merge(data, tmp, how='left', on=['air_store_id','dow']) 
    tmp = data.groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
    data = pd.merge(data, tmp, how='left', on=['air_store_id','dow']) 
    tmp = data.groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
    data = pd.merge(data, tmp, how='left', on=['air_store_id','dow']) 
    tmp = data.groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
    data = pd.merge(data, tmp, how='left', on=['air_store_id','dow']) 
    tmp = data.groupby(['air_store_id','dow'], as_index=False)['visitors'].std().rename(columns={'visitors':'std_visitors'})
    data = pd.merge(data, tmp, how='left', on=['air_store_id','dow']) 
    tmp = data.groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
    data = pd.merge(data, tmp, how='left', on=['air_store_id','dow']) 
    
    # 相対的な緯度・経度
    data['difference_max_lat'] = data['latitude'].max() - data['latitude']
    data['difference_max_long'] = data['longitude'].max() - data['longitude']
    data['plus_longitude_latitude'] = data['longitude'] + data['latitude']
    data['Product_longitude_latitude'] = data['longitude'] * data['latitude']

    return data

data = transformation_of_data(data)


# In[23]:


print(data.isnull().sum())


# In[24]:


sa.visit_date.min(),sa.visit_date.max()


# In[25]:


sa['visit_date'] = pd.to_datetime(sa['visit_date'])


# In[26]:


# train = data[(data['is_test'] == False)]
# こっちにすると損失がかなーり減る
train = data[(data['is_test'] == False) & (data['is_outlier'] == False) & (data['was_nil'] == False)].reset_index(drop=True)
test = data[data['is_test']].sort_values('test_number')

drop_list = ['air_store_id', 'is_test', 'test_number', 'visit_date', 'was_nil',
           'is_outlier', 'visitors', 'visitors_capped', 'day_of_week']
# to_drop = ['air_store_id', 'is_test', 'test_number', 'visit_date', 'was_nil',
#            'is_outlier', 'visitors_capped', 'visitors', 'air_area_name',
#            'station_id', 'station_latitude', 'station_longitude', 'station_vincenty',
#            'station_great_circle']

train = train.drop(drop_list, axis='columns')
y = train['visitors_capped_log1p']
train = train.drop('visitors_capped_log1p', axis='columns')
test = test.drop(drop_list, axis='columns')
test = test.drop('visitors_capped_log1p', axis='columns')
test = test.reset_index(drop=True)


# In[27]:


train


# In[28]:


test


# In[29]:


train.shape,y.shape,test.shape


# In[30]:


# # 積集合
# a = set(train.columns.values)
# b = set(test.columns.values)
# s_intersection = a-b
# print(s_intersection)


# In[31]:


train.describe()
y.describe()


# In[32]:


from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(train, y)
# print("Result on validation data: ", clf.evaluate(X_val, y_val))
from sklearn.metrics import mean_squared_error as mse
mse(y,lr.predict(train))


# In[38]:


from sklearn.model_selection import KFold
import lightgbm as lgb

# Now you can use 'lgb' to work with LightGBM


# In[ ]:





# In[39]:


# X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.25)
feature_importances = pd.DataFrame(index=train.columns)
test_preds = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(train, y)):
    print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
    print(train_idx.shape, test_idx.shape)
        
    X_train, X_valid = train.iloc[train_idx], train.iloc[test_idx]
    y_train, y_valid = y[train_idx], y[test_idx]
        
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    lgbm_params = {
            "objective": "regression",
            "learning_rate": 0.05,
            "boosting_type": "gbdt",
            # "min_data_in_leaf":600,
            # "max_bin": 196,
            # "feature_fraction":0.4,
            # "lambda_l1":36, "lambda_l2":80,
            # "max_depth":10,
            # "num_leaves":1000,
            # "metric": "mae",
            'metric': 'rmse',
            'verbose': 1,
            "tree_learner": "voting",
            "n_jobs": 9,
            "seed": 71
        }

    model = lgb.train(
        lgbm_params, lgb_train,
        valid_sets=[lgb_train,lgb_eval],
        valid_names=['train', 'valid'],
        verbose_eval=100,
        num_boost_round=10000,
        early_stopping_rounds=1000,
    )
    test_pred = model.predict(test)
    test_preds.append(test_pred)
    
    y_pred = model.predict(X_valid)
    score = mse(y_valid, y_pred)
    score = np.expm1(mse(y_valid, y_pred))
    print(f"Fold-{fold+1} | OOF Score: {score}")

# val_mean = np.mean(val_scores)
# val_std = np.std(val_scores)

# print('Local RMSLE: {:.5f} (±{:.5f})'.format(val_mean, val_std))


# In[34]:


sub = sa['id'].to_frame()
sub['visitors'] = 0
test_preds = np.expm1(test_preds)
sub['visitors'] = sum(test_preds)/5
# sub['visitors'] = np.expm1(sub['visitors'])


# In[35]:


sub.describe()


# In[36]:


import datetime
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
sub.to_csv(f'../../output/{now}.csv',index=False)
pd.read_csv(f'../../output/{now}.csv')


# In[37]:


now


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
import shap
shap.initjs()


# In[39]:


# test = shap.sample(test, 1000)
explainer = shap.TreeExplainer(
    model=model,
    data=test,
    feature_perturbation='interventional'
)
# Consider using shap.sample(data, 100) to create a smaller background data set.


# In[40]:


train.shape,test.shape


# In[41]:


# test = shap.sample(test, 10000)
test_shap_values = explainer(test,check_additivity=False)


# In[45]:


test_shap_values[0]


# In[46]:


test.head(1)


# In[56]:


shap.plots.waterfall(test_shap_values[0])


# In[48]:


test


# In[58]:


test.columns


# In[57]:


shap.plots.waterfall(test_shap_values[5])


# In[50]:


shap.plots.bar(shap_values=test_shap_values)


# In[51]:


shap.plots.beeswarm(shap_values=test_shap_values)


# In[269]:


sub


# In[81]:


# そして、TreeExplainer を使って、モデルがどのように推論するか解釈したいデータについて SHAP Value を計算しよう。
# この SHAP Value は、入力したのと同じ次元と要素数で得られる。 そして、値が大きいほど推論において影響が大きいと見なすことができる。
tr_x_shap_values = explainer.shap_values(test)
# つまり、行方向に見れば「特定の予測に、それぞれの特徴量がどれくらい寄与したか」と解釈できる。 
# 同様に、列方向に見れば「予測全体で、その特徴量がどれくらい寄与したか」と解釈できる。

# Summary Plot
# このグラフは、デフォルトでは特徴量ごとに SHAP Valueを一軸の散布図として描画する。

shap.summary_plot(shap_values=tr_x_shap_values,
                  features=test,
                  feature_names=test.columns)

