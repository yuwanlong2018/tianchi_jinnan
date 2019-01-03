# coding=utf-8
# Author:哟嚯走大运了
# Date:2019-01-02
# Email: yuwanlong2018@163.com

import os
import time
from uitl import *
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def get_Data():
    os.chdir('/home/webedit/user/yuwanlong/cm/jinan')
    train = pd.read_csv('jinnan_round1_train_20181227.csv', encoding = 'gb18030')
    test = pd.read_csv('jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
    sub = pd.read_csv('jinnan_round1_submit_20181227.csv', encoding = 'gb18030', heade = None)

    return train, test, sub


def get_CategoricalFeature(df):
    # 寻找描述变量
    cat_vars = []
    print ("\n描述变量有:")
    for col in df.columns:
        if df[col].dtype == "object":
            print (col)
            cat_vars.append(col)

    return cat_vars

def get_labelEncoder(train, test, c):
    # 用于字符串的编码
    le = preprocessing.LabelEncoder()
    le.fit(train[c].fillna('0'))
    test[c] = test[c].map(lambda s: '<unknown>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<unknown>')
    train[c] = le.transform(train[c].fillna('0'))
    test[c] = le.transform(test[c].fillna('0'))
    print('finish LabelEncoder feature {}'.format(c))


def get_CF_to_VF(train, test):
    # 构建派生特征
    train, test = merge_median(train, test, ['A5'], 'A6', 'A5_A6_median')
    train, test = merge_mean(train, test, ['A5'], 'A6', 'A5_A6_mean')
    train, test = merge_sum(train, test, ['A5'], 'A6', 'A5_A6_sum')
    train, test = merge_max(train, test, ['A5'], 'A6', 'A5_A6_max')
    train, test = merge_min(train, test, ['A5'], 'A6', 'A5_A6_min')
    train, test = merge_std(train, test, ['A5'], 'A6', 'A5_A6_std')

    train, test = merge_median(train, test, ['A5'], 'A12', 'A5_A12_median')
    train, test = merge_mean(train, test, ['A5'], 'A12', 'A5_A12_mean')
    train, test = merge_sum(train, test, ['A5'], 'A12', 'A5_A12_sum')
    train, test = merge_max(train, test, ['A5'], 'A12', 'A5_A12_max')
    train, test = merge_min(train, test, ['A5'], 'A12', 'A5_A12_min')
    train, test = merge_std(train, test, ['A5'], 'A12', 'A5_A12_std')

    train, test = merge_count(train, test, ['A5'], 'A25', 'A5_A25_count')
    train, test = merge_nunique(train, test, ['A5'], 'A25', 'A5_A25_nunique')

    train, test = merge_count(train, test, ['A5'], 'B4', 'A5_B4_count')
    train, test = merge_nunique(train, test, ['A5'], 'B4', 'A5_B4_nunique')

    train, test = merge_median(train, test, ['A5'], 'B6', 'A5_B6_median')
    train, test = merge_mean(train, test, ['A5'], 'B6', 'A5_B6_mean')
    train, test = merge_sum(train, test, ['A5'], 'B6', 'A5_B6_sum')
    train, test = merge_max(train, test, ['A5'], 'B6', 'A5_B6_max')
    train, test = merge_min(train, test, ['A5'], 'B6', 'A5_B6_min')
    train, test = merge_std(train, test, ['A5'], 'B6', 'A5_B6_std')

    train, test = merge_count(train, test, ['A5'], 'B9', 'A5_B9_count')
    train, test = merge_nunique(train, test, ['A5'], 'B9', 'A5_B9_nunique')

    train, test = merge_count(train, test, ['A5'], 'B10', 'A5_B10_count')
    train, test = merge_nunique(train, test, ['A5'], 'B10', 'A5_B10_nunique')

    train, test = merge_count(train, test, ['A5'], 'B11', 'A5_B11_count')
    train, test = merge_nunique(train, test, ['A5'], 'B11', 'A5_B11_nunique')

    train, test = merge_median(train, test, ['A5'], 'B14', 'A5_B14_median')
    train, test = merge_mean(train, test, ['A5'], 'B14', 'A5_B14_mean')
    train, test = merge_sum(train, test, ['A5'], 'B14', 'A5_B14_sum')
    train, test = merge_max(train, test, ['A5'], 'B14', 'A5_B14_max')
    train, test = merge_min(train, test, ['A5'], 'B14', 'A5_B14_min')
    train, test = merge_std(train, test, ['A5'], 'B14', 'A5_B14_std')

    return train, test


if __name__ == '__main__':
    seed = 0
    label = '收率'
    Online = False

    train, test, sub = get_Data()
    CategoricalFeature = get_CategoricalFeature(train)
    CategoricalFeature.remove('样本id')

    for i in CategoricalFeature:
        get_labelEncoder(train, test, i)

    feature = [i for i in train.columns if
               i not in ['样本id', '收率', 'A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A20', 'A24', 'A25', 'A26', 'A28', 'B4',
                         'B5', 'B7', 'B9', 'B10', 'B11']]
    X_train, X_valid, y_train, y_valid = train_test_split(train[feature], train[label], test_size=0.33, random_state=seed)

    param = {'num_leaves': 30,
             'min_data_in_leaf': 30,
             'objective': 'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
             "min_child_samples": 30,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 11,
             "metric": 'mse',
             "lambda_l1": 0.1,
             "verbosity": -1}

    train_data = lgb.Dataset(data=X_train, label=y_train, categorical_feature=CategoricalFeature)
    valid_data = lgb.Dataset(data=X_valid, label=y_valid, categorical_feature=CategoricalFeature)

    num_round = 10000
    clf = lgb.train(param, train_data, num_round, valid_sets=[train_data, valid_data], verbose_eval=50, early_stopping_rounds=100)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, feature)), columns=['Value', 'Feature'])
    feature_imp.sort_values(by=['score'], inplace=True, ascending=False)
    print(feature_imp)

    if Online != False:
        test_pred = clf.predict(test[feature], num_iteration=clf.best_iteration)
        sub[1] = test_pred
        sub.to_csv('20190103_1.csv', index=None, header=None)


