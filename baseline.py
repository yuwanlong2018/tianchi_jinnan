# coding=utf-8
# Author:哟嚯走大运了
# Date:2019-01-02
# Email: yuwanlong2018@163.com

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def get_Data():
    os.chdir('/home/webedit/user/yuwanlong/cm/jinan')
    train = pd.read_csv('jinnan_round1_train_20181227.csv', encoding = 'gb18030')
    test = pd.read_csv('jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
    sub = pd.read_csv('jinnan_round1_submit_20181227.csv', encoding = 'gb18030')

    return train, test, sub


def get_TrainValidSplit(train, feature, label, seed=0):
    # 用于训练集和验证集的划分
    X_train, X_valid, y_train, y_valid = train_test_split(train[feature], train[label], test_size=0.33, random_state=seed)
    data = {}
    data['all'] = train[label]
    data['train'] = y_train
    data['valid'] = y_valid
    sns.swarmplot(data=pd.DataFrame(data))

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


if __name__ == '__main__':
    seed = 0
    label = '收率'
    Online = False

    train, test, sub = get_Data()
    CategoricalFeature = get_CategoricalFeature(train)
    for i in CategoricalFeature:
        get_labelEncoder(train, test, i)

    feature = [i for i in train.columns if i not in ['样本id', '收率']]
    X_train, X_valid, y_train, y_valid = train_test_split(train[feature], train[label], test_size=0.33,
                                                          random_state=seed)

    clf_xgboost = xgb.XGBRegressor(learning_rate=0.8, max_depth=4, n_estimators=1000, silent=True,
                                   objective='reg:linear', booster='gbtree', min_child_weight=1.4,  # 叶子节点中分裂时最小的样本权重和
                                   gamma=0,  # 指定了节点分裂所需的最小损失函数下降值
                                   subsample=1,  # 用于每棵树随机采样的比例
                                   colsample_bytree=1,  # 用于每棵树的特征划分比重
                                   colsample_bylevel=1,  # 用于每次分裂的特征划分比重
                                   reg_alpha=0,  # L1正则
                                   reg_lambda=1,  # L2正则
                                   scale_pos_weight=1,  # 正负样本的平衡
                                   random_state=2018)
    clf_xgboost.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='error',
                    early_stopping_rounds=100, verbose=True)

    # train_pred = clf_xgboost.predict(X_train, ntree_limit=clf_xgboost.best_ntree_limit)
    # valid_pred = clf_xgboost.predict(X_valid, ntree_limit=clf_xgboost.best_ntree_limit)

    fm = pd.DataFrame(clf_xgboost.feature_importances_, columns=['score'])
    fm['feature'] = feature
    fm.sort_values(by=['score'], inplace=True, ascending=False)
    print(fm)

    if Online != False:
        test_pred = clf_xgboost.predict(test[feature], ntree_limit=clf_xgboost.best_ntree_limit)
        sub[1] = test_pred
        sub.to_csv('20190102_1.csv', index=None, header=None)














