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

os.chdir('/home/webedit/user/yuwanlong/cm/jinan')

train = pd.read_csv('jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test = pd.read_csv('jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
sub = pd.read_csv('jinnan_round1_submit_20181227.csv', encoding = 'gb18030')

def fea_categorical_check(df):
    # 寻找描述变量
    cat_vars = []
    print ("\n描述变量有:")
    for col in df.columns:
        if df[col].dtype == "object":
            print (col)
            cat_vars.append(col)

    return cat_vars

def get_labelEncoder(train, test, c):
    le = preprocessing.LabelEncoder()
    le.fit(train[c])
    test[c] = test[c].map(lambda s: '<unknown>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<unknown>')
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])













