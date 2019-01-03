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
    #------------------------- A5 -----------------------#
    train, test = merge_median(train, test, ['A5'], 'A6', 'A5_A6_median')
    train, test = merge_mean(train, test, ['A5'], 'A6', 'A5_A6_mean')
    train, test = merge_sum(train, test, ['A5'], 'A6', 'A5_A6_sum')
    train, test = merge_max(train, test, ['A5'], 'A6', 'A5_A6_max')
    train, test = merge_min(train, test, ['A5'], 'A6', 'A5_A6_min')
    train, test = merge_std(train, test, ['A5'], 'A6', 'A5_A6_std')

    train, test = merge_median(train, test, ['A5'], 'A10', 'A5_A10_median')
    train, test = merge_mean(train, test, ['A5'], 'A10', 'A5_A10_mean')
    train, test = merge_sum(train, test, ['A5'], 'A10', 'A5_A10_sum')
    train, test = merge_max(train, test, ['A5'], 'A10', 'A5_A10_max')
    train, test = merge_min(train, test, ['A5'], 'A10', 'A5_A10_min')
    train, test = merge_std(train, test, ['A5'], 'A10', 'A5_A10_std')

    train, test = merge_median(train, test, ['A5'], 'A12', 'A5_A12_median')
    train, test = merge_mean(train, test, ['A5'], 'A12', 'A5_A12_mean')
    train, test = merge_sum(train, test, ['A5'], 'A12', 'A5_A12_sum')
    train, test = merge_max(train, test, ['A5'], 'A12', 'A5_A12_max')
    train, test = merge_min(train, test, ['A5'], 'A12', 'A5_A12_min')
    train, test = merge_std(train, test, ['A5'], 'A12', 'A5_A12_std')

    train, test = merge_median(train, test, ['A5'], 'A15', 'A5_A15_median')
    train, test = merge_mean(train, test, ['A5'], 'A15', 'A5_A15_mean')
    train, test = merge_sum(train, test, ['A5'], 'A15', 'A5_A15_sum')
    train, test = merge_max(train, test, ['A5'], 'A15', 'A5_A15_max')
    train, test = merge_min(train, test, ['A5'], 'A15', 'A5_A15_min')
    train, test = merge_std(train, test, ['A5'], 'A15', 'A5_A15_std')

    train, test = merge_median(train, test, ['A5'], 'A17', 'A5_A17_median')
    train, test = merge_mean(train, test, ['A5'], 'A17', 'A5_A17_mean')
    train, test = merge_sum(train, test, ['A5'], 'A17', 'A5_A17_sum')
    train, test = merge_max(train, test, ['A5'], 'A17', 'A5_A17_max')
    train, test = merge_min(train, test, ['A5'], 'A17', 'A5_A17_min')
    train, test = merge_std(train, test, ['A5'], 'A17', 'A5_A17_std')

    train, test = merge_median(train, test, ['A5'], 'A21', 'A5_A21_median')
    train, test = merge_mean(train, test, ['A5'], 'A21', 'A5_A21_mean')
    train, test = merge_sum(train, test, ['A5'], 'A21', 'A5_A21_sum')
    train, test = merge_max(train, test, ['A5'], 'A21', 'A5_A21_max')
    train, test = merge_min(train, test, ['A5'], 'A21', 'A5_A21_min')
    train, test = merge_std(train, test, ['A5'], 'A21', 'A5_A21_std')

    train, test = merge_median(train, test, ['A5'], 'A22', 'A5_A22_median')
    train, test = merge_mean(train, test, ['A5'], 'A22', 'A5_A22_mean')
    train, test = merge_sum(train, test, ['A5'], 'A22', 'A5_A22_sum')
    train, test = merge_max(train, test, ['A5'], 'A22', 'A5_A22_max')
    train, test = merge_min(train, test, ['A5'], 'A22', 'A5_A22_min')
    train, test = merge_std(train, test, ['A5'], 'A22', 'A5_A22_std')

    train, test = merge_count(train, test, ['A5'], 'A25', 'A5_A25_count')
    train, test = merge_nunique(train, test, ['A5'], 'A25', 'A5_A25_nunique')

    train, test = merge_median(train, test, ['A5'], 'A27', 'A5_A27_median')
    train, test = merge_mean(train, test, ['A5'], 'A27', 'A5_A27_mean')
    train, test = merge_sum(train, test, ['A5'], 'A27', 'A5_A27_sum')
    train, test = merge_max(train, test, ['A5'], 'A27', 'A5_A27_max')
    train, test = merge_min(train, test, ['A5'], 'A27', 'A5_A27_min')
    train, test = merge_std(train, test, ['A5'], 'A27', 'A5_A27_std')

    train, test = merge_median(train, test, ['A5'], 'B1', 'A5_B1_median')
    train, test = merge_mean(train, test, ['A5'], 'B1', 'A5_B1_mean')
    train, test = merge_sum(train, test, ['A5'], 'B1', 'A5_B1_sum')
    train, test = merge_max(train, test, ['A5'], 'B1', 'A5_B1_max')
    train, test = merge_min(train, test, ['A5'], 'B1', 'A5_B1_min')
    train, test = merge_std(train, test, ['A5'], 'B1', 'A5_B1_std')

    train, test = merge_count(train, test, ['A5'], 'B4', 'A5_B4_count')
    train, test = merge_nunique(train, test, ['A5'], 'B4', 'A5_B4_nunique')

    train, test = merge_count(train, test, ['A5'], 'B5', 'A5_B5_count')
    train, test = merge_nunique(train, test, ['A5'], 'B5', 'A5_B5_nunique')

    train, test = merge_median(train, test, ['A5'], 'B6', 'A5_B6_median')
    train, test = merge_mean(train, test, ['A5'], 'B6', 'A5_B6_mean')
    train, test = merge_sum(train, test, ['A5'], 'B6', 'A5_B6_sum')
    train, test = merge_max(train, test, ['A5'], 'B6', 'A5_B6_max')
    train, test = merge_min(train, test, ['A5'], 'B6', 'A5_B6_min')
    train, test = merge_std(train, test, ['A5'], 'B6', 'A5_B6_std')

    train, test = merge_count(train, test, ['A5'], 'B7', 'A5_B7_count')
    train, test = merge_nunique(train, test, ['A5'], 'B7', 'A5_B7_nunique')

    train, test = merge_median(train, test, ['A5'], 'B8', 'A5_B8_median')
    train, test = merge_mean(train, test, ['A5'], 'B8', 'A5_B8_mean')
    train, test = merge_sum(train, test, ['A5'], 'B8', 'A5_B8_sum')
    train, test = merge_max(train, test, ['A5'], 'B8', 'A5_B8_max')
    train, test = merge_min(train, test, ['A5'], 'B8', 'A5_B8_min')
    train, test = merge_std(train, test, ['A5'], 'B8', 'A5_B8_std')

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

    # ------------------------- A7 -----------------------#
    train, test = merge_count(train, test, ['A7'], 'A5', 'A7_A5_count')
    train, test = merge_nunique(train, test, ['A7'], 'A5', 'A7_A5_nunique')

    train, test = merge_median(train, test, ['A7'], 'A6', 'A7_A6_median')
    train, test = merge_mean(train, test, ['A7'], 'A6', 'A7_A6_mean')
    train, test = merge_sum(train, test, ['A7'], 'A6', 'A7_A6_sum')
    train, test = merge_max(train, test, ['A7'], 'A6', 'A7_A6_max')
    train, test = merge_min(train, test, ['A7'], 'A6', 'A7_A6_min')
    train, test = merge_std(train, test, ['A7'], 'A6', 'A7_A6_std')

    train, test = merge_median(train, test, ['A7'], 'A10', 'A7_A10_median')
    train, test = merge_mean(train, test, ['A7'], 'A10', 'A7_A10_mean')
    train, test = merge_sum(train, test, ['A7'], 'A10', 'A7_A10_sum')
    train, test = merge_max(train, test, ['A7'], 'A10', 'A7_A10_max')
    train, test = merge_min(train, test, ['A7'], 'A10', 'A7_A10_min')
    train, test = merge_std(train, test, ['A7'], 'A10', 'A7_A10_std')

    train, test = merge_median(train, test, ['A7'], 'A12', 'A7_A12_median')
    train, test = merge_mean(train, test, ['A7'], 'A12', 'A7_A12_mean')
    train, test = merge_sum(train, test, ['A7'], 'A12', 'A7_A12_sum')
    train, test = merge_max(train, test, ['A7'], 'A12', 'A7_A12_max')
    train, test = merge_min(train, test, ['A7'], 'A12', 'A7_A12_min')
    train, test = merge_std(train, test, ['A7'], 'A12', 'A7_A12_std')

    train, test = merge_median(train, test, ['A7'], 'A15', 'A7_A15_median')
    train, test = merge_mean(train, test, ['A7'], 'A15', 'A7_A15_mean')
    train, test = merge_sum(train, test, ['A7'], 'A15', 'A7_A15_sum')
    train, test = merge_max(train, test, ['A7'], 'A15', 'A7_A15_max')
    train, test = merge_min(train, test, ['A7'], 'A15', 'A7_A15_min')
    train, test = merge_std(train, test, ['A7'], 'A15', 'A7_A15_std')

    train, test = merge_median(train, test, ['A7'], 'A17', 'A7_A17_median')
    train, test = merge_mean(train, test, ['A7'], 'A17', 'A7_A17_mean')
    train, test = merge_sum(train, test, ['A7'], 'A17', 'A7_A17_sum')
    train, test = merge_max(train, test, ['A7'], 'A17', 'A7_A17_max')
    train, test = merge_min(train, test, ['A7'], 'A17', 'A7_A17_min')
    train, test = merge_std(train, test, ['A7'], 'A17', 'A7_A17_std')

    train, test = merge_median(train, test, ['A7'], 'A21', 'A7_A21_median')
    train, test = merge_mean(train, test, ['A7'], 'A21', 'A7_A21_mean')
    train, test = merge_sum(train, test, ['A7'], 'A21', 'A7_A21_sum')
    train, test = merge_max(train, test, ['A7'], 'A21', 'A7_A21_max')
    train, test = merge_min(train, test, ['A7'], 'A21', 'A7_A21_min')
    train, test = merge_std(train, test, ['A7'], 'A21', 'A7_A21_std')

    train, test = merge_median(train, test, ['A7'], 'A22', 'A7_A22_median')
    train, test = merge_mean(train, test, ['A7'], 'A22', 'A7_A22_mean')
    train, test = merge_sum(train, test, ['A7'], 'A22', 'A7_A22_sum')
    train, test = merge_max(train, test, ['A7'], 'A22', 'A7_A22_max')
    train, test = merge_min(train, test, ['A7'], 'A22', 'A7_A22_min')
    train, test = merge_std(train, test, ['A7'], 'A22', 'A7_A22_std')

    train, test = merge_count(train, test, ['A7'], 'A25', 'A7_A25_count')
    train, test = merge_nunique(train, test, ['A7'], 'A25', 'A7_A25_nunique')

    train, test = merge_median(train, test, ['A7'], 'A27', 'A7_A27_median')
    train, test = merge_mean(train, test, ['A7'], 'A27', 'A7_A27_mean')
    train, test = merge_sum(train, test, ['A7'], 'A27', 'A7_A27_sum')
    train, test = merge_max(train, test, ['A7'], 'A27', 'A7_A27_max')
    train, test = merge_min(train, test, ['A7'], 'A27', 'A7_A27_min')
    train, test = merge_std(train, test, ['A7'], 'A27', 'A7_A27_std')

    train, test = merge_median(train, test, ['A7'], 'B1', 'A7_B1_median')
    train, test = merge_mean(train, test, ['A7'], 'B1', 'A7_B1_mean')
    train, test = merge_sum(train, test, ['A7'], 'B1', 'A7_B1_sum')
    train, test = merge_max(train, test, ['A7'], 'B1', 'A7_B1_max')
    train, test = merge_min(train, test, ['A7'], 'B1', 'A7_B1_min')
    train, test = merge_std(train, test, ['A7'], 'B1', 'A7_B1_std')

    train, test = merge_count(train, test, ['A7'], 'B4', 'A7_B4_count')
    train, test = merge_nunique(train, test, ['A7'], 'B4', 'A7_B4_nunique')

    train, test = merge_count(train, test, ['A7'], 'B5', 'A7_B5_count')
    train, test = merge_nunique(train, test, ['A7'], 'B5', 'A7_B5_nunique')

    train, test = merge_median(train, test, ['A7'], 'B6', 'A7_B6_median')
    train, test = merge_mean(train, test, ['A7'], 'B6', 'A7_B6_mean')
    train, test = merge_sum(train, test, ['A7'], 'B6', 'A7_B6_sum')
    train, test = merge_max(train, test, ['A7'], 'B6', 'A7_B6_max')
    train, test = merge_min(train, test, ['A7'], 'B6', 'A7_B6_min')
    train, test = merge_std(train, test, ['A7'], 'B6', 'A7_B6_std')

    train, test = merge_count(train, test, ['A7'], 'B7', 'A7_B7_count')
    train, test = merge_nunique(train, test, ['A7'], 'B7', 'A7_B7_nunique')

    train, test = merge_median(train, test, ['A7'], 'B8', 'A7_B8_median')
    train, test = merge_mean(train, test, ['A7'], 'B8', 'A7_B8_mean')
    train, test = merge_sum(train, test, ['A7'], 'B8', 'A7_B8_sum')
    train, test = merge_max(train, test, ['A7'], 'B8', 'A7_B8_max')
    train, test = merge_min(train, test, ['A7'], 'B8', 'A7_B8_min')
    train, test = merge_std(train, test, ['A7'], 'B8', 'A7_B8_std')

    train, test = merge_count(train, test, ['A7'], 'B9', 'A7_B9_count')
    train, test = merge_nunique(train, test, ['A7'], 'B9', 'A7_B9_nunique')

    train, test = merge_count(train, test, ['A7'], 'B10', 'A7_B10_count')
    train, test = merge_nunique(train, test, ['A7'], 'B10', 'A7_B10_nunique')

    train, test = merge_count(train, test, ['A7'], 'B11', 'A7_B11_count')
    train, test = merge_nunique(train, test, ['A7'], 'B11', 'A7_B11_nunique')

    train, test = merge_median(train, test, ['A7'], 'B14', 'A7_B14_median')
    train, test = merge_mean(train, test, ['A7'], 'B14', 'A7_B14_mean')
    train, test = merge_sum(train, test, ['A7'], 'B14', 'A7_B14_sum')
    train, test = merge_max(train, test, ['A7'], 'B14', 'A7_B14_max')
    train, test = merge_min(train, test, ['A7'], 'B14', 'A7_B14_min')
    train, test = merge_std(train, test, ['A7'], 'B14', 'A7_B14_std')

    # ------------------------- A9 -----------------------#
    train, test = merge_count(train, test, ['A9'], 'A5', 'A9_A5_count')
    train, test = merge_nunique(train, test, ['A9'], 'A5', 'A9_A5_nunique')

    train, test = merge_median(train, test, ['A9'], 'A6', 'A9_A6_median')
    train, test = merge_mean(train, test, ['A9'], 'A6', 'A9_A6_mean')
    train, test = merge_sum(train, test, ['A9'], 'A6', 'A9_A6_sum')
    train, test = merge_max(train, test, ['A9'], 'A6', 'A9_A6_max')
    train, test = merge_min(train, test, ['A9'], 'A6', 'A9_A6_min')
    train, test = merge_std(train, test, ['A9'], 'A6', 'A9_A6_std')

    train, test = merge_median(train, test, ['A9'], 'A10', 'A9_A10_median')
    train, test = merge_mean(train, test, ['A9'], 'A10', 'A9_A10_mean')
    train, test = merge_sum(train, test, ['A9'], 'A10', 'A9_A10_sum')
    train, test = merge_max(train, test, ['A9'], 'A10', 'A9_A10_max')
    train, test = merge_min(train, test, ['A9'], 'A10', 'A9_A10_min')
    train, test = merge_std(train, test, ['A9'], 'A10', 'A9_A10_std')

    train, test = merge_median(train, test, ['A9'], 'A12', 'A9_A12_median')
    train, test = merge_mean(train, test, ['A9'], 'A12', 'A9_A12_mean')
    train, test = merge_sum(train, test, ['A9'], 'A12', 'A9_A12_sum')
    train, test = merge_max(train, test, ['A9'], 'A12', 'A9_A12_max')
    train, test = merge_min(train, test, ['A9'], 'A12', 'A9_A12_min')
    train, test = merge_std(train, test, ['A9'], 'A12', 'A9_A12_std')

    train, test = merge_median(train, test, ['A9'], 'A15', 'A9_A15_median')
    train, test = merge_mean(train, test, ['A9'], 'A15', 'A9_A15_mean')
    train, test = merge_sum(train, test, ['A9'], 'A15', 'A9_A15_sum')
    train, test = merge_max(train, test, ['A9'], 'A15', 'A9_A15_max')
    train, test = merge_min(train, test, ['A9'], 'A15', 'A9_A15_min')
    train, test = merge_std(train, test, ['A9'], 'A15', 'A9_A15_std')

    train, test = merge_median(train, test, ['A9'], 'A17', 'A9_A17_median')
    train, test = merge_mean(train, test, ['A9'], 'A17', 'A9_A17_mean')
    train, test = merge_sum(train, test, ['A9'], 'A17', 'A9_A17_sum')
    train, test = merge_max(train, test, ['A9'], 'A17', 'A9_A17_max')
    train, test = merge_min(train, test, ['A9'], 'A17', 'A9_A17_min')
    train, test = merge_std(train, test, ['A9'], 'A17', 'A9_A17_std')

    train, test = merge_median(train, test, ['A9'], 'A21', 'A9_A21_median')
    train, test = merge_mean(train, test, ['A9'], 'A21', 'A9_A21_mean')
    train, test = merge_sum(train, test, ['A9'], 'A21', 'A9_A21_sum')
    train, test = merge_max(train, test, ['A9'], 'A21', 'A9_A21_max')
    train, test = merge_min(train, test, ['A9'], 'A21', 'A9_A21_min')
    train, test = merge_std(train, test, ['A9'], 'A21', 'A9_A21_std')

    train, test = merge_median(train, test, ['A9'], 'A22', 'A9_A22_median')
    train, test = merge_mean(train, test, ['A9'], 'A22', 'A9_A22_mean')
    train, test = merge_sum(train, test, ['A9'], 'A22', 'A9_A22_sum')
    train, test = merge_max(train, test, ['A9'], 'A22', 'A9_A22_max')
    train, test = merge_min(train, test, ['A9'], 'A22', 'A9_A22_min')
    train, test = merge_std(train, test, ['A9'], 'A22', 'A9_A22_std')

    train, test = merge_count(train, test, ['A9'], 'A25', 'A9_A25_count')
    train, test = merge_nunique(train, test, ['A9'], 'A25', 'A9_A25_nunique')

    train, test = merge_median(train, test, ['A9'], 'A27', 'A9_A27_median')
    train, test = merge_mean(train, test, ['A9'], 'A27', 'A9_A27_mean')
    train, test = merge_sum(train, test, ['A9'], 'A27', 'A9_A27_sum')
    train, test = merge_max(train, test, ['A9'], 'A27', 'A9_A27_max')
    train, test = merge_min(train, test, ['A9'], 'A27', 'A9_A27_min')
    train, test = merge_std(train, test, ['A9'], 'A27', 'A9_A27_std')

    train, test = merge_median(train, test, ['A9'], 'B1', 'A9_B1_median')
    train, test = merge_mean(train, test, ['A9'], 'B1', 'A9_B1_mean')
    train, test = merge_sum(train, test, ['A9'], 'B1', 'A9_B1_sum')
    train, test = merge_max(train, test, ['A9'], 'B1', 'A9_B1_max')
    train, test = merge_min(train, test, ['A9'], 'B1', 'A9_B1_min')
    train, test = merge_std(train, test, ['A9'], 'B1', 'A9_B1_std')

    train, test = merge_count(train, test, ['A9'], 'B4', 'A9_B4_count')
    train, test = merge_nunique(train, test, ['A9'], 'B4', 'A9_B4_nunique')

    train, test = merge_count(train, test, ['A9'], 'B5', 'A9_B5_count')
    train, test = merge_nunique(train, test, ['A9'], 'B5', 'A9_B5_nunique')

    train, test = merge_median(train, test, ['A9'], 'B6', 'A9_B6_median')
    train, test = merge_mean(train, test, ['A9'], 'B6', 'A9_B6_mean')
    train, test = merge_sum(train, test, ['A9'], 'B6', 'A9_B6_sum')
    train, test = merge_max(train, test, ['A9'], 'B6', 'A9_B6_max')
    train, test = merge_min(train, test, ['A9'], 'B6', 'A9_B6_min')
    train, test = merge_std(train, test, ['A9'], 'B6', 'A9_B6_std')

    train, test = merge_count(train, test, ['A9'], 'B7', 'A9_B7_count')
    train, test = merge_nunique(train, test, ['A9'], 'B7', 'A9_B7_nunique')

    train, test = merge_median(train, test, ['A9'], 'B8', 'A9_B8_median')
    train, test = merge_mean(train, test, ['A9'], 'B8', 'A9_B8_mean')
    train, test = merge_sum(train, test, ['A9'], 'B8', 'A9_B8_sum')
    train, test = merge_max(train, test, ['A9'], 'B8', 'A9_B8_max')
    train, test = merge_min(train, test, ['A9'], 'B8', 'A9_B8_min')
    train, test = merge_std(train, test, ['A9'], 'B8', 'A9_B8_std')

    train, test = merge_count(train, test, ['A9'], 'B9', 'A9_B9_count')
    train, test = merge_nunique(train, test, ['A9'], 'B9', 'A9_B9_nunique')

    train, test = merge_count(train, test, ['A9'], 'B10', 'A9_B10_count')
    train, test = merge_nunique(train, test, ['A9'], 'B10', 'A9_B10_nunique')

    train, test = merge_count(train, test, ['A9'], 'B11', 'A9_B11_count')
    train, test = merge_nunique(train, test, ['A9'], 'B11', 'A9_B11_nunique')

    train, test = merge_median(train, test, ['A9'], 'B14', 'A9_B14_median')
    train, test = merge_mean(train, test, ['A9'], 'B14', 'A9_B14_mean')
    train, test = merge_sum(train, test, ['A9'], 'B14', 'A9_B14_sum')
    train, test = merge_max(train, test, ['A9'], 'B14', 'A9_B14_max')
    train, test = merge_min(train, test, ['A9'], 'B14', 'A9_B14_min')
    train, test = merge_std(train, test, ['A9'], 'B14', 'A9_B14_std')

    # ------------------------- A11 -----------------------#
    train, test = merge_count(train, test, ['A11'], 'A5', 'A11_A5_count')
    train, test = merge_nunique(train, test, ['A11'], 'A5', 'A11_A5_nunique')

    train, test = merge_median(train, test, ['A11'], 'A6', 'A11_A6_median')
    train, test = merge_mean(train, test, ['A11'], 'A6', 'A11_A6_mean')
    train, test = merge_sum(train, test, ['A11'], 'A6', 'A11_A6_sum')
    train, test = merge_max(train, test, ['A11'], 'A6', 'A11_A6_max')
    train, test = merge_min(train, test, ['A11'], 'A6', 'A11_A6_min')
    train, test = merge_std(train, test, ['A11'], 'A6', 'A11_A6_std')

    train, test = merge_median(train, test, ['A11'], 'A10', 'A11_A10_median')
    train, test = merge_mean(train, test, ['A11'], 'A10', 'A11_A10_mean')
    train, test = merge_sum(train, test, ['A11'], 'A10', 'A11_A10_sum')
    train, test = merge_max(train, test, ['A11'], 'A10', 'A11_A10_max')
    train, test = merge_min(train, test, ['A11'], 'A10', 'A11_A10_min')
    train, test = merge_std(train, test, ['A11'], 'A10', 'A11_A10_std')

    train, test = merge_median(train, test, ['A11'], 'A12', 'A11_A12_median')
    train, test = merge_mean(train, test, ['A11'], 'A12', 'A11_A12_mean')
    train, test = merge_sum(train, test, ['A11'], 'A12', 'A11_A12_sum')
    train, test = merge_max(train, test, ['A11'], 'A12', 'A11_A12_max')
    train, test = merge_min(train, test, ['A11'], 'A12', 'A11_A12_min')
    train, test = merge_std(train, test, ['A11'], 'A12', 'A11_A12_std')

    train, test = merge_median(train, test, ['A11'], 'A15', 'A11_A15_median')
    train, test = merge_mean(train, test, ['A11'], 'A15', 'A11_A15_mean')
    train, test = merge_sum(train, test, ['A11'], 'A15', 'A11_A15_sum')
    train, test = merge_max(train, test, ['A11'], 'A15', 'A11_A15_max')
    train, test = merge_min(train, test, ['A11'], 'A15', 'A11_A15_min')
    train, test = merge_std(train, test, ['A11'], 'A15', 'A11_A15_std')

    train, test = merge_median(train, test, ['A11'], 'A17', 'A11_A17_median')
    train, test = merge_mean(train, test, ['A11'], 'A17', 'A11_A17_mean')
    train, test = merge_sum(train, test, ['A11'], 'A17', 'A11_A17_sum')
    train, test = merge_max(train, test, ['A11'], 'A17', 'A11_A17_max')
    train, test = merge_min(train, test, ['A11'], 'A17', 'A11_A17_min')
    train, test = merge_std(train, test, ['A11'], 'A17', 'A11_A17_std')

    train, test = merge_median(train, test, ['A11'], 'A21', 'A11_A21_median')
    train, test = merge_mean(train, test, ['A11'], 'A21', 'A11_A21_mean')
    train, test = merge_sum(train, test, ['A11'], 'A21', 'A11_A21_sum')
    train, test = merge_max(train, test, ['A11'], 'A21', 'A11_A21_max')
    train, test = merge_min(train, test, ['A11'], 'A21', 'A11_A21_min')
    train, test = merge_std(train, test, ['A11'], 'A21', 'A11_A21_std')

    train, test = merge_median(train, test, ['A11'], 'A22', 'A11_A22_median')
    train, test = merge_mean(train, test, ['A11'], 'A22', 'A11_A22_mean')
    train, test = merge_sum(train, test, ['A11'], 'A22', 'A11_A22_sum')
    train, test = merge_max(train, test, ['A11'], 'A22', 'A11_A22_max')
    train, test = merge_min(train, test, ['A11'], 'A22', 'A11_A22_min')
    train, test = merge_std(train, test, ['A11'], 'A22', 'A11_A22_std')

    train, test = merge_count(train, test, ['A11'], 'A25', 'A11_A25_count')
    train, test = merge_nunique(train, test, ['A11'], 'A25', 'A11_A25_nunique')

    train, test = merge_median(train, test, ['A11'], 'A27', 'A11_A27_median')
    train, test = merge_mean(train, test, ['A11'], 'A27', 'A11_A27_mean')
    train, test = merge_sum(train, test, ['A11'], 'A27', 'A11_A27_sum')
    train, test = merge_max(train, test, ['A11'], 'A27', 'A11_A27_max')
    train, test = merge_min(train, test, ['A11'], 'A27', 'A11_A27_min')
    train, test = merge_std(train, test, ['A11'], 'A27', 'A11_A27_std')

    train, test = merge_median(train, test, ['A11'], 'B1', 'A11_B1_median')
    train, test = merge_mean(train, test, ['A11'], 'B1', 'A11_B1_mean')
    train, test = merge_sum(train, test, ['A11'], 'B1', 'A11_B1_sum')
    train, test = merge_max(train, test, ['A11'], 'B1', 'A11_B1_max')
    train, test = merge_min(train, test, ['A11'], 'B1', 'A11_B1_min')
    train, test = merge_std(train, test, ['A11'], 'B1', 'A11_B1_std')

    train, test = merge_count(train, test, ['A11'], 'B4', 'A11_B4_count')
    train, test = merge_nunique(train, test, ['A11'], 'B4', 'A11_B4_nunique')

    train, test = merge_count(train, test, ['A11'], 'B5', 'A11_B5_count')
    train, test = merge_nunique(train, test, ['A11'], 'B5', 'A11_B5_nunique')

    train, test = merge_median(train, test, ['A11'], 'B6', 'A11_B6_median')
    train, test = merge_mean(train, test, ['A11'], 'B6', 'A11_B6_mean')
    train, test = merge_sum(train, test, ['A11'], 'B6', 'A11_B6_sum')
    train, test = merge_max(train, test, ['A11'], 'B6', 'A11_B6_max')
    train, test = merge_min(train, test, ['A11'], 'B6', 'A11_B6_min')
    train, test = merge_std(train, test, ['A11'], 'B6', 'A11_B6_std')

    train, test = merge_count(train, test, ['A11'], 'B7', 'A11_B7_count')
    train, test = merge_nunique(train, test, ['A11'], 'B7', 'A11_B7_nunique')

    train, test = merge_median(train, test, ['A11'], 'B8', 'A11_B8_median')
    train, test = merge_mean(train, test, ['A11'], 'B8', 'A11_B8_mean')
    train, test = merge_sum(train, test, ['A11'], 'B8', 'A11_B8_sum')
    train, test = merge_max(train, test, ['A11'], 'B8', 'A11_B8_max')
    train, test = merge_min(train, test, ['A11'], 'B8', 'A11_B8_min')
    train, test = merge_std(train, test, ['A11'], 'B8', 'A11_B8_std')

    train, test = merge_count(train, test, ['A11'], 'B9', 'A11_B9_count')
    train, test = merge_nunique(train, test, ['A11'], 'B9', 'A11_B9_nunique')

    train, test = merge_count(train, test, ['A11'], 'B10', 'A11_B10_count')
    train, test = merge_nunique(train, test, ['A11'], 'B10', 'A11_B10_nunique')

    train, test = merge_count(train, test, ['A11'], 'B11', 'A11_B11_count')
    train, test = merge_nunique(train, test, ['A11'], 'B11', 'A11_B11_nunique')

    train, test = merge_median(train, test, ['A11'], 'B14', 'A11_B14_median')
    train, test = merge_mean(train, test, ['A11'], 'B14', 'A11_B14_mean')
    train, test = merge_sum(train, test, ['A11'], 'B14', 'A11_B14_sum')
    train, test = merge_max(train, test, ['A11'], 'B14', 'A11_B14_max')
    train, test = merge_min(train, test, ['A11'], 'B14', 'A11_B14_min')
    train, test = merge_std(train, test, ['A11'], 'B14', 'A11_B14_std')

    # ------------------------- A14 -----------------------#
    train, test = merge_count(train, test, ['A14'], 'A5', 'A14_A5_count')
    train, test = merge_nunique(train, test, ['A14'], 'A5', 'A14_A5_nunique')

    train, test = merge_median(train, test, ['A14'], 'A6', 'A14_A6_median')
    train, test = merge_mean(train, test, ['A14'], 'A6', 'A14_A6_mean')
    train, test = merge_sum(train, test, ['A14'], 'A6', 'A14_A6_sum')
    train, test = merge_max(train, test, ['A14'], 'A6', 'A14_A6_max')
    train, test = merge_min(train, test, ['A14'], 'A6', 'A14_A6_min')
    train, test = merge_std(train, test, ['A14'], 'A6', 'A14_A6_std')

    train, test = merge_median(train, test, ['A14'], 'A10', 'A14_A10_median')
    train, test = merge_mean(train, test, ['A14'], 'A10', 'A14_A10_mean')
    train, test = merge_sum(train, test, ['A14'], 'A10', 'A14_A10_sum')
    train, test = merge_max(train, test, ['A14'], 'A10', 'A14_A10_max')
    train, test = merge_min(train, test, ['A14'], 'A10', 'A14_A10_min')
    train, test = merge_std(train, test, ['A14'], 'A10', 'A14_A10_std')

    train, test = merge_median(train, test, ['A14'], 'A12', 'A14_A12_median')
    train, test = merge_mean(train, test, ['A14'], 'A12', 'A14_A12_mean')
    train, test = merge_sum(train, test, ['A14'], 'A12', 'A14_A12_sum')
    train, test = merge_max(train, test, ['A14'], 'A12', 'A14_A12_max')
    train, test = merge_min(train, test, ['A14'], 'A12', 'A14_A12_min')
    train, test = merge_std(train, test, ['A14'], 'A12', 'A14_A12_std')

    train, test = merge_median(train, test, ['A14'], 'A15', 'A14_A15_median')
    train, test = merge_mean(train, test, ['A14'], 'A15', 'A14_A15_mean')
    train, test = merge_sum(train, test, ['A14'], 'A15', 'A14_A15_sum')
    train, test = merge_max(train, test, ['A14'], 'A15', 'A14_A15_max')
    train, test = merge_min(train, test, ['A14'], 'A15', 'A14_A15_min')
    train, test = merge_std(train, test, ['A14'], 'A15', 'A14_A15_std')

    train, test = merge_median(train, test, ['A14'], 'A17', 'A14_A17_median')
    train, test = merge_mean(train, test, ['A14'], 'A17', 'A14_A17_mean')
    train, test = merge_sum(train, test, ['A14'], 'A17', 'A14_A17_sum')
    train, test = merge_max(train, test, ['A14'], 'A17', 'A14_A17_max')
    train, test = merge_min(train, test, ['A14'], 'A17', 'A14_A17_min')
    train, test = merge_std(train, test, ['A14'], 'A17', 'A14_A17_std')

    train, test = merge_median(train, test, ['A14'], 'A21', 'A14_A21_median')
    train, test = merge_mean(train, test, ['A14'], 'A21', 'A14_A21_mean')
    train, test = merge_sum(train, test, ['A14'], 'A21', 'A14_A21_sum')
    train, test = merge_max(train, test, ['A14'], 'A21', 'A14_A21_max')
    train, test = merge_min(train, test, ['A14'], 'A21', 'A14_A21_min')
    train, test = merge_std(train, test, ['A14'], 'A21', 'A14_A21_std')

    train, test = merge_median(train, test, ['A14'], 'A22', 'A14_A22_median')
    train, test = merge_mean(train, test, ['A14'], 'A22', 'A14_A22_mean')
    train, test = merge_sum(train, test, ['A14'], 'A22', 'A14_A22_sum')
    train, test = merge_max(train, test, ['A14'], 'A22', 'A14_A22_max')
    train, test = merge_min(train, test, ['A14'], 'A22', 'A14_A22_min')
    train, test = merge_std(train, test, ['A14'], 'A22', 'A14_A22_std')

    train, test = merge_count(train, test, ['A14'], 'A25', 'A14_A25_count')
    train, test = merge_nunique(train, test, ['A14'], 'A25', 'A14_A25_nunique')

    train, test = merge_median(train, test, ['A14'], 'A27', 'A14_A27_median')
    train, test = merge_mean(train, test, ['A14'], 'A27', 'A14_A27_mean')
    train, test = merge_sum(train, test, ['A14'], 'A27', 'A14_A27_sum')
    train, test = merge_max(train, test, ['A14'], 'A27', 'A14_A27_max')
    train, test = merge_min(train, test, ['A14'], 'A27', 'A14_A27_min')
    train, test = merge_std(train, test, ['A14'], 'A27', 'A14_A27_std')

    train, test = merge_median(train, test, ['A14'], 'B1', 'A14_B1_median')
    train, test = merge_mean(train, test, ['A14'], 'B1', 'A14_B1_mean')
    train, test = merge_sum(train, test, ['A14'], 'B1', 'A14_B1_sum')
    train, test = merge_max(train, test, ['A14'], 'B1', 'A14_B1_max')
    train, test = merge_min(train, test, ['A14'], 'B1', 'A14_B1_min')
    train, test = merge_std(train, test, ['A14'], 'B1', 'A14_B1_std')

    train, test = merge_count(train, test, ['A14'], 'B4', 'A14_B4_count')
    train, test = merge_nunique(train, test, ['A14'], 'B4', 'A14_B4_nunique')

    train, test = merge_count(train, test, ['A14'], 'B5', 'A14_B5_count')
    train, test = merge_nunique(train, test, ['A14'], 'B5', 'A14_B5_nunique')

    train, test = merge_median(train, test, ['A14'], 'B6', 'A14_B6_median')
    train, test = merge_mean(train, test, ['A14'], 'B6', 'A14_B6_mean')
    train, test = merge_sum(train, test, ['A14'], 'B6', 'A14_B6_sum')
    train, test = merge_max(train, test, ['A14'], 'B6', 'A14_B6_max')
    train, test = merge_min(train, test, ['A14'], 'B6', 'A14_B6_min')
    train, test = merge_std(train, test, ['A14'], 'B6', 'A14_B6_std')

    train, test = merge_count(train, test, ['A14'], 'B7', 'A14_B7_count')
    train, test = merge_nunique(train, test, ['A14'], 'B7', 'A14_B7_nunique')

    train, test = merge_median(train, test, ['A14'], 'B8', 'A14_B8_median')
    train, test = merge_mean(train, test, ['A14'], 'B8', 'A14_B8_mean')
    train, test = merge_sum(train, test, ['A14'], 'B8', 'A14_B8_sum')
    train, test = merge_max(train, test, ['A14'], 'B8', 'A14_B8_max')
    train, test = merge_min(train, test, ['A14'], 'B8', 'A14_B8_min')
    train, test = merge_std(train, test, ['A14'], 'B8', 'A14_B8_std')

    train, test = merge_count(train, test, ['A14'], 'B9', 'A14_B9_count')
    train, test = merge_nunique(train, test, ['A14'], 'B9', 'A14_B9_nunique')

    train, test = merge_count(train, test, ['A14'], 'B10', 'A14_B10_count')
    train, test = merge_nunique(train, test, ['A14'], 'B10', 'A14_B10_nunique')

    train, test = merge_count(train, test, ['A14'], 'B11', 'A14_B11_count')
    train, test = merge_nunique(train, test, ['A14'], 'B11', 'A14_B11_nunique')

    train, test = merge_median(train, test, ['A14'], 'B14', 'A14_B14_median')
    train, test = merge_mean(train, test, ['A14'], 'B14', 'A14_B14_mean')
    train, test = merge_sum(train, test, ['A14'], 'B14', 'A14_B14_sum')
    train, test = merge_max(train, test, ['A14'], 'B14', 'A14_B14_max')
    train, test = merge_min(train, test, ['A14'], 'B14', 'A14_B14_min')
    train, test = merge_std(train, test, ['A14'], 'B14', 'A14_B14_std')

    # ------------------------- A16 -----------------------#
    train, test = merge_count(train, test, ['A16'], 'A5', 'A16_A5_count')
    train, test = merge_nunique(train, test, ['A16'], 'A5', 'A16_A5_nunique')

    train, test = merge_median(train, test, ['A16'], 'A6', 'A16_A6_median')
    train, test = merge_mean(train, test, ['A16'], 'A6', 'A16_A6_mean')
    train, test = merge_sum(train, test, ['A16'], 'A6', 'A16_A6_sum')
    train, test = merge_max(train, test, ['A16'], 'A6', 'A16_A6_max')
    train, test = merge_min(train, test, ['A16'], 'A6', 'A16_A6_min')
    train, test = merge_std(train, test, ['A16'], 'A6', 'A16_A6_std')

    train, test = merge_median(train, test, ['A16'], 'A10', 'A16_A10_median')
    train, test = merge_mean(train, test, ['A16'], 'A10', 'A16_A10_mean')
    train, test = merge_sum(train, test, ['A16'], 'A10', 'A16_A10_sum')
    train, test = merge_max(train, test, ['A16'], 'A10', 'A16_A10_max')
    train, test = merge_min(train, test, ['A16'], 'A10', 'A16_A10_min')
    train, test = merge_std(train, test, ['A16'], 'A10', 'A16_A10_std')

    train, test = merge_median(train, test, ['A16'], 'A12', 'A16_A12_median')
    train, test = merge_mean(train, test, ['A16'], 'A12', 'A16_A12_mean')
    train, test = merge_sum(train, test, ['A16'], 'A12', 'A16_A12_sum')
    train, test = merge_max(train, test, ['A16'], 'A12', 'A16_A12_max')
    train, test = merge_min(train, test, ['A16'], 'A12', 'A16_A12_min')
    train, test = merge_std(train, test, ['A16'], 'A12', 'A16_A12_std')

    train, test = merge_median(train, test, ['A16'], 'A15', 'A16_A15_median')
    train, test = merge_mean(train, test, ['A16'], 'A15', 'A16_A15_mean')
    train, test = merge_sum(train, test, ['A16'], 'A15', 'A16_A15_sum')
    train, test = merge_max(train, test, ['A16'], 'A15', 'A16_A15_max')
    train, test = merge_min(train, test, ['A16'], 'A15', 'A16_A15_min')
    train, test = merge_std(train, test, ['A16'], 'A15', 'A16_A15_std')

    train, test = merge_median(train, test, ['A16'], 'A17', 'A16_A17_median')
    train, test = merge_mean(train, test, ['A16'], 'A17', 'A16_A17_mean')
    train, test = merge_sum(train, test, ['A16'], 'A17', 'A16_A17_sum')
    train, test = merge_max(train, test, ['A16'], 'A17', 'A16_A17_max')
    train, test = merge_min(train, test, ['A16'], 'A17', 'A16_A17_min')
    train, test = merge_std(train, test, ['A16'], 'A17', 'A16_A17_std')

    train, test = merge_median(train, test, ['A16'], 'A21', 'A16_A21_median')
    train, test = merge_mean(train, test, ['A16'], 'A21', 'A16_A21_mean')
    train, test = merge_sum(train, test, ['A16'], 'A21', 'A16_A21_sum')
    train, test = merge_max(train, test, ['A16'], 'A21', 'A16_A21_max')
    train, test = merge_min(train, test, ['A16'], 'A21', 'A16_A21_min')
    train, test = merge_std(train, test, ['A16'], 'A21', 'A16_A21_std')

    train, test = merge_median(train, test, ['A16'], 'A22', 'A16_A22_median')
    train, test = merge_mean(train, test, ['A16'], 'A22', 'A16_A22_mean')
    train, test = merge_sum(train, test, ['A16'], 'A22', 'A16_A22_sum')
    train, test = merge_max(train, test, ['A16'], 'A22', 'A16_A22_max')
    train, test = merge_min(train, test, ['A16'], 'A22', 'A16_A22_min')
    train, test = merge_std(train, test, ['A16'], 'A22', 'A16_A22_std')

    train, test = merge_count(train, test, ['A16'], 'A25', 'A16_A25_count')
    train, test = merge_nunique(train, test, ['A16'], 'A25', 'A16_A25_nunique')

    train, test = merge_median(train, test, ['A16'], 'A27', 'A16_A27_median')
    train, test = merge_mean(train, test, ['A16'], 'A27', 'A16_A27_mean')
    train, test = merge_sum(train, test, ['A16'], 'A27', 'A16_A27_sum')
    train, test = merge_max(train, test, ['A16'], 'A27', 'A16_A27_max')
    train, test = merge_min(train, test, ['A16'], 'A27', 'A16_A27_min')
    train, test = merge_std(train, test, ['A16'], 'A27', 'A16_A27_std')

    train, test = merge_median(train, test, ['A16'], 'B1', 'A16_B1_median')
    train, test = merge_mean(train, test, ['A16'], 'B1', 'A16_B1_mean')
    train, test = merge_sum(train, test, ['A16'], 'B1', 'A16_B1_sum')
    train, test = merge_max(train, test, ['A16'], 'B1', 'A16_B1_max')
    train, test = merge_min(train, test, ['A16'], 'B1', 'A16_B1_min')
    train, test = merge_std(train, test, ['A16'], 'B1', 'A16_B1_std')

    train, test = merge_count(train, test, ['A16'], 'B4', 'A16_B4_count')
    train, test = merge_nunique(train, test, ['A16'], 'B4', 'A16_B4_nunique')

    train, test = merge_count(train, test, ['A16'], 'B5', 'A16_B5_count')
    train, test = merge_nunique(train, test, ['A16'], 'B5', 'A16_B5_nunique')

    train, test = merge_median(train, test, ['A16'], 'B6', 'A16_B6_median')
    train, test = merge_mean(train, test, ['A16'], 'B6', 'A16_B6_mean')
    train, test = merge_sum(train, test, ['A16'], 'B6', 'A16_B6_sum')
    train, test = merge_max(train, test, ['A16'], 'B6', 'A16_B6_max')
    train, test = merge_min(train, test, ['A16'], 'B6', 'A16_B6_min')
    train, test = merge_std(train, test, ['A16'], 'B6', 'A16_B6_std')

    train, test = merge_count(train, test, ['A16'], 'B7', 'A16_B7_count')
    train, test = merge_nunique(train, test, ['A16'], 'B7', 'A16_B7_nunique')

    train, test = merge_median(train, test, ['A16'], 'B8', 'A16_B8_median')
    train, test = merge_mean(train, test, ['A16'], 'B8', 'A16_B8_mean')
    train, test = merge_sum(train, test, ['A16'], 'B8', 'A16_B8_sum')
    train, test = merge_max(train, test, ['A16'], 'B8', 'A16_B8_max')
    train, test = merge_min(train, test, ['A16'], 'B8', 'A16_B8_min')
    train, test = merge_std(train, test, ['A16'], 'B8', 'A16_B8_std')

    train, test = merge_count(train, test, ['A16'], 'B9', 'A16_B9_count')
    train, test = merge_nunique(train, test, ['A16'], 'B9', 'A16_B9_nunique')

    train, test = merge_count(train, test, ['A16'], 'B10', 'A16_B10_count')
    train, test = merge_nunique(train, test, ['A16'], 'B10', 'A16_B10_nunique')

    train, test = merge_count(train, test, ['A16'], 'B11', 'A16_B11_count')
    train, test = merge_nunique(train, test, ['A16'], 'B11', 'A16_B11_nunique')

    train, test = merge_median(train, test, ['A16'], 'B14', 'A16_B14_median')
    train, test = merge_mean(train, test, ['A16'], 'B14', 'A16_B14_mean')
    train, test = merge_sum(train, test, ['A16'], 'B14', 'A16_B14_sum')
    train, test = merge_max(train, test, ['A16'], 'B14', 'A16_B14_max')
    train, test = merge_min(train, test, ['A16'], 'B14', 'A16_B14_min')
    train, test = merge_std(train, test, ['A16'], 'B14', 'A16_B14_std')

    # ------------------------- A20 -----------------------#
    train, test = merge_count(train, test, ['A20'], 'A5', 'A20_A5_count')
    train, test = merge_nunique(train, test, ['A20'], 'A5', 'A20_A5_nunique')

    train, test = merge_median(train, test, ['A20'], 'A6', 'A20_A6_median')
    train, test = merge_mean(train, test, ['A20'], 'A6', 'A20_A6_mean')
    train, test = merge_sum(train, test, ['A20'], 'A6', 'A20_A6_sum')
    train, test = merge_max(train, test, ['A20'], 'A6', 'A20_A6_max')
    train, test = merge_min(train, test, ['A20'], 'A6', 'A20_A6_min')
    train, test = merge_std(train, test, ['A20'], 'A6', 'A20_A6_std')

    train, test = merge_median(train, test, ['A20'], 'A10', 'A20_A10_median')
    train, test = merge_mean(train, test, ['A20'], 'A10', 'A20_A10_mean')
    train, test = merge_sum(train, test, ['A20'], 'A10', 'A20_A10_sum')
    train, test = merge_max(train, test, ['A20'], 'A10', 'A20_A10_max')
    train, test = merge_min(train, test, ['A20'], 'A10', 'A20_A10_min')
    train, test = merge_std(train, test, ['A20'], 'A10', 'A20_A10_std')

    train, test = merge_median(train, test, ['A20'], 'A12', 'A20_A12_median')
    train, test = merge_mean(train, test, ['A20'], 'A12', 'A20_A12_mean')
    train, test = merge_sum(train, test, ['A20'], 'A12', 'A20_A12_sum')
    train, test = merge_max(train, test, ['A20'], 'A12', 'A20_A12_max')
    train, test = merge_min(train, test, ['A20'], 'A12', 'A20_A12_min')
    train, test = merge_std(train, test, ['A20'], 'A12', 'A20_A12_std')

    train, test = merge_median(train, test, ['A20'], 'A15', 'A20_A15_median')
    train, test = merge_mean(train, test, ['A20'], 'A15', 'A20_A15_mean')
    train, test = merge_sum(train, test, ['A20'], 'A15', 'A20_A15_sum')
    train, test = merge_max(train, test, ['A20'], 'A15', 'A20_A15_max')
    train, test = merge_min(train, test, ['A20'], 'A15', 'A20_A15_min')
    train, test = merge_std(train, test, ['A20'], 'A15', 'A20_A15_std')

    train, test = merge_median(train, test, ['A20'], 'A17', 'A20_A17_median')
    train, test = merge_mean(train, test, ['A20'], 'A17', 'A20_A17_mean')
    train, test = merge_sum(train, test, ['A20'], 'A17', 'A20_A17_sum')
    train, test = merge_max(train, test, ['A20'], 'A17', 'A20_A17_max')
    train, test = merge_min(train, test, ['A20'], 'A17', 'A20_A17_min')
    train, test = merge_std(train, test, ['A20'], 'A17', 'A20_A17_std')

    train, test = merge_median(train, test, ['A20'], 'A21', 'A20_A21_median')
    train, test = merge_mean(train, test, ['A20'], 'A21', 'A20_A21_mean')
    train, test = merge_sum(train, test, ['A20'], 'A21', 'A20_A21_sum')
    train, test = merge_max(train, test, ['A20'], 'A21', 'A20_A21_max')
    train, test = merge_min(train, test, ['A20'], 'A21', 'A20_A21_min')
    train, test = merge_std(train, test, ['A20'], 'A21', 'A20_A21_std')

    train, test = merge_median(train, test, ['A20'], 'A22', 'A20_A22_median')
    train, test = merge_mean(train, test, ['A20'], 'A22', 'A20_A22_mean')
    train, test = merge_sum(train, test, ['A20'], 'A22', 'A20_A22_sum')
    train, test = merge_max(train, test, ['A20'], 'A22', 'A20_A22_max')
    train, test = merge_min(train, test, ['A20'], 'A22', 'A20_A22_min')
    train, test = merge_std(train, test, ['A20'], 'A22', 'A20_A22_std')

    train, test = merge_count(train, test, ['A20'], 'A25', 'A20_A25_count')
    train, test = merge_nunique(train, test, ['A20'], 'A25', 'A20_A25_nunique')

    train, test = merge_median(train, test, ['A20'], 'A27', 'A20_A27_median')
    train, test = merge_mean(train, test, ['A20'], 'A27', 'A20_A27_mean')
    train, test = merge_sum(train, test, ['A20'], 'A27', 'A20_A27_sum')
    train, test = merge_max(train, test, ['A20'], 'A27', 'A20_A27_max')
    train, test = merge_min(train, test, ['A20'], 'A27', 'A20_A27_min')
    train, test = merge_std(train, test, ['A20'], 'A27', 'A20_A27_std')

    train, test = merge_median(train, test, ['A20'], 'B1', 'A20_B1_median')
    train, test = merge_mean(train, test, ['A20'], 'B1', 'A20_B1_mean')
    train, test = merge_sum(train, test, ['A20'], 'B1', 'A20_B1_sum')
    train, test = merge_max(train, test, ['A20'], 'B1', 'A20_B1_max')
    train, test = merge_min(train, test, ['A20'], 'B1', 'A20_B1_min')
    train, test = merge_std(train, test, ['A20'], 'B1', 'A20_B1_std')

    train, test = merge_count(train, test, ['A20'], 'B4', 'A20_B4_count')
    train, test = merge_nunique(train, test, ['A20'], 'B4', 'A20_B4_nunique')

    train, test = merge_count(train, test, ['A20'], 'B5', 'A20_B5_count')
    train, test = merge_nunique(train, test, ['A20'], 'B5', 'A20_B5_nunique')

    train, test = merge_median(train, test, ['A20'], 'B6', 'A20_B6_median')
    train, test = merge_mean(train, test, ['A20'], 'B6', 'A20_B6_mean')
    train, test = merge_sum(train, test, ['A20'], 'B6', 'A20_B6_sum')
    train, test = merge_max(train, test, ['A20'], 'B6', 'A20_B6_max')
    train, test = merge_min(train, test, ['A20'], 'B6', 'A20_B6_min')
    train, test = merge_std(train, test, ['A20'], 'B6', 'A20_B6_std')

    train, test = merge_count(train, test, ['A20'], 'B7', 'A20_B7_count')
    train, test = merge_nunique(train, test, ['A20'], 'B7', 'A20_B7_nunique')

    train, test = merge_median(train, test, ['A20'], 'B8', 'A20_B8_median')
    train, test = merge_mean(train, test, ['A20'], 'B8', 'A20_B8_mean')
    train, test = merge_sum(train, test, ['A20'], 'B8', 'A20_B8_sum')
    train, test = merge_max(train, test, ['A20'], 'B8', 'A20_B8_max')
    train, test = merge_min(train, test, ['A20'], 'B8', 'A20_B8_min')
    train, test = merge_std(train, test, ['A20'], 'B8', 'A20_B8_std')

    train, test = merge_count(train, test, ['A20'], 'B9', 'A20_B9_count')
    train, test = merge_nunique(train, test, ['A20'], 'B9', 'A20_B9_nunique')

    train, test = merge_count(train, test, ['A20'], 'B10', 'A20_B10_count')
    train, test = merge_nunique(train, test, ['A20'], 'B10', 'A20_B10_nunique')

    train, test = merge_count(train, test, ['A20'], 'B11', 'A20_B11_count')
    train, test = merge_nunique(train, test, ['A20'], 'B11', 'A20_B11_nunique')

    train, test = merge_median(train, test, ['A20'], 'B14', 'A20_B14_median')
    train, test = merge_mean(train, test, ['A20'], 'B14', 'A20_B14_mean')
    train, test = merge_sum(train, test, ['A20'], 'B14', 'A20_B14_sum')
    train, test = merge_max(train, test, ['A20'], 'B14', 'A20_B14_max')
    train, test = merge_min(train, test, ['A20'], 'B14', 'A20_B14_min')
    train, test = merge_std(train, test, ['A20'], 'B14', 'A20_B14_std')

    # ------------------------- A24 -----------------------#
    train, test = merge_count(train, test, ['A24'], 'A5', 'A24_A5_count')
    train, test = merge_nunique(train, test, ['A24'], 'A5', 'A24_A5_nunique')

    train, test = merge_median(train, test, ['A24'], 'A6', 'A24_A6_median')
    train, test = merge_mean(train, test, ['A24'], 'A6', 'A24_A6_mean')
    train, test = merge_sum(train, test, ['A24'], 'A6', 'A24_A6_sum')
    train, test = merge_max(train, test, ['A24'], 'A6', 'A24_A6_max')
    train, test = merge_min(train, test, ['A24'], 'A6', 'A24_A6_min')
    train, test = merge_std(train, test, ['A24'], 'A6', 'A24_A6_std')

    train, test = merge_median(train, test, ['A24'], 'A10', 'A24_A10_median')
    train, test = merge_mean(train, test, ['A24'], 'A10', 'A24_A10_mean')
    train, test = merge_sum(train, test, ['A24'], 'A10', 'A24_A10_sum')
    train, test = merge_max(train, test, ['A24'], 'A10', 'A24_A10_max')
    train, test = merge_min(train, test, ['A24'], 'A10', 'A24_A10_min')
    train, test = merge_std(train, test, ['A24'], 'A10', 'A24_A10_std')

    train, test = merge_median(train, test, ['A24'], 'A12', 'A24_A12_median')
    train, test = merge_mean(train, test, ['A24'], 'A12', 'A24_A12_mean')
    train, test = merge_sum(train, test, ['A24'], 'A12', 'A24_A12_sum')
    train, test = merge_max(train, test, ['A24'], 'A12', 'A24_A12_max')
    train, test = merge_min(train, test, ['A24'], 'A12', 'A24_A12_min')
    train, test = merge_std(train, test, ['A24'], 'A12', 'A24_A12_std')

    train, test = merge_median(train, test, ['A24'], 'A15', 'A24_A15_median')
    train, test = merge_mean(train, test, ['A24'], 'A15', 'A24_A15_mean')
    train, test = merge_sum(train, test, ['A24'], 'A15', 'A24_A15_sum')
    train, test = merge_max(train, test, ['A24'], 'A15', 'A24_A15_max')
    train, test = merge_min(train, test, ['A24'], 'A15', 'A24_A15_min')
    train, test = merge_std(train, test, ['A24'], 'A15', 'A24_A15_std')

    train, test = merge_median(train, test, ['A24'], 'A17', 'A24_A17_median')
    train, test = merge_mean(train, test, ['A24'], 'A17', 'A24_A17_mean')
    train, test = merge_sum(train, test, ['A24'], 'A17', 'A24_A17_sum')
    train, test = merge_max(train, test, ['A24'], 'A17', 'A24_A17_max')
    train, test = merge_min(train, test, ['A24'], 'A17', 'A24_A17_min')
    train, test = merge_std(train, test, ['A24'], 'A17', 'A24_A17_std')

    train, test = merge_median(train, test, ['A24'], 'A21', 'A24_A21_median')
    train, test = merge_mean(train, test, ['A24'], 'A21', 'A24_A21_mean')
    train, test = merge_sum(train, test, ['A24'], 'A21', 'A24_A21_sum')
    train, test = merge_max(train, test, ['A24'], 'A21', 'A24_A21_max')
    train, test = merge_min(train, test, ['A24'], 'A21', 'A24_A21_min')
    train, test = merge_std(train, test, ['A24'], 'A21', 'A24_A21_std')

    train, test = merge_median(train, test, ['A24'], 'A22', 'A24_A22_median')
    train, test = merge_mean(train, test, ['A24'], 'A22', 'A24_A22_mean')
    train, test = merge_sum(train, test, ['A24'], 'A22', 'A24_A22_sum')
    train, test = merge_max(train, test, ['A24'], 'A22', 'A24_A22_max')
    train, test = merge_min(train, test, ['A24'], 'A22', 'A24_A22_min')
    train, test = merge_std(train, test, ['A24'], 'A22', 'A24_A22_std')

    train, test = merge_count(train, test, ['A24'], 'A25', 'A24_A25_count')
    train, test = merge_nunique(train, test, ['A24'], 'A25', 'A24_A25_nunique')

    train, test = merge_median(train, test, ['A24'], 'A27', 'A24_A27_median')
    train, test = merge_mean(train, test, ['A24'], 'A27', 'A24_A27_mean')
    train, test = merge_sum(train, test, ['A24'], 'A27', 'A24_A27_sum')
    train, test = merge_max(train, test, ['A24'], 'A27', 'A24_A27_max')
    train, test = merge_min(train, test, ['A24'], 'A27', 'A24_A27_min')
    train, test = merge_std(train, test, ['A24'], 'A27', 'A24_A27_std')

    train, test = merge_median(train, test, ['A24'], 'B1', 'A24_B1_median')
    train, test = merge_mean(train, test, ['A24'], 'B1', 'A24_B1_mean')
    train, test = merge_sum(train, test, ['A24'], 'B1', 'A24_B1_sum')
    train, test = merge_max(train, test, ['A24'], 'B1', 'A24_B1_max')
    train, test = merge_min(train, test, ['A24'], 'B1', 'A24_B1_min')
    train, test = merge_std(train, test, ['A24'], 'B1', 'A24_B1_std')

    train, test = merge_count(train, test, ['A24'], 'B4', 'A24_B4_count')
    train, test = merge_nunique(train, test, ['A24'], 'B4', 'A24_B4_nunique')

    train, test = merge_count(train, test, ['A24'], 'B5', 'A24_B5_count')
    train, test = merge_nunique(train, test, ['A24'], 'B5', 'A24_B5_nunique')

    train, test = merge_median(train, test, ['A24'], 'B6', 'A24_B6_median')
    train, test = merge_mean(train, test, ['A24'], 'B6', 'A24_B6_mean')
    train, test = merge_sum(train, test, ['A24'], 'B6', 'A24_B6_sum')
    train, test = merge_max(train, test, ['A24'], 'B6', 'A24_B6_max')
    train, test = merge_min(train, test, ['A24'], 'B6', 'A24_B6_min')
    train, test = merge_std(train, test, ['A24'], 'B6', 'A24_B6_std')

    train, test = merge_count(train, test, ['A24'], 'B7', 'A24_B7_count')
    train, test = merge_nunique(train, test, ['A24'], 'B7', 'A24_B7_nunique')

    train, test = merge_median(train, test, ['A24'], 'B8', 'A24_B8_median')
    train, test = merge_mean(train, test, ['A24'], 'B8', 'A24_B8_mean')
    train, test = merge_sum(train, test, ['A24'], 'B8', 'A24_B8_sum')
    train, test = merge_max(train, test, ['A24'], 'B8', 'A24_B8_max')
    train, test = merge_min(train, test, ['A24'], 'B8', 'A24_B8_min')
    train, test = merge_std(train, test, ['A24'], 'B8', 'A24_B8_std')

    train, test = merge_count(train, test, ['A24'], 'B9', 'A24_B9_count')
    train, test = merge_nunique(train, test, ['A24'], 'B9', 'A24_B9_nunique')

    train, test = merge_count(train, test, ['A24'], 'B10', 'A24_B10_count')
    train, test = merge_nunique(train, test, ['A24'], 'B10', 'A24_B10_nunique')

    train, test = merge_count(train, test, ['A24'], 'B11', 'A24_B11_count')
    train, test = merge_nunique(train, test, ['A24'], 'B11', 'A24_B11_nunique')

    train, test = merge_median(train, test, ['A24'], 'B14', 'A24_B14_median')
    train, test = merge_mean(train, test, ['A24'], 'B14', 'A24_B14_mean')
    train, test = merge_sum(train, test, ['A24'], 'B14', 'A24_B14_sum')
    train, test = merge_max(train, test, ['A24'], 'B14', 'A24_B14_max')
    train, test = merge_min(train, test, ['A24'], 'B14', 'A24_B14_min')
    train, test = merge_std(train, test, ['A24'], 'B14', 'A24_B14_std')

    # ------------------------- A25 -----------------------#
    train, test = merge_count(train, test, ['A25'], 'A5', 'A25_A5_count')
    train, test = merge_nunique(train, test, ['A25'], 'A5', 'A25_A5_nunique')

    train, test = merge_median(train, test, ['A25'], 'A6', 'A25_A6_median')
    train, test = merge_mean(train, test, ['A25'], 'A6', 'A25_A6_mean')
    train, test = merge_sum(train, test, ['A25'], 'A6', 'A25_A6_sum')
    train, test = merge_max(train, test, ['A25'], 'A6', 'A25_A6_max')
    train, test = merge_min(train, test, ['A25'], 'A6', 'A25_A6_min')
    train, test = merge_std(train, test, ['A25'], 'A6', 'A25_A6_std')

    train, test = merge_median(train, test, ['A25'], 'A10', 'A25_A10_median')
    train, test = merge_mean(train, test, ['A25'], 'A10', 'A25_A10_mean')
    train, test = merge_sum(train, test, ['A25'], 'A10', 'A25_A10_sum')
    train, test = merge_max(train, test, ['A25'], 'A10', 'A25_A10_max')
    train, test = merge_min(train, test, ['A25'], 'A10', 'A25_A10_min')
    train, test = merge_std(train, test, ['A25'], 'A10', 'A25_A10_std')

    train, test = merge_median(train, test, ['A25'], 'A12', 'A25_A12_median')
    train, test = merge_mean(train, test, ['A25'], 'A12', 'A25_A12_mean')
    train, test = merge_sum(train, test, ['A25'], 'A12', 'A25_A12_sum')
    train, test = merge_max(train, test, ['A25'], 'A12', 'A25_A12_max')
    train, test = merge_min(train, test, ['A25'], 'A12', 'A25_A12_min')
    train, test = merge_std(train, test, ['A25'], 'A12', 'A25_A12_std')

    train, test = merge_median(train, test, ['A25'], 'A15', 'A25_A15_median')
    train, test = merge_mean(train, test, ['A25'], 'A15', 'A25_A15_mean')
    train, test = merge_sum(train, test, ['A25'], 'A15', 'A25_A15_sum')
    train, test = merge_max(train, test, ['A25'], 'A15', 'A25_A15_max')
    train, test = merge_min(train, test, ['A25'], 'A15', 'A25_A15_min')
    train, test = merge_std(train, test, ['A25'], 'A15', 'A25_A15_std')

    train, test = merge_median(train, test, ['A25'], 'A17', 'A25_A17_median')
    train, test = merge_mean(train, test, ['A25'], 'A17', 'A25_A17_mean')
    train, test = merge_sum(train, test, ['A25'], 'A17', 'A25_A17_sum')
    train, test = merge_max(train, test, ['A25'], 'A17', 'A25_A17_max')
    train, test = merge_min(train, test, ['A25'], 'A17', 'A25_A17_min')
    train, test = merge_std(train, test, ['A25'], 'A17', 'A25_A17_std')

    train, test = merge_median(train, test, ['A25'], 'A21', 'A25_A21_median')
    train, test = merge_mean(train, test, ['A25'], 'A21', 'A25_A21_mean')
    train, test = merge_sum(train, test, ['A25'], 'A21', 'A25_A21_sum')
    train, test = merge_max(train, test, ['A25'], 'A21', 'A25_A21_max')
    train, test = merge_min(train, test, ['A25'], 'A21', 'A25_A21_min')
    train, test = merge_std(train, test, ['A25'], 'A21', 'A25_A21_std')

    train, test = merge_median(train, test, ['A25'], 'A22', 'A25_A22_median')
    train, test = merge_mean(train, test, ['A25'], 'A22', 'A25_A22_mean')
    train, test = merge_sum(train, test, ['A25'], 'A22', 'A25_A22_sum')
    train, test = merge_max(train, test, ['A25'], 'A22', 'A25_A22_max')
    train, test = merge_min(train, test, ['A25'], 'A22', 'A25_A22_min')
    train, test = merge_std(train, test, ['A25'], 'A22', 'A25_A22_std')

    train, test = merge_median(train, test, ['A25'], 'A27', 'A25_A27_median')
    train, test = merge_mean(train, test, ['A25'], 'A27', 'A25_A27_mean')
    train, test = merge_sum(train, test, ['A25'], 'A27', 'A25_A27_sum')
    train, test = merge_max(train, test, ['A25'], 'A27', 'A25_A27_max')
    train, test = merge_min(train, test, ['A25'], 'A27', 'A25_A27_min')
    train, test = merge_std(train, test, ['A25'], 'A27', 'A25_A27_std')

    train, test = merge_median(train, test, ['A25'], 'B1', 'A25_B1_median')
    train, test = merge_mean(train, test, ['A25'], 'B1', 'A25_B1_mean')
    train, test = merge_sum(train, test, ['A25'], 'B1', 'A25_B1_sum')
    train, test = merge_max(train, test, ['A25'], 'B1', 'A25_B1_max')
    train, test = merge_min(train, test, ['A25'], 'B1', 'A25_B1_min')
    train, test = merge_std(train, test, ['A25'], 'B1', 'A25_B1_std')

    train, test = merge_count(train, test, ['A25'], 'B4', 'A25_B4_count')
    train, test = merge_nunique(train, test, ['A25'], 'B4', 'A25_B4_nunique')

    train, test = merge_count(train, test, ['A25'], 'B5', 'A25_B5_count')
    train, test = merge_nunique(train, test, ['A25'], 'B5', 'A25_B5_nunique')

    train, test = merge_median(train, test, ['A25'], 'B6', 'A25_B6_median')
    train, test = merge_mean(train, test, ['A25'], 'B6', 'A25_B6_mean')
    train, test = merge_sum(train, test, ['A25'], 'B6', 'A25_B6_sum')
    train, test = merge_max(train, test, ['A25'], 'B6', 'A25_B6_max')
    train, test = merge_min(train, test, ['A25'], 'B6', 'A25_B6_min')
    train, test = merge_std(train, test, ['A25'], 'B6', 'A25_B6_std')

    train, test = merge_count(train, test, ['A25'], 'B7', 'A25_B7_count')
    train, test = merge_nunique(train, test, ['A25'], 'B7', 'A25_B7_nunique')

    train, test = merge_median(train, test, ['A25'], 'B8', 'A25_B8_median')
    train, test = merge_mean(train, test, ['A25'], 'B8', 'A25_B8_mean')
    train, test = merge_sum(train, test, ['A25'], 'B8', 'A25_B8_sum')
    train, test = merge_max(train, test, ['A25'], 'B8', 'A25_B8_max')
    train, test = merge_min(train, test, ['A25'], 'B8', 'A25_B8_min')
    train, test = merge_std(train, test, ['A25'], 'B8', 'A25_B8_std')

    train, test = merge_count(train, test, ['A25'], 'B9', 'A25_B9_count')
    train, test = merge_nunique(train, test, ['A25'], 'B9', 'A25_B9_nunique')

    train, test = merge_count(train, test, ['A25'], 'B10', 'A25_B10_count')
    train, test = merge_nunique(train, test, ['A25'], 'B10', 'A25_B10_nunique')

    train, test = merge_count(train, test, ['A25'], 'B11', 'A25_B11_count')
    train, test = merge_nunique(train, test, ['A25'], 'B11', 'A25_B11_nunique')

    train, test = merge_median(train, test, ['A25'], 'B14', 'A25_B14_median')
    train, test = merge_mean(train, test, ['A25'], 'B14', 'A25_B14_mean')
    train, test = merge_sum(train, test, ['A25'], 'B14', 'A25_B14_sum')
    train, test = merge_max(train, test, ['A25'], 'B14', 'A25_B14_max')
    train, test = merge_min(train, test, ['A25'], 'B14', 'A25_B14_min')
    train, test = merge_std(train, test, ['A25'], 'B14', 'A25_B14_std')

    # ------------------------- A26 -----------------------#
    train, test = merge_count(train, test, ['A26'], 'A5', 'A26_A5_count')
    train, test = merge_nunique(train, test, ['A26'], 'A5', 'A26_A5_nunique')

    train, test = merge_median(train, test, ['A26'], 'A6', 'A26_A6_median')
    train, test = merge_mean(train, test, ['A26'], 'A6', 'A26_A6_mean')
    train, test = merge_sum(train, test, ['A26'], 'A6', 'A26_A6_sum')
    train, test = merge_max(train, test, ['A26'], 'A6', 'A26_A6_max')
    train, test = merge_min(train, test, ['A26'], 'A6', 'A26_A6_min')
    train, test = merge_std(train, test, ['A26'], 'A6', 'A26_A6_std')

    train, test = merge_median(train, test, ['A26'], 'A10', 'A26_A10_median')
    train, test = merge_mean(train, test, ['A26'], 'A10', 'A26_A10_mean')
    train, test = merge_sum(train, test, ['A26'], 'A10', 'A26_A10_sum')
    train, test = merge_max(train, test, ['A26'], 'A10', 'A26_A10_max')
    train, test = merge_min(train, test, ['A26'], 'A10', 'A26_A10_min')
    train, test = merge_std(train, test, ['A26'], 'A10', 'A26_A10_std')

    train, test = merge_median(train, test, ['A26'], 'A12', 'A26_A12_median')
    train, test = merge_mean(train, test, ['A26'], 'A12', 'A26_A12_mean')
    train, test = merge_sum(train, test, ['A26'], 'A12', 'A26_A12_sum')
    train, test = merge_max(train, test, ['A26'], 'A12', 'A26_A12_max')
    train, test = merge_min(train, test, ['A26'], 'A12', 'A26_A12_min')
    train, test = merge_std(train, test, ['A26'], 'A12', 'A26_A12_std')

    train, test = merge_median(train, test, ['A26'], 'A15', 'A26_A15_median')
    train, test = merge_mean(train, test, ['A26'], 'A15', 'A26_A15_mean')
    train, test = merge_sum(train, test, ['A26'], 'A15', 'A26_A15_sum')
    train, test = merge_max(train, test, ['A26'], 'A15', 'A26_A15_max')
    train, test = merge_min(train, test, ['A26'], 'A15', 'A26_A15_min')
    train, test = merge_std(train, test, ['A26'], 'A15', 'A26_A15_std')

    train, test = merge_median(train, test, ['A26'], 'A17', 'A26_A17_median')
    train, test = merge_mean(train, test, ['A26'], 'A17', 'A26_A17_mean')
    train, test = merge_sum(train, test, ['A26'], 'A17', 'A26_A17_sum')
    train, test = merge_max(train, test, ['A26'], 'A17', 'A26_A17_max')
    train, test = merge_min(train, test, ['A26'], 'A17', 'A26_A17_min')
    train, test = merge_std(train, test, ['A26'], 'A17', 'A26_A17_std')

    train, test = merge_median(train, test, ['A26'], 'A21', 'A26_A21_median')
    train, test = merge_mean(train, test, ['A26'], 'A21', 'A26_A21_mean')
    train, test = merge_sum(train, test, ['A26'], 'A21', 'A26_A21_sum')
    train, test = merge_max(train, test, ['A26'], 'A21', 'A26_A21_max')
    train, test = merge_min(train, test, ['A26'], 'A21', 'A26_A21_min')
    train, test = merge_std(train, test, ['A26'], 'A21', 'A26_A21_std')

    train, test = merge_median(train, test, ['A26'], 'A22', 'A26_A22_median')
    train, test = merge_mean(train, test, ['A26'], 'A22', 'A26_A22_mean')
    train, test = merge_sum(train, test, ['A26'], 'A22', 'A26_A22_sum')
    train, test = merge_max(train, test, ['A26'], 'A22', 'A26_A22_max')
    train, test = merge_min(train, test, ['A26'], 'A22', 'A26_A22_min')
    train, test = merge_std(train, test, ['A26'], 'A22', 'A26_A22_std')

    train, test = merge_count(train, test, ['A26'], 'A25', 'A26_A25_count')
    train, test = merge_nunique(train, test, ['A26'], 'A25', 'A26_A25_nunique')

    train, test = merge_median(train, test, ['A26'], 'A27', 'A26_A27_median')
    train, test = merge_mean(train, test, ['A26'], 'A27', 'A26_A27_mean')
    train, test = merge_sum(train, test, ['A26'], 'A27', 'A26_A27_sum')
    train, test = merge_max(train, test, ['A26'], 'A27', 'A26_A27_max')
    train, test = merge_min(train, test, ['A26'], 'A27', 'A26_A27_min')
    train, test = merge_std(train, test, ['A26'], 'A27', 'A26_A27_std')

    train, test = merge_median(train, test, ['A26'], 'B1', 'A26_B1_median')
    train, test = merge_mean(train, test, ['A26'], 'B1', 'A26_B1_mean')
    train, test = merge_sum(train, test, ['A26'], 'B1', 'A26_B1_sum')
    train, test = merge_max(train, test, ['A26'], 'B1', 'A26_B1_max')
    train, test = merge_min(train, test, ['A26'], 'B1', 'A26_B1_min')
    train, test = merge_std(train, test, ['A26'], 'B1', 'A26_B1_std')

    train, test = merge_count(train, test, ['A26'], 'B4', 'A26_B4_count')
    train, test = merge_nunique(train, test, ['A26'], 'B4', 'A26_B4_nunique')

    train, test = merge_count(train, test, ['A26'], 'B5', 'A26_B5_count')
    train, test = merge_nunique(train, test, ['A26'], 'B5', 'A26_B5_nunique')

    train, test = merge_median(train, test, ['A26'], 'B6', 'A26_B6_median')
    train, test = merge_mean(train, test, ['A26'], 'B6', 'A26_B6_mean')
    train, test = merge_sum(train, test, ['A26'], 'B6', 'A26_B6_sum')
    train, test = merge_max(train, test, ['A26'], 'B6', 'A26_B6_max')
    train, test = merge_min(train, test, ['A26'], 'B6', 'A26_B6_min')
    train, test = merge_std(train, test, ['A26'], 'B6', 'A26_B6_std')

    train, test = merge_count(train, test, ['A26'], 'B7', 'A26_B7_count')
    train, test = merge_nunique(train, test, ['A26'], 'B7', 'A26_B7_nunique')

    train, test = merge_median(train, test, ['A26'], 'B8', 'A26_B8_median')
    train, test = merge_mean(train, test, ['A26'], 'B8', 'A26_B8_mean')
    train, test = merge_sum(train, test, ['A26'], 'B8', 'A26_B8_sum')
    train, test = merge_max(train, test, ['A26'], 'B8', 'A26_B8_max')
    train, test = merge_min(train, test, ['A26'], 'B8', 'A26_B8_min')
    train, test = merge_std(train, test, ['A26'], 'B8', 'A26_B8_std')

    train, test = merge_count(train, test, ['A26'], 'B9', 'A26_B9_count')
    train, test = merge_nunique(train, test, ['A26'], 'B9', 'A26_B9_nunique')

    train, test = merge_count(train, test, ['A26'], 'B10', 'A26_B10_count')
    train, test = merge_nunique(train, test, ['A26'], 'B10', 'A26_B10_nunique')

    train, test = merge_count(train, test, ['A26'], 'B11', 'A26_B11_count')
    train, test = merge_nunique(train, test, ['A26'], 'B11', 'A26_B11_nunique')

    train, test = merge_median(train, test, ['A26'], 'B14', 'A26_B14_median')
    train, test = merge_mean(train, test, ['A26'], 'B14', 'A26_B14_mean')
    train, test = merge_sum(train, test, ['A26'], 'B14', 'A26_B14_sum')
    train, test = merge_max(train, test, ['A26'], 'B14', 'A26_B14_max')
    train, test = merge_min(train, test, ['A26'], 'B14', 'A26_B14_min')
    train, test = merge_std(train, test, ['A26'], 'B14', 'A26_B14_std')

    # ------------------------- A28 -----------------------#
    train, test = merge_count(train, test, ['A28'], 'A5', 'A28_A5_count')
    train, test = merge_nunique(train, test, ['A28'], 'A5', 'A28_A5_nunique')

    train, test = merge_median(train, test, ['A28'], 'A6', 'A28_A6_median')
    train, test = merge_mean(train, test, ['A28'], 'A6', 'A28_A6_mean')
    train, test = merge_sum(train, test, ['A28'], 'A6', 'A28_A6_sum')
    train, test = merge_max(train, test, ['A28'], 'A6', 'A28_A6_max')
    train, test = merge_min(train, test, ['A28'], 'A6', 'A28_A6_min')
    train, test = merge_std(train, test, ['A28'], 'A6', 'A28_A6_std')

    train, test = merge_median(train, test, ['A28'], 'A10', 'A28_A10_median')
    train, test = merge_mean(train, test, ['A28'], 'A10', 'A28_A10_mean')
    train, test = merge_sum(train, test, ['A28'], 'A10', 'A28_A10_sum')
    train, test = merge_max(train, test, ['A28'], 'A10', 'A28_A10_max')
    train, test = merge_min(train, test, ['A28'], 'A10', 'A28_A10_min')
    train, test = merge_std(train, test, ['A28'], 'A10', 'A28_A10_std')

    train, test = merge_median(train, test, ['A28'], 'A12', 'A28_A12_median')
    train, test = merge_mean(train, test, ['A28'], 'A12', 'A28_A12_mean')
    train, test = merge_sum(train, test, ['A28'], 'A12', 'A28_A12_sum')
    train, test = merge_max(train, test, ['A28'], 'A12', 'A28_A12_max')
    train, test = merge_min(train, test, ['A28'], 'A12', 'A28_A12_min')
    train, test = merge_std(train, test, ['A28'], 'A12', 'A28_A12_std')

    train, test = merge_median(train, test, ['A28'], 'A15', 'A28_A15_median')
    train, test = merge_mean(train, test, ['A28'], 'A15', 'A28_A15_mean')
    train, test = merge_sum(train, test, ['A28'], 'A15', 'A28_A15_sum')
    train, test = merge_max(train, test, ['A28'], 'A15', 'A28_A15_max')
    train, test = merge_min(train, test, ['A28'], 'A15', 'A28_A15_min')
    train, test = merge_std(train, test, ['A28'], 'A15', 'A28_A15_std')

    train, test = merge_median(train, test, ['A28'], 'A17', 'A28_A17_median')
    train, test = merge_mean(train, test, ['A28'], 'A17', 'A28_A17_mean')
    train, test = merge_sum(train, test, ['A28'], 'A17', 'A28_A17_sum')
    train, test = merge_max(train, test, ['A28'], 'A17', 'A28_A17_max')
    train, test = merge_min(train, test, ['A28'], 'A17', 'A28_A17_min')
    train, test = merge_std(train, test, ['A28'], 'A17', 'A28_A17_std')

    train, test = merge_median(train, test, ['A28'], 'A21', 'A28_A21_median')
    train, test = merge_mean(train, test, ['A28'], 'A21', 'A28_A21_mean')
    train, test = merge_sum(train, test, ['A28'], 'A21', 'A28_A21_sum')
    train, test = merge_max(train, test, ['A28'], 'A21', 'A28_A21_max')
    train, test = merge_min(train, test, ['A28'], 'A21', 'A28_A21_min')
    train, test = merge_std(train, test, ['A28'], 'A21', 'A28_A21_std')

    train, test = merge_median(train, test, ['A28'], 'A22', 'A28_A22_median')
    train, test = merge_mean(train, test, ['A28'], 'A22', 'A28_A22_mean')
    train, test = merge_sum(train, test, ['A28'], 'A22', 'A28_A22_sum')
    train, test = merge_max(train, test, ['A28'], 'A22', 'A28_A22_max')
    train, test = merge_min(train, test, ['A28'], 'A22', 'A28_A22_min')
    train, test = merge_std(train, test, ['A28'], 'A22', 'A28_A22_std')

    train, test = merge_count(train, test, ['A28'], 'A25', 'A28_A25_count')
    train, test = merge_nunique(train, test, ['A28'], 'A25', 'A28_A25_nunique')

    train, test = merge_median(train, test, ['A28'], 'A27', 'A28_A27_median')
    train, test = merge_mean(train, test, ['A28'], 'A27', 'A28_A27_mean')
    train, test = merge_sum(train, test, ['A28'], 'A27', 'A28_A27_sum')
    train, test = merge_max(train, test, ['A28'], 'A27', 'A28_A27_max')
    train, test = merge_min(train, test, ['A28'], 'A27', 'A28_A27_min')
    train, test = merge_std(train, test, ['A28'], 'A27', 'A28_A27_std')

    train, test = merge_median(train, test, ['A28'], 'B1', 'A28_B1_median')
    train, test = merge_mean(train, test, ['A28'], 'B1', 'A28_B1_mean')
    train, test = merge_sum(train, test, ['A28'], 'B1', 'A28_B1_sum')
    train, test = merge_max(train, test, ['A28'], 'B1', 'A28_B1_max')
    train, test = merge_min(train, test, ['A28'], 'B1', 'A28_B1_min')
    train, test = merge_std(train, test, ['A28'], 'B1', 'A28_B1_std')

    train, test = merge_count(train, test, ['A28'], 'B4', 'A28_B4_count')
    train, test = merge_nunique(train, test, ['A28'], 'B4', 'A28_B4_nunique')

    train, test = merge_count(train, test, ['A28'], 'B5', 'A28_B5_count')
    train, test = merge_nunique(train, test, ['A28'], 'B5', 'A28_B5_nunique')

    train, test = merge_median(train, test, ['A28'], 'B6', 'A28_B6_median')
    train, test = merge_mean(train, test, ['A28'], 'B6', 'A28_B6_mean')
    train, test = merge_sum(train, test, ['A28'], 'B6', 'A28_B6_sum')
    train, test = merge_max(train, test, ['A28'], 'B6', 'A28_B6_max')
    train, test = merge_min(train, test, ['A28'], 'B6', 'A28_B6_min')
    train, test = merge_std(train, test, ['A28'], 'B6', 'A28_B6_std')

    train, test = merge_count(train, test, ['A28'], 'B7', 'A28_B7_count')
    train, test = merge_nunique(train, test, ['A28'], 'B7', 'A28_B7_nunique')

    train, test = merge_median(train, test, ['A28'], 'B8', 'A28_B8_median')
    train, test = merge_mean(train, test, ['A28'], 'B8', 'A28_B8_mean')
    train, test = merge_sum(train, test, ['A28'], 'B8', 'A28_B8_sum')
    train, test = merge_max(train, test, ['A28'], 'B8', 'A28_B8_max')
    train, test = merge_min(train, test, ['A28'], 'B8', 'A28_B8_min')
    train, test = merge_std(train, test, ['A28'], 'B8', 'A28_B8_std')

    train, test = merge_count(train, test, ['A28'], 'B9', 'A28_B9_count')
    train, test = merge_nunique(train, test, ['A28'], 'B9', 'A28_B9_nunique')

    train, test = merge_count(train, test, ['A28'], 'B10', 'A28_B10_count')
    train, test = merge_nunique(train, test, ['A28'], 'B10', 'A28_B10_nunique')

    train, test = merge_count(train, test, ['A28'], 'B11', 'A28_B11_count')
    train, test = merge_nunique(train, test, ['A28'], 'B11', 'A28_B11_nunique')

    train, test = merge_median(train, test, ['A28'], 'B14', 'A28_B14_median')
    train, test = merge_mean(train, test, ['A28'], 'B14', 'A28_B14_mean')
    train, test = merge_sum(train, test, ['A28'], 'B14', 'A28_B14_sum')
    train, test = merge_max(train, test, ['A28'], 'B14', 'A28_B14_max')
    train, test = merge_min(train, test, ['A28'], 'B14', 'A28_B14_min')
    train, test = merge_std(train, test, ['A28'], 'B14', 'A28_B14_std')

    # ------------------------- B4 -----------------------#
    train, test = merge_count(train, test, ['B4'], 'A5', 'B4_A5_count')
    train, test = merge_nunique(train, test, ['B4'], 'A5', 'B4_A5_nunique')

    train, test = merge_median(train, test, ['B4'], 'A6', 'B4_A6_median')
    train, test = merge_mean(train, test, ['B4'], 'A6', 'B4_A6_mean')
    train, test = merge_sum(train, test, ['B4'], 'A6', 'B4_A6_sum')
    train, test = merge_max(train, test, ['B4'], 'A6', 'B4_A6_max')
    train, test = merge_min(train, test, ['B4'], 'A6', 'B4_A6_min')
    train, test = merge_std(train, test, ['B4'], 'A6', 'B4_A6_std')

    train, test = merge_median(train, test, ['B4'], 'A10', 'B4_A10_median')
    train, test = merge_mean(train, test, ['B4'], 'A10', 'B4_A10_mean')
    train, test = merge_sum(train, test, ['B4'], 'A10', 'B4_A10_sum')
    train, test = merge_max(train, test, ['B4'], 'A10', 'B4_A10_max')
    train, test = merge_min(train, test, ['B4'], 'A10', 'B4_A10_min')
    train, test = merge_std(train, test, ['B4'], 'A10', 'B4_A10_std')

    train, test = merge_median(train, test, ['B4'], 'A12', 'B4_A12_median')
    train, test = merge_mean(train, test, ['B4'], 'A12', 'B4_A12_mean')
    train, test = merge_sum(train, test, ['B4'], 'A12', 'B4_A12_sum')
    train, test = merge_max(train, test, ['B4'], 'A12', 'B4_A12_max')
    train, test = merge_min(train, test, ['B4'], 'A12', 'B4_A12_min')
    train, test = merge_std(train, test, ['B4'], 'A12', 'B4_A12_std')

    train, test = merge_median(train, test, ['B4'], 'A15', 'B4_A15_median')
    train, test = merge_mean(train, test, ['B4'], 'A15', 'B4_A15_mean')
    train, test = merge_sum(train, test, ['B4'], 'A15', 'B4_A15_sum')
    train, test = merge_max(train, test, ['B4'], 'A15', 'B4_A15_max')
    train, test = merge_min(train, test, ['B4'], 'A15', 'B4_A15_min')
    train, test = merge_std(train, test, ['B4'], 'A15', 'B4_A15_std')

    train, test = merge_median(train, test, ['B4'], 'A17', 'B4_A17_median')
    train, test = merge_mean(train, test, ['B4'], 'A17', 'B4_A17_mean')
    train, test = merge_sum(train, test, ['B4'], 'A17', 'B4_A17_sum')
    train, test = merge_max(train, test, ['B4'], 'A17', 'B4_A17_max')
    train, test = merge_min(train, test, ['B4'], 'A17', 'B4_A17_min')
    train, test = merge_std(train, test, ['B4'], 'A17', 'B4_A17_std')

    train, test = merge_median(train, test, ['B4'], 'A21', 'B4_A21_median')
    train, test = merge_mean(train, test, ['B4'], 'A21', 'B4_A21_mean')
    train, test = merge_sum(train, test, ['B4'], 'A21', 'B4_A21_sum')
    train, test = merge_max(train, test, ['B4'], 'A21', 'B4_A21_max')
    train, test = merge_min(train, test, ['B4'], 'A21', 'B4_A21_min')
    train, test = merge_std(train, test, ['B4'], 'A21', 'B4_A21_std')

    train, test = merge_median(train, test, ['B4'], 'A22', 'B4_A22_median')
    train, test = merge_mean(train, test, ['B4'], 'A22', 'B4_A22_mean')
    train, test = merge_sum(train, test, ['B4'], 'A22', 'B4_A22_sum')
    train, test = merge_max(train, test, ['B4'], 'A22', 'B4_A22_max')
    train, test = merge_min(train, test, ['B4'], 'A22', 'B4_A22_min')
    train, test = merge_std(train, test, ['B4'], 'A22', 'B4_A22_std')

    train, test = merge_count(train, test, ['B4'], 'A25', 'B4_A25_count')
    train, test = merge_nunique(train, test, ['B4'], 'A25', 'B4_A25_nunique')

    train, test = merge_median(train, test, ['B4'], 'A27', 'B4_A27_median')
    train, test = merge_mean(train, test, ['B4'], 'A27', 'B4_A27_mean')
    train, test = merge_sum(train, test, ['B4'], 'A27', 'B4_A27_sum')
    train, test = merge_max(train, test, ['B4'], 'A27', 'B4_A27_max')
    train, test = merge_min(train, test, ['B4'], 'A27', 'B4_A27_min')
    train, test = merge_std(train, test, ['B4'], 'A27', 'B4_A27_std')

    train, test = merge_median(train, test, ['B4'], 'B1', 'B4_B1_median')
    train, test = merge_mean(train, test, ['B4'], 'B1', 'B4_B1_mean')
    train, test = merge_sum(train, test, ['B4'], 'B1', 'B4_B1_sum')
    train, test = merge_max(train, test, ['B4'], 'B1', 'B4_B1_max')
    train, test = merge_min(train, test, ['B4'], 'B1', 'B4_B1_min')
    train, test = merge_std(train, test, ['B4'], 'B1', 'B4_B1_std')

    train, test = merge_count(train, test, ['B4'], 'B5', 'B4_B5_count')
    train, test = merge_nunique(train, test, ['B4'], 'B5', 'B4_B5_nunique')

    train, test = merge_median(train, test, ['B4'], 'B6', 'B4_B6_median')
    train, test = merge_mean(train, test, ['B4'], 'B6', 'B4_B6_mean')
    train, test = merge_sum(train, test, ['B4'], 'B6', 'B4_B6_sum')
    train, test = merge_max(train, test, ['B4'], 'B6', 'B4_B6_max')
    train, test = merge_min(train, test, ['B4'], 'B6', 'B4_B6_min')
    train, test = merge_std(train, test, ['B4'], 'B6', 'B4_B6_std')

    train, test = merge_count(train, test, ['B4'], 'B7', 'B4_B7_count')
    train, test = merge_nunique(train, test, ['B4'], 'B7', 'B4_B7_nunique')

    train, test = merge_median(train, test, ['B4'], 'B8', 'B4_B8_median')
    train, test = merge_mean(train, test, ['B4'], 'B8', 'B4_B8_mean')
    train, test = merge_sum(train, test, ['B4'], 'B8', 'B4_B8_sum')
    train, test = merge_max(train, test, ['B4'], 'B8', 'B4_B8_max')
    train, test = merge_min(train, test, ['B4'], 'B8', 'B4_B8_min')
    train, test = merge_std(train, test, ['B4'], 'B8', 'B4_B8_std')

    train, test = merge_count(train, test, ['B4'], 'B9', 'B4_B9_count')
    train, test = merge_nunique(train, test, ['B4'], 'B9', 'B4_B9_nunique')

    train, test = merge_count(train, test, ['B4'], 'B10', 'B4_B10_count')
    train, test = merge_nunique(train, test, ['B4'], 'B10', 'B4_B10_nunique')

    train, test = merge_count(train, test, ['B4'], 'B11', 'B4_B11_count')
    train, test = merge_nunique(train, test, ['B4'], 'B11', 'B4_B11_nunique')

    train, test = merge_median(train, test, ['B4'], 'B14', 'B4_B14_median')
    train, test = merge_mean(train, test, ['B4'], 'B14', 'B4_B14_mean')
    train, test = merge_sum(train, test, ['B4'], 'B14', 'B4_B14_sum')
    train, test = merge_max(train, test, ['B4'], 'B14', 'B4_B14_max')
    train, test = merge_min(train, test, ['B4'], 'B14', 'B4_B14_min')
    train, test = merge_std(train, test, ['B4'], 'B14', 'B4_B14_std')

    # ------------------------- B5 -----------------------#
    train, test = merge_count(train, test, ['B5'], 'A5', 'B5_A5_count')
    train, test = merge_nunique(train, test, ['B5'], 'A5', 'B5_A5_nunique')

    train, test = merge_median(train, test, ['B5'], 'A6', 'B5_A6_median')
    train, test = merge_mean(train, test, ['B5'], 'A6', 'B5_A6_mean')
    train, test = merge_sum(train, test, ['B5'], 'A6', 'B5_A6_sum')
    train, test = merge_max(train, test, ['B5'], 'A6', 'B5_A6_max')
    train, test = merge_min(train, test, ['B5'], 'A6', 'B5_A6_min')
    train, test = merge_std(train, test, ['B5'], 'A6', 'B5_A6_std')

    train, test = merge_median(train, test, ['B5'], 'A10', 'B5_A10_median')
    train, test = merge_mean(train, test, ['B5'], 'A10', 'B5_A10_mean')
    train, test = merge_sum(train, test, ['B5'], 'A10', 'B5_A10_sum')
    train, test = merge_max(train, test, ['B5'], 'A10', 'B5_A10_max')
    train, test = merge_min(train, test, ['B5'], 'A10', 'B5_A10_min')
    train, test = merge_std(train, test, ['B5'], 'A10', 'B5_A10_std')

    train, test = merge_median(train, test, ['B5'], 'A12', 'B5_A12_median')
    train, test = merge_mean(train, test, ['B5'], 'A12', 'B5_A12_mean')
    train, test = merge_sum(train, test, ['B5'], 'A12', 'B5_A12_sum')
    train, test = merge_max(train, test, ['B5'], 'A12', 'B5_A12_max')
    train, test = merge_min(train, test, ['B5'], 'A12', 'B5_A12_min')
    train, test = merge_std(train, test, ['B5'], 'A12', 'B5_A12_std')

    train, test = merge_median(train, test, ['B5'], 'A15', 'B5_A15_median')
    train, test = merge_mean(train, test, ['B5'], 'A15', 'B5_A15_mean')
    train, test = merge_sum(train, test, ['B5'], 'A15', 'B5_A15_sum')
    train, test = merge_max(train, test, ['B5'], 'A15', 'B5_A15_max')
    train, test = merge_min(train, test, ['B5'], 'A15', 'B5_A15_min')
    train, test = merge_std(train, test, ['B5'], 'A15', 'B5_A15_std')

    train, test = merge_median(train, test, ['B5'], 'A17', 'B5_A17_median')
    train, test = merge_mean(train, test, ['B5'], 'A17', 'B5_A17_mean')
    train, test = merge_sum(train, test, ['B5'], 'A17', 'B5_A17_sum')
    train, test = merge_max(train, test, ['B5'], 'A17', 'B5_A17_max')
    train, test = merge_min(train, test, ['B5'], 'A17', 'B5_A17_min')
    train, test = merge_std(train, test, ['B5'], 'A17', 'B5_A17_std')

    train, test = merge_median(train, test, ['B5'], 'A21', 'B5_A21_median')
    train, test = merge_mean(train, test, ['B5'], 'A21', 'B5_A21_mean')
    train, test = merge_sum(train, test, ['B5'], 'A21', 'B5_A21_sum')
    train, test = merge_max(train, test, ['B5'], 'A21', 'B5_A21_max')
    train, test = merge_min(train, test, ['B5'], 'A21', 'B5_A21_min')
    train, test = merge_std(train, test, ['B5'], 'A21', 'B5_A21_std')

    train, test = merge_median(train, test, ['B5'], 'A22', 'B5_A22_median')
    train, test = merge_mean(train, test, ['B5'], 'A22', 'B5_A22_mean')
    train, test = merge_sum(train, test, ['B5'], 'A22', 'B5_A22_sum')
    train, test = merge_max(train, test, ['B5'], 'A22', 'B5_A22_max')
    train, test = merge_min(train, test, ['B5'], 'A22', 'B5_A22_min')
    train, test = merge_std(train, test, ['B5'], 'A22', 'B5_A22_std')

    train, test = merge_count(train, test, ['B5'], 'A25', 'B5_A25_count')
    train, test = merge_nunique(train, test, ['B5'], 'A25', 'B5_A25_nunique')

    train, test = merge_median(train, test, ['B5'], 'A27', 'B5_A27_median')
    train, test = merge_mean(train, test, ['B5'], 'A27', 'B5_A27_mean')
    train, test = merge_sum(train, test, ['B5'], 'A27', 'B5_A27_sum')
    train, test = merge_max(train, test, ['B5'], 'A27', 'B5_A27_max')
    train, test = merge_min(train, test, ['B5'], 'A27', 'B5_A27_min')
    train, test = merge_std(train, test, ['B5'], 'A27', 'B5_A27_std')

    train, test = merge_median(train, test, ['B5'], 'B1', 'B5_B1_median')
    train, test = merge_mean(train, test, ['B5'], 'B1', 'B5_B1_mean')
    train, test = merge_sum(train, test, ['B5'], 'B1', 'B5_B1_sum')
    train, test = merge_max(train, test, ['B5'], 'B1', 'B5_B1_max')
    train, test = merge_min(train, test, ['B5'], 'B1', 'B5_B1_min')
    train, test = merge_std(train, test, ['B5'], 'B1', 'B5_B1_std')

    train, test = merge_count(train, test, ['B5'], 'B4', 'B5_B4_count')
    train, test = merge_nunique(train, test, ['B5'], 'B4', 'B5_B4_nunique')

    train, test = merge_median(train, test, ['B5'], 'B6', 'B5_B6_median')
    train, test = merge_mean(train, test, ['B5'], 'B6', 'B5_B6_mean')
    train, test = merge_sum(train, test, ['B5'], 'B6', 'B5_B6_sum')
    train, test = merge_max(train, test, ['B5'], 'B6', 'B5_B6_max')
    train, test = merge_min(train, test, ['B5'], 'B6', 'B5_B6_min')
    train, test = merge_std(train, test, ['B5'], 'B6', 'B5_B6_std')

    train, test = merge_count(train, test, ['B5'], 'B7', 'B5_B7_count')
    train, test = merge_nunique(train, test, ['B5'], 'B7', 'B5_B7_nunique')

    train, test = merge_median(train, test, ['B5'], 'B8', 'B5_B8_median')
    train, test = merge_mean(train, test, ['B5'], 'B8', 'B5_B8_mean')
    train, test = merge_sum(train, test, ['B5'], 'B8', 'B5_B8_sum')
    train, test = merge_max(train, test, ['B5'], 'B8', 'B5_B8_max')
    train, test = merge_min(train, test, ['B5'], 'B8', 'B5_B8_min')
    train, test = merge_std(train, test, ['B5'], 'B8', 'B5_B8_std')

    train, test = merge_count(train, test, ['B5'], 'B9', 'B5_B9_count')
    train, test = merge_nunique(train, test, ['B5'], 'B9', 'B5_B9_nunique')

    train, test = merge_count(train, test, ['B5'], 'B10', 'B5_B10_count')
    train, test = merge_nunique(train, test, ['B5'], 'B10', 'B5_B10_nunique')

    train, test = merge_count(train, test, ['B5'], 'B11', 'B5_B11_count')
    train, test = merge_nunique(train, test, ['B5'], 'B11', 'B5_B11_nunique')

    train, test = merge_median(train, test, ['B5'], 'B14', 'B5_B14_median')
    train, test = merge_mean(train, test, ['B5'], 'B14', 'B5_B14_mean')
    train, test = merge_sum(train, test, ['B5'], 'B14', 'B5_B14_sum')
    train, test = merge_max(train, test, ['B5'], 'B14', 'B5_B14_max')
    train, test = merge_min(train, test, ['B5'], 'B14', 'B5_B14_min')
    train, test = merge_std(train, test, ['B5'], 'B14', 'B5_B14_std')

    # ------------------------- B7 -----------------------#
    train, test = merge_count(train, test, ['B7'], 'A5', 'B7_A5_count')
    train, test = merge_nunique(train, test, ['B7'], 'A5', 'B7_A5_nunique')

    train, test = merge_median(train, test, ['B7'], 'A6', 'B7_A6_median')
    train, test = merge_mean(train, test, ['B7'], 'A6', 'B7_A6_mean')
    train, test = merge_sum(train, test, ['B7'], 'A6', 'B7_A6_sum')
    train, test = merge_max(train, test, ['B7'], 'A6', 'B7_A6_max')
    train, test = merge_min(train, test, ['B7'], 'A6', 'B7_A6_min')
    train, test = merge_std(train, test, ['B7'], 'A6', 'B7_A6_std')

    train, test = merge_median(train, test, ['B7'], 'A10', 'B7_A10_median')
    train, test = merge_mean(train, test, ['B7'], 'A10', 'B7_A10_mean')
    train, test = merge_sum(train, test, ['B7'], 'A10', 'B7_A10_sum')
    train, test = merge_max(train, test, ['B7'], 'A10', 'B7_A10_max')
    train, test = merge_min(train, test, ['B7'], 'A10', 'B7_A10_min')
    train, test = merge_std(train, test, ['B7'], 'A10', 'B7_A10_std')

    train, test = merge_median(train, test, ['B7'], 'A12', 'B7_A12_median')
    train, test = merge_mean(train, test, ['B7'], 'A12', 'B7_A12_mean')
    train, test = merge_sum(train, test, ['B7'], 'A12', 'B7_A12_sum')
    train, test = merge_max(train, test, ['B7'], 'A12', 'B7_A12_max')
    train, test = merge_min(train, test, ['B7'], 'A12', 'B7_A12_min')
    train, test = merge_std(train, test, ['B7'], 'A12', 'B7_A12_std')

    train, test = merge_median(train, test, ['B7'], 'A15', 'B7_A15_median')
    train, test = merge_mean(train, test, ['B7'], 'A15', 'B7_A15_mean')
    train, test = merge_sum(train, test, ['B7'], 'A15', 'B7_A15_sum')
    train, test = merge_max(train, test, ['B7'], 'A15', 'B7_A15_max')
    train, test = merge_min(train, test, ['B7'], 'A15', 'B7_A15_min')
    train, test = merge_std(train, test, ['B7'], 'A15', 'B7_A15_std')

    train, test = merge_median(train, test, ['B7'], 'A17', 'B7_A17_median')
    train, test = merge_mean(train, test, ['B7'], 'A17', 'B7_A17_mean')
    train, test = merge_sum(train, test, ['B7'], 'A17', 'B7_A17_sum')
    train, test = merge_max(train, test, ['B7'], 'A17', 'B7_A17_max')
    train, test = merge_min(train, test, ['B7'], 'A17', 'B7_A17_min')
    train, test = merge_std(train, test, ['B7'], 'A17', 'B7_A17_std')

    train, test = merge_median(train, test, ['B7'], 'A21', 'B7_A21_median')
    train, test = merge_mean(train, test, ['B7'], 'A21', 'B7_A21_mean')
    train, test = merge_sum(train, test, ['B7'], 'A21', 'B7_A21_sum')
    train, test = merge_max(train, test, ['B7'], 'A21', 'B7_A21_max')
    train, test = merge_min(train, test, ['B7'], 'A21', 'B7_A21_min')
    train, test = merge_std(train, test, ['B7'], 'A21', 'B7_A21_std')

    train, test = merge_median(train, test, ['B7'], 'A22', 'B7_A22_median')
    train, test = merge_mean(train, test, ['B7'], 'A22', 'B7_A22_mean')
    train, test = merge_sum(train, test, ['B7'], 'A22', 'B7_A22_sum')
    train, test = merge_max(train, test, ['B7'], 'A22', 'B7_A22_max')
    train, test = merge_min(train, test, ['B7'], 'A22', 'B7_A22_min')
    train, test = merge_std(train, test, ['B7'], 'A22', 'B7_A22_std')

    train, test = merge_count(train, test, ['B7'], 'A25', 'B7_A25_count')
    train, test = merge_nunique(train, test, ['B7'], 'A25', 'B7_A25_nunique')

    train, test = merge_median(train, test, ['B7'], 'A27', 'B7_A27_median')
    train, test = merge_mean(train, test, ['B7'], 'A27', 'B7_A27_mean')
    train, test = merge_sum(train, test, ['B7'], 'A27', 'B7_A27_sum')
    train, test = merge_max(train, test, ['B7'], 'A27', 'B7_A27_max')
    train, test = merge_min(train, test, ['B7'], 'A27', 'B7_A27_min')
    train, test = merge_std(train, test, ['B7'], 'A27', 'B7_A27_std')

    train, test = merge_median(train, test, ['B7'], 'B1', 'B7_B1_median')
    train, test = merge_mean(train, test, ['B7'], 'B1', 'B7_B1_mean')
    train, test = merge_sum(train, test, ['B7'], 'B1', 'B7_B1_sum')
    train, test = merge_max(train, test, ['B7'], 'B1', 'B7_B1_max')
    train, test = merge_min(train, test, ['B7'], 'B1', 'B7_B1_min')
    train, test = merge_std(train, test, ['B7'], 'B1', 'B7_B1_std')

    train, test = merge_count(train, test, ['B7'], 'B4', 'B7_B4_count')
    train, test = merge_nunique(train, test, ['B7'], 'B4', 'B7_B4_nunique')

    train, test = merge_count(train, test, ['B7'], 'B5', 'B7_B5_count')
    train, test = merge_nunique(train, test, ['B7'], 'B5', 'B7_B5_nunique')

    train, test = merge_median(train, test, ['B7'], 'B6', 'B7_B6_median')
    train, test = merge_mean(train, test, ['B7'], 'B6', 'B7_B6_mean')
    train, test = merge_sum(train, test, ['B7'], 'B6', 'B7_B6_sum')
    train, test = merge_max(train, test, ['B7'], 'B6', 'B7_B6_max')
    train, test = merge_min(train, test, ['B7'], 'B6', 'B7_B6_min')
    train, test = merge_std(train, test, ['B7'], 'B6', 'B7_B6_std')

    train, test = merge_median(train, test, ['B7'], 'B8', 'B7_B8_median')
    train, test = merge_mean(train, test, ['B7'], 'B8', 'B7_B8_mean')
    train, test = merge_sum(train, test, ['B7'], 'B8', 'B7_B8_sum')
    train, test = merge_max(train, test, ['B7'], 'B8', 'B7_B8_max')
    train, test = merge_min(train, test, ['B7'], 'B8', 'B7_B8_min')
    train, test = merge_std(train, test, ['B7'], 'B8', 'B7_B8_std')

    train, test = merge_count(train, test, ['B7'], 'B9', 'B7_B9_count')
    train, test = merge_nunique(train, test, ['B7'], 'B9', 'B7_B9_nunique')

    train, test = merge_count(train, test, ['B7'], 'B10', 'B7_B10_count')
    train, test = merge_nunique(train, test, ['B7'], 'B10', 'B7_B10_nunique')

    train, test = merge_count(train, test, ['B7'], 'B11', 'B7_B11_count')
    train, test = merge_nunique(train, test, ['B7'], 'B11', 'B7_B11_nunique')

    train, test = merge_median(train, test, ['B7'], 'B14', 'B7_B14_median')
    train, test = merge_mean(train, test, ['B7'], 'B14', 'B7_B14_mean')
    train, test = merge_sum(train, test, ['B7'], 'B14', 'B7_B14_sum')
    train, test = merge_max(train, test, ['B7'], 'B14', 'B7_B14_max')
    train, test = merge_min(train, test, ['B7'], 'B14', 'B7_B14_min')
    train, test = merge_std(train, test, ['B7'], 'B14', 'B7_B14_std')

    # ------------------------- B9 -----------------------#
    train, test = merge_count(train, test, ['B9'], 'A5', 'B9_A5_count')
    train, test = merge_nunique(train, test, ['B9'], 'A5', 'B9_A5_nunique')

    train, test = merge_median(train, test, ['B9'], 'A6', 'B9_A6_median')
    train, test = merge_mean(train, test, ['B9'], 'A6', 'B9_A6_mean')
    train, test = merge_sum(train, test, ['B9'], 'A6', 'B9_A6_sum')
    train, test = merge_max(train, test, ['B9'], 'A6', 'B9_A6_max')
    train, test = merge_min(train, test, ['B9'], 'A6', 'B9_A6_min')
    train, test = merge_std(train, test, ['B9'], 'A6', 'B9_A6_std')

    train, test = merge_median(train, test, ['B9'], 'A10', 'B9_A10_median')
    train, test = merge_mean(train, test, ['B9'], 'A10', 'B9_A10_mean')
    train, test = merge_sum(train, test, ['B9'], 'A10', 'B9_A10_sum')
    train, test = merge_max(train, test, ['B9'], 'A10', 'B9_A10_max')
    train, test = merge_min(train, test, ['B9'], 'A10', 'B9_A10_min')
    train, test = merge_std(train, test, ['B9'], 'A10', 'B9_A10_std')

    train, test = merge_median(train, test, ['B9'], 'A12', 'B9_A12_median')
    train, test = merge_mean(train, test, ['B9'], 'A12', 'B9_A12_mean')
    train, test = merge_sum(train, test, ['B9'], 'A12', 'B9_A12_sum')
    train, test = merge_max(train, test, ['B9'], 'A12', 'B9_A12_max')
    train, test = merge_min(train, test, ['B9'], 'A12', 'B9_A12_min')
    train, test = merge_std(train, test, ['B9'], 'A12', 'B9_A12_std')

    train, test = merge_median(train, test, ['B9'], 'A15', 'B9_A15_median')
    train, test = merge_mean(train, test, ['B9'], 'A15', 'B9_A15_mean')
    train, test = merge_sum(train, test, ['B9'], 'A15', 'B9_A15_sum')
    train, test = merge_max(train, test, ['B9'], 'A15', 'B9_A15_max')
    train, test = merge_min(train, test, ['B9'], 'A15', 'B9_A15_min')
    train, test = merge_std(train, test, ['B9'], 'A15', 'B9_A15_std')

    train, test = merge_median(train, test, ['B9'], 'A17', 'B9_A17_median')
    train, test = merge_mean(train, test, ['B9'], 'A17', 'B9_A17_mean')
    train, test = merge_sum(train, test, ['B9'], 'A17', 'B9_A17_sum')
    train, test = merge_max(train, test, ['B9'], 'A17', 'B9_A17_max')
    train, test = merge_min(train, test, ['B9'], 'A17', 'B9_A17_min')
    train, test = merge_std(train, test, ['B9'], 'A17', 'B9_A17_std')

    train, test = merge_median(train, test, ['B9'], 'A21', 'B9_A21_median')
    train, test = merge_mean(train, test, ['B9'], 'A21', 'B9_A21_mean')
    train, test = merge_sum(train, test, ['B9'], 'A21', 'B9_A21_sum')
    train, test = merge_max(train, test, ['B9'], 'A21', 'B9_A21_max')
    train, test = merge_min(train, test, ['B9'], 'A21', 'B9_A21_min')
    train, test = merge_std(train, test, ['B9'], 'A21', 'B9_A21_std')

    train, test = merge_median(train, test, ['B9'], 'A22', 'B9_A22_median')
    train, test = merge_mean(train, test, ['B9'], 'A22', 'B9_A22_mean')
    train, test = merge_sum(train, test, ['B9'], 'A22', 'B9_A22_sum')
    train, test = merge_max(train, test, ['B9'], 'A22', 'B9_A22_max')
    train, test = merge_min(train, test, ['B9'], 'A22', 'B9_A22_min')
    train, test = merge_std(train, test, ['B9'], 'A22', 'B9_A22_std')

    train, test = merge_count(train, test, ['B9'], 'A25', 'B9_A25_count')
    train, test = merge_nunique(train, test, ['B9'], 'A25', 'B9_A25_nunique')

    train, test = merge_median(train, test, ['B9'], 'A27', 'B9_A27_median')
    train, test = merge_mean(train, test, ['B9'], 'A27', 'B9_A27_mean')
    train, test = merge_sum(train, test, ['B9'], 'A27', 'B9_A27_sum')
    train, test = merge_max(train, test, ['B9'], 'A27', 'B9_A27_max')
    train, test = merge_min(train, test, ['B9'], 'A27', 'B9_A27_min')
    train, test = merge_std(train, test, ['B9'], 'A27', 'B9_A27_std')

    train, test = merge_median(train, test, ['B9'], 'B1', 'B9_B1_median')
    train, test = merge_mean(train, test, ['B9'], 'B1', 'B9_B1_mean')
    train, test = merge_sum(train, test, ['B9'], 'B1', 'B9_B1_sum')
    train, test = merge_max(train, test, ['B9'], 'B1', 'B9_B1_max')
    train, test = merge_min(train, test, ['B9'], 'B1', 'B9_B1_min')
    train, test = merge_std(train, test, ['B9'], 'B1', 'B9_B1_std')

    train, test = merge_count(train, test, ['B9'], 'B4', 'B9_B4_count')
    train, test = merge_nunique(train, test, ['B9'], 'B4', 'B9_B4_nunique')

    train, test = merge_count(train, test, ['B9'], 'B5', 'B9_B5_count')
    train, test = merge_nunique(train, test, ['B9'], 'B5', 'B9_B5_nunique')

    train, test = merge_median(train, test, ['B9'], 'B6', 'B9_B6_median')
    train, test = merge_mean(train, test, ['B9'], 'B6', 'B9_B6_mean')
    train, test = merge_sum(train, test, ['B9'], 'B6', 'B9_B6_sum')
    train, test = merge_max(train, test, ['B9'], 'B6', 'B9_B6_max')
    train, test = merge_min(train, test, ['B9'], 'B6', 'B9_B6_min')
    train, test = merge_std(train, test, ['B9'], 'B6', 'B9_B6_std')

    train, test = merge_count(train, test, ['B9'], 'B7', 'B9_B7_count')
    train, test = merge_nunique(train, test, ['B9'], 'B7', 'B9_B7_nunique')

    train, test = merge_median(train, test, ['B9'], 'B8', 'B9_B8_median')
    train, test = merge_mean(train, test, ['B9'], 'B8', 'B9_B8_mean')
    train, test = merge_sum(train, test, ['B9'], 'B8', 'B9_B8_sum')
    train, test = merge_max(train, test, ['B9'], 'B8', 'B9_B8_max')
    train, test = merge_min(train, test, ['B9'], 'B8', 'B9_B8_min')
    train, test = merge_std(train, test, ['B9'], 'B8', 'B9_B8_std')

    train, test = merge_count(train, test, ['B9'], 'B10', 'B9_B10_count')
    train, test = merge_nunique(train, test, ['B9'], 'B10', 'B9_B10_nunique')

    train, test = merge_count(train, test, ['B9'], 'B11', 'B9_B11_count')
    train, test = merge_nunique(train, test, ['B9'], 'B11', 'B9_B11_nunique')

    train, test = merge_median(train, test, ['B9'], 'B14', 'B9_B14_median')
    train, test = merge_mean(train, test, ['B9'], 'B14', 'B9_B14_mean')
    train, test = merge_sum(train, test, ['B9'], 'B14', 'B9_B14_sum')
    train, test = merge_max(train, test, ['B9'], 'B14', 'B9_B14_max')
    train, test = merge_min(train, test, ['B9'], 'B14', 'B9_B14_min')
    train, test = merge_std(train, test, ['B9'], 'B14', 'B9_B14_std')

    # ------------------------- B10 -----------------------#
    train, test = merge_count(train, test, ['B10'], 'A5', 'B10_A5_count')
    train, test = merge_nunique(train, test, ['B10'], 'A5', 'B10_A5_nunique')

    train, test = merge_median(train, test, ['B10'], 'A6', 'B10_A6_median')
    train, test = merge_mean(train, test, ['B10'], 'A6', 'B10_A6_mean')
    train, test = merge_sum(train, test, ['B10'], 'A6', 'B10_A6_sum')
    train, test = merge_max(train, test, ['B10'], 'A6', 'B10_A6_max')
    train, test = merge_min(train, test, ['B10'], 'A6', 'B10_A6_min')
    train, test = merge_std(train, test, ['B10'], 'A6', 'B10_A6_std')

    train, test = merge_median(train, test, ['B10'], 'A10', 'B10_A10_median')
    train, test = merge_mean(train, test, ['B10'], 'A10', 'B10_A10_mean')
    train, test = merge_sum(train, test, ['B10'], 'A10', 'B10_A10_sum')
    train, test = merge_max(train, test, ['B10'], 'A10', 'B10_A10_max')
    train, test = merge_min(train, test, ['B10'], 'A10', 'B10_A10_min')
    train, test = merge_std(train, test, ['B10'], 'A10', 'B10_A10_std')

    train, test = merge_median(train, test, ['B10'], 'A12', 'B10_A12_median')
    train, test = merge_mean(train, test, ['B10'], 'A12', 'B10_A12_mean')
    train, test = merge_sum(train, test, ['B10'], 'A12', 'B10_A12_sum')
    train, test = merge_max(train, test, ['B10'], 'A12', 'B10_A12_max')
    train, test = merge_min(train, test, ['B10'], 'A12', 'B10_A12_min')
    train, test = merge_std(train, test, ['B10'], 'A12', 'B10_A12_std')

    train, test = merge_median(train, test, ['B10'], 'A15', 'B10_A15_median')
    train, test = merge_mean(train, test, ['B10'], 'A15', 'B10_A15_mean')
    train, test = merge_sum(train, test, ['B10'], 'A15', 'B10_A15_sum')
    train, test = merge_max(train, test, ['B10'], 'A15', 'B10_A15_max')
    train, test = merge_min(train, test, ['B10'], 'A15', 'B10_A15_min')
    train, test = merge_std(train, test, ['B10'], 'A15', 'B10_A15_std')

    train, test = merge_median(train, test, ['B10'], 'A17', 'B10_A17_median')
    train, test = merge_mean(train, test, ['B10'], 'A17', 'B10_A17_mean')
    train, test = merge_sum(train, test, ['B10'], 'A17', 'B10_A17_sum')
    train, test = merge_max(train, test, ['B10'], 'A17', 'B10_A17_max')
    train, test = merge_min(train, test, ['B10'], 'A17', 'B10_A17_min')
    train, test = merge_std(train, test, ['B10'], 'A17', 'B10_A17_std')

    train, test = merge_median(train, test, ['B10'], 'A21', 'B10_A21_median')
    train, test = merge_mean(train, test, ['B10'], 'A21', 'B10_A21_mean')
    train, test = merge_sum(train, test, ['B10'], 'A21', 'B10_A21_sum')
    train, test = merge_max(train, test, ['B10'], 'A21', 'B10_A21_max')
    train, test = merge_min(train, test, ['B10'], 'A21', 'B10_A21_min')
    train, test = merge_std(train, test, ['B10'], 'A21', 'B10_A21_std')

    train, test = merge_median(train, test, ['B10'], 'A22', 'B10_A22_median')
    train, test = merge_mean(train, test, ['B10'], 'A22', 'B10_A22_mean')
    train, test = merge_sum(train, test, ['B10'], 'A22', 'B10_A22_sum')
    train, test = merge_max(train, test, ['B10'], 'A22', 'B10_A22_max')
    train, test = merge_min(train, test, ['B10'], 'A22', 'B10_A22_min')
    train, test = merge_std(train, test, ['B10'], 'A22', 'B10_A22_std')

    train, test = merge_count(train, test, ['B10'], 'A25', 'B10_A25_count')
    train, test = merge_nunique(train, test, ['B10'], 'A25', 'B10_A25_nunique')

    train, test = merge_median(train, test, ['B10'], 'A27', 'B10_A27_median')
    train, test = merge_mean(train, test, ['B10'], 'A27', 'B10_A27_mean')
    train, test = merge_sum(train, test, ['B10'], 'A27', 'B10_A27_sum')
    train, test = merge_max(train, test, ['B10'], 'A27', 'B10_A27_max')
    train, test = merge_min(train, test, ['B10'], 'A27', 'B10_A27_min')
    train, test = merge_std(train, test, ['B10'], 'A27', 'B10_A27_std')

    train, test = merge_median(train, test, ['B10'], 'B1', 'B10_B1_median')
    train, test = merge_mean(train, test, ['B10'], 'B1', 'B10_B1_mean')
    train, test = merge_sum(train, test, ['B10'], 'B1', 'B10_B1_sum')
    train, test = merge_max(train, test, ['B10'], 'B1', 'B10_B1_max')
    train, test = merge_min(train, test, ['B10'], 'B1', 'B10_B1_min')
    train, test = merge_std(train, test, ['B10'], 'B1', 'B10_B1_std')

    train, test = merge_count(train, test, ['B10'], 'B4', 'B10_B4_count')
    train, test = merge_nunique(train, test, ['B10'], 'B4', 'B10_B4_nunique')

    train, test = merge_count(train, test, ['B10'], 'B5', 'B10_B5_count')
    train, test = merge_nunique(train, test, ['B10'], 'B5', 'B10_B5_nunique')

    train, test = merge_median(train, test, ['B10'], 'B6', 'B10_B6_median')
    train, test = merge_mean(train, test, ['B10'], 'B6', 'B10_B6_mean')
    train, test = merge_sum(train, test, ['B10'], 'B6', 'B10_B6_sum')
    train, test = merge_max(train, test, ['B10'], 'B6', 'B10_B6_max')
    train, test = merge_min(train, test, ['B10'], 'B6', 'B10_B6_min')
    train, test = merge_std(train, test, ['B10'], 'B6', 'B10_B6_std')

    train, test = merge_count(train, test, ['B10'], 'B7', 'B10_B10_count')
    train, test = merge_nunique(train, test, ['B10'], 'B7', 'B10_B10_nunique')

    train, test = merge_median(train, test, ['B10'], 'B8', 'B10_B8_median')
    train, test = merge_mean(train, test, ['B10'], 'B8', 'B10_B8_mean')
    train, test = merge_sum(train, test, ['B10'], 'B8', 'B10_B8_sum')
    train, test = merge_max(train, test, ['B10'], 'B8', 'B10_B8_max')
    train, test = merge_min(train, test, ['B10'], 'B8', 'B10_B8_min')
    train, test = merge_std(train, test, ['B10'], 'B8', 'B10_B8_std')

    train, test = merge_count(train, test, ['B10'], 'B9', 'B10_B9_count')
    train, test = merge_nunique(train, test, ['B10'], 'B9', 'B10_B9_nunique')

    train, test = merge_count(train, test, ['B10'], 'B11', 'B10_B11_count')
    train, test = merge_nunique(train, test, ['B10'], 'B11', 'B10_B11_nunique')

    train, test = merge_median(train, test, ['B10'], 'B14', 'B10_B14_median')
    train, test = merge_mean(train, test, ['B10'], 'B14', 'B10_B14_mean')
    train, test = merge_sum(train, test, ['B10'], 'B14', 'B10_B14_sum')
    train, test = merge_max(train, test, ['B10'], 'B14', 'B10_B14_max')
    train, test = merge_min(train, test, ['B10'], 'B14', 'B10_B14_min')
    train, test = merge_std(train, test, ['B10'], 'B14', 'B10_B14_std')

    # ------------------------- B11 -----------------------#
    train, test = merge_count(train, test, ['B11'], 'A5', 'B11_A5_count')
    train, test = merge_nunique(train, test, ['B11'], 'A5', 'B11_A5_nunique')

    train, test = merge_median(train, test, ['B11'], 'A6', 'B11_A6_median')
    train, test = merge_mean(train, test, ['B11'], 'A6', 'B11_A6_mean')
    train, test = merge_sum(train, test, ['B11'], 'A6', 'B11_A6_sum')
    train, test = merge_max(train, test, ['B11'], 'A6', 'B11_A6_max')
    train, test = merge_min(train, test, ['B11'], 'A6', 'B11_A6_min')
    train, test = merge_std(train, test, ['B11'], 'A6', 'B11_A6_std')

    train, test = merge_median(train, test, ['B11'], 'A10', 'B11_A10_median')
    train, test = merge_mean(train, test, ['B11'], 'A10', 'B11_A10_mean')
    train, test = merge_sum(train, test, ['B11'], 'A10', 'B11_A10_sum')
    train, test = merge_max(train, test, ['B11'], 'A10', 'B11_A10_max')
    train, test = merge_min(train, test, ['B11'], 'A10', 'B11_A10_min')
    train, test = merge_std(train, test, ['B11'], 'A10', 'B11_A10_std')

    train, test = merge_median(train, test, ['B11'], 'A12', 'B11_A12_median')
    train, test = merge_mean(train, test, ['B11'], 'A12', 'B11_A12_mean')
    train, test = merge_sum(train, test, ['B11'], 'A12', 'B11_A12_sum')
    train, test = merge_max(train, test, ['B11'], 'A12', 'B11_A12_max')
    train, test = merge_min(train, test, ['B11'], 'A12', 'B11_A12_min')
    train, test = merge_std(train, test, ['B11'], 'A12', 'B11_A12_std')

    train, test = merge_median(train, test, ['B11'], 'A15', 'B11_A15_median')
    train, test = merge_mean(train, test, ['B11'], 'A15', 'B11_A15_mean')
    train, test = merge_sum(train, test, ['B11'], 'A15', 'B11_A15_sum')
    train, test = merge_max(train, test, ['B11'], 'A15', 'B11_A15_max')
    train, test = merge_min(train, test, ['B11'], 'A15', 'B11_A15_min')
    train, test = merge_std(train, test, ['B11'], 'A15', 'B11_A15_std')

    train, test = merge_median(train, test, ['B11'], 'A17', 'B11_A17_median')
    train, test = merge_mean(train, test, ['B11'], 'A17', 'B11_A17_mean')
    train, test = merge_sum(train, test, ['B11'], 'A17', 'B11_A17_sum')
    train, test = merge_max(train, test, ['B11'], 'A17', 'B11_A17_max')
    train, test = merge_min(train, test, ['B11'], 'A17', 'B11_A17_min')
    train, test = merge_std(train, test, ['B11'], 'A17', 'B11_A17_std')

    train, test = merge_median(train, test, ['B11'], 'A21', 'B11_A21_median')
    train, test = merge_mean(train, test, ['B11'], 'A21', 'B11_A21_mean')
    train, test = merge_sum(train, test, ['B11'], 'A21', 'B11_A21_sum')
    train, test = merge_max(train, test, ['B11'], 'A21', 'B11_A21_max')
    train, test = merge_min(train, test, ['B11'], 'A21', 'B11_A21_min')
    train, test = merge_std(train, test, ['B11'], 'A21', 'B11_A21_std')

    train, test = merge_median(train, test, ['B11'], 'A22', 'B11_A22_median')
    train, test = merge_mean(train, test, ['B11'], 'A22', 'B11_A22_mean')
    train, test = merge_sum(train, test, ['B11'], 'A22', 'B11_A22_sum')
    train, test = merge_max(train, test, ['B11'], 'A22', 'B11_A22_max')
    train, test = merge_min(train, test, ['B11'], 'A22', 'B11_A22_min')
    train, test = merge_std(train, test, ['B11'], 'A22', 'B11_A22_std')

    train, test = merge_count(train, test, ['B11'], 'A25', 'B11_A25_count')
    train, test = merge_nunique(train, test, ['B11'], 'A25', 'B11_A25_nunique')

    train, test = merge_median(train, test, ['B11'], 'A27', 'B11_A27_median')
    train, test = merge_mean(train, test, ['B11'], 'A27', 'B11_A27_mean')
    train, test = merge_sum(train, test, ['B11'], 'A27', 'B11_A27_sum')
    train, test = merge_max(train, test, ['B11'], 'A27', 'B11_A27_max')
    train, test = merge_min(train, test, ['B11'], 'A27', 'B11_A27_min')
    train, test = merge_std(train, test, ['B11'], 'A27', 'B11_A27_std')

    train, test = merge_median(train, test, ['B11'], 'B1', 'B11_B1_median')
    train, test = merge_mean(train, test, ['B11'], 'B1', 'B11_B1_mean')
    train, test = merge_sum(train, test, ['B11'], 'B1', 'B11_B1_sum')
    train, test = merge_max(train, test, ['B11'], 'B1', 'B11_B1_max')
    train, test = merge_min(train, test, ['B11'], 'B1', 'B11_B1_min')
    train, test = merge_std(train, test, ['B11'], 'B1', 'B11_B1_std')

    train, test = merge_count(train, test, ['B11'], 'B4', 'B11_B4_count')
    train, test = merge_nunique(train, test, ['B11'], 'B4', 'B11_B4_nunique')

    train, test = merge_count(train, test, ['B11'], 'B5', 'B11_B5_count')
    train, test = merge_nunique(train, test, ['B11'], 'B5', 'B11_B5_nunique')

    train, test = merge_median(train, test, ['B11'], 'B6', 'B11_B6_median')
    train, test = merge_mean(train, test, ['B11'], 'B6', 'B11_B6_mean')
    train, test = merge_sum(train, test, ['B11'], 'B6', 'B11_B6_sum')
    train, test = merge_max(train, test, ['B11'], 'B6', 'B11_B6_max')
    train, test = merge_min(train, test, ['B11'], 'B6', 'B11_B6_min')
    train, test = merge_std(train, test, ['B11'], 'B6', 'B11_B6_std')

    train, test = merge_count(train, test, ['B11'], 'B7', 'B11_B7_count')
    train, test = merge_nunique(train, test, ['B11'], 'B7', 'B11_B7_nunique')

    train, test = merge_median(train, test, ['B11'], 'B8', 'B11_B8_median')
    train, test = merge_mean(train, test, ['B11'], 'B8', 'B11_B8_mean')
    train, test = merge_sum(train, test, ['B11'], 'B8', 'B11_B8_sum')
    train, test = merge_max(train, test, ['B11'], 'B8', 'B11_B8_max')
    train, test = merge_min(train, test, ['B11'], 'B8', 'B11_B8_min')
    train, test = merge_std(train, test, ['B11'], 'B8', 'B11_B8_std')

    train, test = merge_count(train, test, ['B11'], 'B9', 'B11_B9_count')
    train, test = merge_nunique(train, test, ['B11'], 'B9', 'B11_B9_nunique')

    train, test = merge_count(train, test, ['B11'], 'B10', 'B11_B10_count')
    train, test = merge_nunique(train, test, ['B11'], 'B10', 'B11_B10_nunique')

    train, test = merge_median(train, test, ['B11'], 'B14', 'B11_B14_median')
    train, test = merge_mean(train, test, ['B11'], 'B14', 'B11_B14_mean')
    train, test = merge_sum(train, test, ['B11'], 'B14', 'B11_B14_sum')
    train, test = merge_max(train, test, ['B11'], 'B14', 'B11_B14_max')
    train, test = merge_min(train, test, ['B11'], 'B14', 'B11_B14_min')
    train, test = merge_std(train, test, ['B11'], 'B14', 'B11_B14_std')

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

    CF = list(map(lambda x: x + '_le', CategoricalFeature))
    train_data = lgb.Dataset(data=X_train, label=y_train, categorical_feature=CF)
    valid_data = lgb.Dataset(data=X_valid, label=y_valid, categorical_feature=CF)

    num_round = 1000
    clf = lgb.train(param, train_data, num_round, valid_sets=[train_data, valid_data], verbose_eval=50,
                    early_stopping_rounds=100)

    feature_imp = pd.DataFrame(sorted(
        zip(clf.feature_importance(importance_type='split'), clf.feature_importance(importance_type='gain'), feature)),
                               columns=['split', 'gain', 'Feature'])
    feature_imp.sort_values(by=['split'], inplace=True, ascending=False)
    print(feature_imp)

    if Online != False:
        test_pred = clf.predict(test[feature], num_iteration=clf.best_iteration)
        sub[1] = test_pred
        sub.to_csv('20190103_1.csv', index=None, header=None)


