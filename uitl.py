#!/bin/env python
# coding=utf-8
# Author:哟嚯走大运了
# Date:2019-01-02
# Email: yuwanlong2018@163.com

import pandas as pd
def merge_count(train, test, columns, value, cname):
    add=pd.DataFrame(train.groupby(columns)[value].count()).reset_index()
    add.columns = columns + [cname]
    train = train.merge(add,on=columns,how="left")
    test = test.merge(add,on=columns,how="left")
    return train, test

def merge_nunique(train, test, columns, value, cname):
    add = pd.DataFrame(train.groupby(columns)[value].nunique()).reset_index()
    add.columns=columns+[cname]
    train=train.merge(add,on=columns,how="left")
    test=test.merge(add,on=columns,how="left")
    return train,test

def merge_median(train,test,columns,value,cname):
    add = pd.DataFrame(train.groupby(columns)[value].median()).reset_index()
    add.columns=columns+[cname]
    train=train.merge(add,on=columns,how="left")
    test=test.merge(add, on=columns, how="left")
    return train,test

def merge_mean(train,test,columns,value,cname):
    add = pd.DataFrame(train.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    train=train.merge(add,on=columns,how="left")
    test=test.merge(add, on=columns, how="left")
    return train,test

def merge_sum(train,test,columns,value,cname):
    add = pd.DataFrame(train.groupby(columns)[value].sum()).reset_index()
    add.columns=columns+[cname]
    train=train.merge(add,on=columns,how="left")
    test=test.merge(add, on=columns, how="left")
    return train,test

def merge_max(train,test,columns,value,cname):
    add = pd.DataFrame(train.groupby(columns)[value].max()).reset_index()
    add.columns=columns+[cname]
    train=train.merge(add,on=columns,how="left")
    test=test.merge(add, on=columns, how="left")
    return train,test

def merge_min(train,test,columns,value,cname):
    add = pd.DataFrame(train.groupby(columns)[value].min()).reset_index()
    add.columns=columns+[cname]
    train=train.merge(add,on=columns,how="left")
    test=test.merge(add, on=columns, how="left")
    return train,test

def merge_std(train,test,columns,value,cname):
    add = pd.DataFrame(train.groupby(columns)[value].std()).reset_index()
    add.columns=columns+[cname]
    train=train.merge(add,on=columns,how="left")
    test=test.merge(add, on=columns, how="left")
    return train,test
