#!/usr/bin/env python3
# %%
# https://docs.microsoft.com/en-us/visualstudio/liveshare/use/vscode
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import typing
import seaborn as sns
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

CROSS_SECTIONAL = 'oasis_cross-sectional.csv'
LONGITUDINAL = 'oasis_longitudinal.csv'

########## https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

data_l = pd.read_csv(LONGITUDINAL) 
data_cs = pd.read_csv(CROSS_SECTIONAL)

# Changes the column name.
data_cs.rename(columns={'Educ':'EDUC'}, inplace=True)
data_cs.rename(columns={'ID':'Subject ID'}, inplace=True)
data_l.rename(columns={'MR Delay':'Delay'}, inplace=True)

# %%
print(data_l.describe())
for column in data_l:
    try:
        data_l[column].plot.hist()
    except TypeError:
        pass
    
# %%

def plot_hist(df):
    fig, axes = plt.subplots(len(df.columns)//3, 3, figsize=(12, 48))
    i = 0
    for triaxis in axes:
        for axis in triaxis:
            try:
                df.hist(column = df.columns[i], bins = 100, ax=axis)
            except ValueError as e:
                pass
            i = i+1

    plt.show()

def plot_scatter(df):
    fig, axes = plt.subplots(len(df.columns)//3, 3, figsize=(12, 48))
    i = 0
    for triaxis in axes:
        for axis in triaxis:
            try:
                df.scatter(column = df.columns[i], bins = 100, ax=axis)
            except ValueError as e:
                pass
            i = i+1

    plt.show()

# %%

# data cleaning
#https://thispointer.com/pandas-replace-nan-with-mean-or-average-in-dataframe-using-fillna/
sub_k = ['EDUC','SES','MMSE']
for k in sub_k:
    data_l[k].fillna(value=data_l[k].mean(), inplace=True)
    data_cs[k].fillna(value=data_l[k].mean(), inplace=True)

# https://izziswift.com/label-encoding-across-multiple-columns-in-scikit-learn/
encodings = ['M/F', 'Hand', 'Subject ID']
for encode in encodings:
    le = LabelEncoder()
    data_l[encode] = data_l[[encode]].apply(le.fit_transform)
    data_cs[encode] = data_cs[[encode]].apply(le.fit_transform)

# %%  
# https://www.codespeedy.com/normalize-a-pandas-dataframe-column/
numerics = ['Delay', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
for k in numerics:
    normed_data_l = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(data_l[[k]].values))
    normed_data_cs = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(data_cs[[k]].values))

# %%
features = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
# correlation matrix
cm_l = data_l[features].corr()
cm_cs = data_cs[features].corr()
# multilinearity
#vif_l['features'] = features
#vif_cs['features'] = features

#from statsmodels.stats.outliers_influence import variance_inflation_factor
#vif_l['VIF'] = [variance_inflation_factor(data_l[features]) for i in range(len(features))]

print(cm_l)
print(cm_cs)
# %%
# histograms for each column
print("DATA_L")
plot_hist(data_l)
plt.show()
print("DATA_CS")
plot_hist(data_cs)

# %%
data_l.describe()
# %%
data_cs.describe()


# %%
############################# RANDOM FORESTS #################################
#  %%



################################# SVM ########################################
# %%
################################ W/ PCA ######################################
# %%
############################### LOGISTIC REGRESSION ##########################