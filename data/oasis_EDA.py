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

best_score = 0
for c_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]: #iterate over the values we need to try for the parameter C
    for gamma_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]: #iterate over the values we need to try for the parameter gamma
        for k_parameter in ['rbf', 'linear', 'poly', 'sigmoid']: # iterate over the values we need to try for the kernel parameter
            svmModel = SVC(kernel=k_parameter, C=c_paramter, gamma=gamma_paramter) #define the model
            # perform cross-validation
            scores = cross_val_score(svmModel, X_trainval_scaled, Y_trainval, cv=kfolds, scoring='accuracy')
            # the training set will be split internally into training and cross validation

            # compute mean cross-validation accuracy
            score = np.mean(scores)
            # if we got a better score, store the score and parameters
            if score > best_score:
                best_score = score #store the score 
                best_parameter_c = c_paramter #store the parameter c
                best_parameter_gamma = gamma_paramter #store the parameter gamma
                best_parameter_k = k_parameter
            

# rebuild a model with best parameters to get score 
SelectedSVMmodel = SVC(C=best_parameter_c, gamma=best_parameter_gamma, kernel=best_parameter_k).fit(X_trainval_scaled, Y_trainval)

test_score = SelectedSVMmodel.score(X_test_scaled, Y_test)
PredictedOutput = SelectedSVMmodel.predict(X_test_scaled)
test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)
print("Best accuracy on cross validation set is:", best_score)
print("Best parameter for c is: ", best_parameter_c)
print("Best parameter for gamma is: ", best_parameter_gamma)
print("Best parameter for kernel is: ", best_parameter_k)
print("Test accuracy with the best parameters is", test_score)
print("Test recall with the best parameters is", test_recall)
print("Test recall with the best parameter is", test_auc)

m = 'SVM'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])

################################# SVM ########################################
# %%
################################ W/ PCA ######################################
# %%
############################### LOGISTIC REGRESSION ##########################