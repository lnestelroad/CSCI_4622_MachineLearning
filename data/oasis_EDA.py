#!/usr/bin/env python3
# %%
# https://docs.microsoft.com/en-us/visualstudio/liveshare/use/vscode
import pathlib
from numpy import testing
from numpy.fft.helper import fftfreq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import typing
import seaborn as sns
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)
from time import time


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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, precision_score

# The CDR column contains the target information.
train_me = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF','CDR']

l_train, l_test = train_test_split(data_l[train_me])
cs_train, cs_test = train_test_split(data_cs[train_me])


#%%

class EnsembleTest:
    """ Test multiple model performance """
    
    def __init__(self, X_train, y_train, X_test, y_test, _type='regression'):
        """
        initialize data and type of problem
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param _type: regression or classification
        """
        self.scores = {}
        self.execution_time = {}
        self.metric = {}
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.type = _type
        if _type == 'regression':
            self.score_name = 'R^2 score' 
            self.metric_name = 'Explained variance' 
        else:
            self.score_name = 'Mean accuracy score'
            self.metric_name = 'Precision'
            
    def fit_model(self, model, name):
        """
        TODO:
        - Fit the model on train data.
        - Store execution time required to fit.
        - Store scores on test data
        - Predict on test data
        
        Each model passed as parameter has member functions of the following form:
          model.fit(x_train, y_train)
          model.score(x_test, y_test) 
          model.predict(x_test)
        
        :param model: model
        :param name: name of model
        """
        
        # YOUR CODE HERE
        t_start = time()
        model.fit(X_train,y_train)
        t_end = time()

        self.execution_time[name] = t_end-t_start
        self.scores[name] = model.score(X_test, y_test) 
        predict = model.predict(X_test)
        
        if self.type == 'regression':
            self.metric[name] = explained_variance_score(self.y_test, predict)
        elif self.type == 'classification':
            self.metric[name] = precision_score(self.y_test, predict)

    def print_result(self):
        """
            print results for all models trained and tested.
        """
        models_cross = pd.DataFrame({
            'Model'         : list(self.metric.keys()),
             self.score_name     : list(self.scores.values()),
             self.metric_name    : list(self.metric.values()),
            'Execution time': list(self.execution_time.values())})
        print(models_cross.sort_values(by=self.score_name, ascending=False))

    def plot_metric(self):
        """
         There are 3 metrics for each type of experiment: 
             time, metric score, scores
         Produce a bar graph for each of the above metrics
         Each bar graph should have results from all experiments 
             of the same type side by side
         
         Note: The Metric name and score name depend on the type of experiment 
             (regression or classification) 
        """
        # YOUR CODE HERE
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        exp_names = [key for key in self.execution_time]
        exp_times = [self.execution_time[key] for key in exp_names]
        ax.bar(exp_names,exp_times)
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Execution Times')
        ax.title.set_text('Experiment Execution Times')
        ax.grid()

        fig1 = plt.figure()
        ax1 = fig1.add_axes([0,0,1,1])
        exp_scores = [self.scores[key] for key in exp_names]
        ax1.bar(exp_names,exp_scores)
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Scores')
        ax1.title.set_text('Experiment Scores')
        ax1.grid()

        fig2 = plt.figure()
        ax2 = fig2.add_axes([0,0,1,1])
        exp_metscore = [self.metric[key] for key in exp_names]
        ax2.bar(exp_names,exp_metscore)
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Metric Scores')
        ax2.title.set_text('Experiment Metric Scores')
        ax2.grid()

#L_trianing data split
y_train = l_train[['CDR']]
del l_train['CDR']

X_train = l_train

# L_testing data split
y_test = l_test[['CDR']]
del l_test['CDR']

X_test = l_test

# # CS_training data split
# y_train = l_train[['CDR']]
# del l_train['CDR']

# X_train = l_train
# %%
l_handler = EnsembleTest(
    X_train, y_train, X_test, y_test, _type='regression')

l_rf = RandomForestRegressor(n_estimators=100)
l_rf.fit(X_train,y_train)



#%%
from sklearn.metrics import f1_score

y_hat_test = l_rf.predict(X_test)
met = f1_score(y_test, y_hat_test)



################################# SVM ########################################
# %%

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

l_pca = PCA()

l_svm = LinearSVC()

################################ W/ PCA ######################################
# %%
############################### LOGISTIC REGRESSION ##########################