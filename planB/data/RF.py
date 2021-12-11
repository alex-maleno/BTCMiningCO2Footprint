#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:28:44 2021

@author: jake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

data = pd.read_csv('final_cleaned_dummy_data.csv').drop(columns=['Unnamed: 0', 'index'])
data_li = pd.read_csv('lowIncomeFilteredData.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

# use one hot encoding to convert categorical variables into series of binary variables
non_binary_cat_vars = ['ACPRIMARY','BLD','COOKFUEL','DIVISION','HEATFUEL','HEATTYPE','HHRACE','HOTWATER','TENURE']

cat_df = data_li[non_binary_cat_vars].copy(deep=True)

enc = OneHotEncoder()
enc.fit(cat_df)
onehotlabels = enc.transform(cat_df).toarray()
one_hot_df = pd.DataFrame(onehotlabels, columns=enc.get_feature_names_out())

# recombine dataframes
final_data_li = pd.concat([one_hot_df,data_li.drop(columns=non_binary_cat_vars)], axis=1)

data.drop(data.loc[data['HINCP']==0].index, inplace=True)

data.drop(data.loc[data['BURDEN']>1.3068144].index, inplace=True)
data.drop(data.loc[data['BURDEN']<=0].index, inplace=True)
#data.drop(data.loc[data['BURDEN']<=0.0070588235].index, inplace=True) #5th percentile

final_data_li.drop(final_data_li.loc[final_data_li['HINCP']==0].index, inplace=True)

# ### 3.2.2 Random Forest
# Random forests take a similar approach to bagging, in which we fit various decision trees on resampled data.
# But, when each tree is constructed, not every feature is considered as split candidates for each decision point;
# we only take a subset of the total predictors in the model.
# 
# When building a random forest compared to bagging, adding randomization into the features that create the model,
# then averaging the predictions across models will typically produce a model that "decorrelates" the trees and in
# turn is more reliable for prediction.
# 
# We'll use scikit-learn's `RandomForestClassifier()` to implement our model. The documentation can be found
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#
# Create a RandomForestClassifier. Use `RandomizedSearchCV` with the parameters shown in the param_dist dictionary.
# Print out the best parameters. Fit a model called rf_tree with the best parameters and print out the score (average accuracy)
# of the model for both the training and validation data.

    
features_all = data.drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE','METRO','BURDEN'])
features_all_urban = data[data.URBAN==1].drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE','METRO','BURDEN','URBAN'])
features_all_rural = data[data.URBAN==0].drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE','METRO','BURDEN','URBAN'])

# categorical only
#features_all_rural = data[data.URBAN==0].drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE','METRO','BURDEN','URBAN',\
#                                                       'NUMELDERS', 'NUMYNGKIDS', 'NUMOLDKIDS', 'UNITSIZE', 'YRBUILT'])

features_all_rural_li = final_data_li[final_data_li.URBAN==0].drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE',\
                                                                            'METRO','BURDEN','HHAGE','NUMKIDS', 'POVTHRESH', 'LITHRESH',\
                                                                            'URBAN'])
features_all_urban_li = final_data_li[final_data_li.URBAN==1].drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE',\
                                                                            'METRO','BURDEN','HHAGE','NUMKIDS', 'POVTHRESH', 'LITHRESH'])
features_all_li = final_data_li.drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE',\
                                                                            'METRO','BURDEN','HHAGE','NUMKIDS', 'POVTHRESH', 'LITHRESH'])

    

###############################################################################

### RURAL ONLY, ALL FEATURES, ALL INCOME, ALL REGIONS
target = data[data.URBAN==0]['BURDEN']
# split test set
X, X_test, y, y_test = train_test_split(features_all_rural, target, random_state = 1, test_size = .2)

### URBAN ONLY, ALL FEATURES, ALL INCOME, ALL REGIONS
target = data[data.URBAN==1]['BURDEN']
# split test set
X, X_test, y, y_test = train_test_split(features_all_urban, target, random_state = 1, test_size = .2)

### BOTH URBAN AND RURAL, ALL FEATURES, ALL INCOME, ALL REGIONS
target = data['BURDEN']
# split test set
X, X_test, y, y_test = train_test_split(features_all, target, random_state = 1, test_size = .2)

# ########## LOW INCOME TESTS ##########

# ### RURAL ONLY, ALL FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li[final_data_li.URBAN==0]['BURDEN']
# # split test set
# X, X_test, y, y_test = train_test_split(features_all_rural_li, target, random_state = 1, test_size = .2)

# ### URBAN ONLY, ALL FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li[final_data_li.URBAN==1]['BURDEN']
# # split test set
# X, X_test, y, y_test = train_test_split(features_all_urban_li, target, random_state = 1, test_size = .2)

# ### BOTH URBAN AND RURAL, ALL FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li['BURDEN']
# # split test set
# X, X_test, y, y_test = train_test_split(features_all_li, target, random_state = 1, test_size = .2)

# split between train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 1, test_size = 0.2)

rf_tree = RandomForestRegressor()
rf_tree.fit(X_train, y_train)

# param_dist_rf = {'n_estimators': randint(10, 100),
#               'max_leaf_nodes': randint(3, 100),
#               'max_features': ["auto"],
#               'max_depth': randint(1, 10),
#               'min_samples_leaf': randint(1, 30),
#               'min_samples_split': randint(2, 20)}
param_dist_rf = {'n_estimators': randint(1, 3000),
              'max_leaf_nodes': randint(3, 200),
              'max_features': ["auto"],
              'max_depth': randint(1, 100),
              'min_samples_leaf': randint(1, 40),
              'min_samples_split': randint(2, 40)}

rnd_search_rf = RandomizedSearchCV(rf_tree, param_distributions=param_dist_rf, cv=10, n_iter=1) ### SHOULD BE 200

rnd_search_rf.fit(X_train, y_train)

print(rnd_search_rf.best_params_)

rf_tree.set_params(max_leaf_nodes=rnd_search_rf.best_params_['max_leaf_nodes'], 
                    n_estimators=rnd_search_rf.best_params_['n_estimators'], 
                    max_features = rnd_search_rf.best_params_['max_features'],
                    max_depth = rnd_search_rf.best_params_['max_depth'],
                    min_samples_leaf = rnd_search_rf.best_params_['min_samples_leaf'],
                    min_samples_split = rnd_search_rf.best_params_['min_samples_split'])

rf_train_score = rf_tree.score(X_train, y_train)
rf_val_score = rf_tree.score(X_val, y_val)

print('Train Score: ', rf_train_score)
print('Validation Score: ', rf_val_score)


# One of the nice built in features of the RandomForestClassifier is that you are able to get features importances
# for all trees in the random forest. This allows you to not only calculate the average importance of each feature,
# but also the standard deviation of the importance of each feature. Below, these are calculated and shown visually.


importances = rf_tree.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_tree.estimators_],
              axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

feat_imp = []
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f, X.columns[indices[f]], importances[indices[f]]))
    feat_imp.append((X.columns[indices[f]], importances[indices[f]]))
    
feat_df = pd.DataFrame(feat_imp, columns=['Feature','Importance'])
feat_df.set_index('Feature',inplace=True)
feat_df.to_csv('importances.csv')
feat_df = feat_df.T

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.savefig('figs/feature_importance.png',dpi=300)
plt.clf()


### RESIDUALS

plt.scatter(np.arange(len(y)),y-rf_tree.predict(X))
plt.savefig('figs/residuals.png',dpi=300)
plt.clf()
plt.scatter(np.arange(len(y)),np.sort(y-rf_tree.predict(X)))
plt.savefig('figs/residuals_sorted.png',dpi=300)
plt.clf()

### PERMUTATION IMPORTANCE

from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(rf_tree, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.savefig('figs/permutation_importance.png',dpi=300)

### SHAPLEY VALUES

import shap
explainer = shap.TreeExplainer(rf_tree)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar",show=False)
plt.savefig('figs/shapley_bar.png',dpi=300, bbox_inches='tight')
plt.clf()
shap.summary_plot(shap_values, X_test,show=False)
plt.savefig('figs/shapley_jitter.png',dpi=300, bbox_inches='tight')

from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test,rf_tree.predict(X_test))
mse_val = mean_squared_error(y_val,rf_tree.predict(X_val))
print('Test MSE is:',mse_test)
print('Val MSE is:',mse_val)

# Test and Val MSE
# Important predictors


