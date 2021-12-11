#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:46:10 2021

@author: jake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
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

final_data_li.drop(final_data_li.loc[final_data_li['HINCP']==0].index, inplace=True)

classes = {1:'Very Low',2:'Low',3:'Medium',4:'High',5:'Very High'}

data['BURDENC']=0
for i in data.index:
    if data.loc[i,'BURDEN'] < np.percentile(data['BURDEN'],20):
        data.loc[i,'BURDENC'] = 1
    elif data.loc[i,'BURDEN'] < np.percentile(data['BURDEN'],40):
        data.loc[i,'BURDENC'] = 2
    elif data.loc[i,'BURDEN'] < np.percentile(data['BURDEN'],60):
        data.loc[i,'BURDENC'] = 3
    elif data.loc[i,'BURDEN'] < np.percentile(data['BURDEN'],80):
        data.loc[i,'BURDENC'] = 4
    else:
        data.loc[i,'BURDENC'] = 5
        
for i in final_data_li.index:
    if final_data_li.loc[i,'BURDEN'] < np.percentile(data['BURDEN'],20):
        final_data_li.loc[i,'BURDENC'] = 1
    elif final_data_li.loc[i,'BURDEN'] < np.percentile(data['BURDEN'],40):
        final_data_li.loc[i,'BURDENC'] = 2
    elif final_data_li.loc[i,'BURDEN'] < np.percentile(data['BURDEN'],60):
        final_data_li.loc[i,'BURDENC'] = 3
    elif final_data_li.loc[i,'BURDEN'] < np.percentile(data['BURDEN'],80):
        final_data_li.loc[i,'BURDENC'] = 4
    else:
        final_data_li.loc[i,'BURDENC'] = 5
        
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

features_demo_all = ['HHRACE_1.0', 'HHRACE_2.0', 'HHRACE_3.0',\
                     'HHRACE_4.0', 'HHRACE_5.0', 'HHRACE_6.0','TENURE_0.0', 'TENURE_1.0', 'TENURE_2.0',\
                     'NUMELDERS', 'NUMYNGKIDS', 'NUMOLDKIDS', 'URBAN',\
                     'DIVISION_1','DIVISION_2','DIVISION_3','DIVISION_4','DIVISION_5',\
                         'DIVISION_6','DIVISION_7','DIVISION_8','DIVISION_9']

features_demo_rural = ['HHRACE_1.0', 'HHRACE_2.0', 'HHRACE_3.0',\
                     'HHRACE_4.0', 'HHRACE_5.0', 'HHRACE_6.0','TENURE_0.0', 'TENURE_1.0', 'TENURE_2.0',\
                     'NUMELDERS', 'NUMYNGKIDS', 'NUMOLDKIDS',\
                     'DIVISION_1','DIVISION_2','DIVISION_3','DIVISION_4','DIVISION_5',\
                         'DIVISION_6','DIVISION_7','DIVISION_8','DIVISION_9']
    
features_demo_urban = ['HHRACE_1.0', 'HHRACE_2.0', 'HHRACE_3.0',\
                     'HHRACE_4.0', 'HHRACE_5.0', 'HHRACE_6.0','TENURE_0.0', 'TENURE_1.0', 'TENURE_2.0',\
                     'NUMELDERS', 'NUMYNGKIDS', 'NUMOLDKIDS',\
                     'DIVISION_1','DIVISION_2','DIVISION_3','DIVISION_4','DIVISION_5',\
                         'DIVISION_6','DIVISION_7','DIVISION_8','DIVISION_9']

features_house_all = ['ACPRIMARY_0', 'ACPRIMARY_1', 'ACPRIMARY_2', 'ACPRIMARY_3', 'ACPRIMARY_4',\
                      'ACPRIMARY_5', 'ACPRIMARY_6', 'ACPRIMARY_7', 'BLD_1', 'BLD_2', 'BLD_3', 'BLD_4',\
                      'BLD_5', 'BLD_6', 'BLD_7', 'BLD_8', 'BLD_9', 'COOKFUEL_0', 'COOKFUEL_1',\
                      'COOKFUEL_2', 'COOKFUEL_3', 'COOKFUEL_4', 'HEATFUEL_0', 'HEATFUEL_1', 'HEATFUEL_2',\
                      'HEATFUEL_3', 'HEATFUEL_4', 'HEATFUEL_5', 'HEATFUEL_6', 'HEATFUEL_7', 'HEATFUEL_8',\
                      'HEATFUEL_9', 'HEATTYPE_0', 'HEATTYPE_1', 'HEATTYPE_2', 'HEATTYPE_3', 'HEATTYPE_4',\
                      'HEATTYPE_5', 'HEATTYPE_6', 'HEATTYPE_7', 'HEATTYPE_8', 'HEATTYPE_9', 'HEATTYPE_10',\
                      'HEATTYPE_11', 'HEATTYPE_12', 'HEATTYPE_13', 'HOTWATER_0', 'HOTWATER_1', 'HOTWATER_2',\
                      'HOTWATER_3', 'HOTWATER_4', 'HOTWATER_5', 'HOTWATER_6', 'FIREPLACE', 'SOLAR', 'UNITSIZE', 'YRBUILT', 'URBAN']

features_house_rural = ['ACPRIMARY_0', 'ACPRIMARY_1', 'ACPRIMARY_2', 'ACPRIMARY_3', 'ACPRIMARY_4',\
                      'ACPRIMARY_5', 'ACPRIMARY_6', 'ACPRIMARY_7', 'BLD_1', 'BLD_2', 'BLD_3', 'BLD_4',\
                      'BLD_5', 'BLD_6', 'BLD_7', 'BLD_8', 'BLD_9', 'COOKFUEL_0', 'COOKFUEL_1',\
                      'COOKFUEL_2', 'COOKFUEL_3', 'COOKFUEL_4', 'HEATFUEL_0', 'HEATFUEL_1', 'HEATFUEL_2',\
                      'HEATFUEL_3', 'HEATFUEL_4', 'HEATFUEL_5', 'HEATFUEL_6', 'HEATFUEL_7', 'HEATFUEL_8',\
                      'HEATFUEL_9', 'HEATTYPE_0', 'HEATTYPE_1', 'HEATTYPE_2', 'HEATTYPE_3', 'HEATTYPE_4',\
                      'HEATTYPE_5', 'HEATTYPE_6', 'HEATTYPE_7', 'HEATTYPE_8', 'HEATTYPE_9', 'HEATTYPE_10',\
                      'HEATTYPE_11', 'HEATTYPE_12', 'HEATTYPE_13', 'HOTWATER_0', 'HOTWATER_1', 'HOTWATER_2',\
                      'HOTWATER_3', 'HOTWATER_4', 'HOTWATER_5', 'HOTWATER_6', 'FIREPLACE', 'SOLAR', 'UNITSIZE', 'YRBUILT']

features_house_urban = ['ACPRIMARY_0', 'ACPRIMARY_1', 'ACPRIMARY_2', 'ACPRIMARY_3', 'ACPRIMARY_4',\
                      'ACPRIMARY_5', 'ACPRIMARY_6', 'ACPRIMARY_7', 'BLD_1', 'BLD_2', 'BLD_3', 'BLD_4',\
                      'BLD_5', 'BLD_6', 'BLD_7', 'BLD_8', 'BLD_9', 'COOKFUEL_0', 'COOKFUEL_1',\
                      'COOKFUEL_2', 'COOKFUEL_3', 'COOKFUEL_4', 'HEATFUEL_0', 'HEATFUEL_1', 'HEATFUEL_2',\
                      'HEATFUEL_3', 'HEATFUEL_4', 'HEATFUEL_5', 'HEATFUEL_6', 'HEATFUEL_7', 'HEATFUEL_8',\
                      'HEATFUEL_9', 'HEATTYPE_0', 'HEATTYPE_1', 'HEATTYPE_2', 'HEATTYPE_3', 'HEATTYPE_4',\
                      'HEATTYPE_5', 'HEATTYPE_6', 'HEATTYPE_7', 'HEATTYPE_8', 'HEATTYPE_9', 'HEATTYPE_10',\
                      'HEATTYPE_11', 'HEATTYPE_12', 'HEATTYPE_13', 'HOTWATER_0', 'HOTWATER_1', 'HOTWATER_2',\
                      'HOTWATER_3', 'HOTWATER_4', 'HOTWATER_5', 'HOTWATER_6', 'FIREPLACE', 'SOLAR', 'UNITSIZE', 'YRBUILT']
    
features_house_li = ['ACPRIMARY_0', 'ACPRIMARY_1', 'ACPRIMARY_2', 'ACPRIMARY_3', 'ACPRIMARY_4',\
                      'ACPRIMARY_5', 'ACPRIMARY_6', 'ACPRIMARY_7', 'BLD_1', 'BLD_2', 'BLD_3', 'BLD_4',\
                      'BLD_5', 'BLD_6', 'BLD_7', 'BLD_8', 'BLD_9', 'COOKFUEL_0', 'COOKFUEL_1',\
                      'COOKFUEL_2', 'COOKFUEL_3', 'COOKFUEL_4', 'HEATFUEL_0', 'HEATFUEL_1', 'HEATFUEL_2',\
                      'HEATFUEL_3', 'HEATFUEL_4', 'HEATFUEL_5', 'HEATFUEL_6', 'HEATFUEL_7',\
                      'HEATFUEL_9', 'HEATTYPE_0', 'HEATTYPE_1', 'HEATTYPE_2', 'HEATTYPE_3', 'HEATTYPE_4',\
                      'HEATTYPE_5', 'HEATTYPE_6', 'HEATTYPE_7', 'HEATTYPE_8', 'HEATTYPE_9', 'HEATTYPE_10',\
                      'HEATTYPE_11', 'HEATTYPE_12', 'HEATTYPE_13', 'HOTWATER_0', 'HOTWATER_1', 'HOTWATER_2',\
                      'HOTWATER_3', 'HOTWATER_4', 'HOTWATER_5', 'HOTWATER_6', 'FIREPLACE', 'SOLAR', 'UNITSIZE', 'YRBUILT']
    
features_all = data.drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE','METRO','BURDEN','BURDENC'])
features_all_urban = data[data.URBAN==1].drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE','METRO','BURDEN','BURDENC','URBAN'])
features_all_rural = data[data.URBAN==0].drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE','METRO','BURDEN','BURDENC','URBAN'])

features_all_rural_li = final_data_li[final_data_li.URBAN==0].drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE',\
                                                                            'METRO','BURDEN','HHAGE','NUMKIDS', 'POVTHRESH', 'LITHRESH',\
                                                                            'URBAN','BURDENC'])
features_all_urban_li = final_data_li[final_data_li.URBAN==1].drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE',\
                                                                            'METRO','BURDEN','HHAGE','NUMKIDS', 'POVTHRESH', 'LITHRESH',\
                                                                            'BURDENC'])
features_all_li = final_data_li.drop(columns=['HINCP','OTHERAMT','GASAMT','OILAMT','ELECAMT','NUMPEOPLE',\
                                                                            'METRO','BURDEN','HHAGE','NUMKIDS', 'POVTHRESH', 'LITHRESH',\
                                                                            'BURDENC'])

###############################################################################

### RURAL ONLY, DEMOGRAPHIC FEATURES, ALL INCOME, ALL REGIONS
target = data[data.URBAN==0]['BURDENC']
# split test set
X, X_test, y, y_test = train_test_split(data[data.URBAN==0][features_demo_rural], target, random_state = 1, test_size = .2)

### URBAN ONLY, DEMOGRAPHIC FEATURES, ALL INCOME, ALL REGIONS
target = data[data.URBAN==1]['BURDENC']
# split test set
X, X_test, y, y_test = train_test_split(data[data.URBAN==1][features_demo_urban], target, random_state = 1, test_size = .2)

### BOTH URBAN AND RURAL, DEMOGRAPHIC FEATURES, ALL INCOME, ALL REGIONS
target = data['BURDENC']
# split test set
X, X_test, y, y_test = train_test_split(data[features_demo_all], target, random_state = 1, test_size = .2)

### RURAL ONLY, HOUSING FEATURES, ALL INCOME, ALL REGIONS
target = data[data.URBAN==0]['BURDENC']
# split test set
X, X_test, y, y_test = train_test_split(data[data.URBAN==0][features_house_rural], target, random_state = 1, test_size = .2)

### URBAN ONLY, HOUSING FEATURES, ALL INCOME, ALL REGIONS
target = data[data.URBAN==1]['BURDENC']
# split test set
X, X_test, y, y_test = train_test_split(data[data.URBAN==1][features_house_urban], target, random_state = 1, test_size = .2)

### BOTH URBAN AND RURAL, HOUSING FEATURES, ALL INCOME, ALL REGIONS
target = data['BURDENC']
# split test set
X, X_test, y, y_test = train_test_split(data[features_house_all], target, random_state = 1, test_size = .2)

### RURAL ONLY, ALL FEATURES, ALL INCOME, ALL REGIONS
target = data[data.URBAN==0]['BURDENC']
# split test set
X, X_test, y, y_test = train_test_split(features_all_rural, target, random_state = 1, test_size = .2)

### URBAN ONLY, ALL FEATURES, ALL INCOME, ALL REGIONS
target = data[data.URBAN==1]['BURDENC']
# split test set
X, X_test, y, y_test = train_test_split(features_all_urban, target, random_state = 1, test_size = .2)

### BOTH URBAN AND RURAL, ALL FEATURES, ALL INCOME, ALL REGIONS
target = data['BURDENC']
# split test set
X, X_test, y, y_test = train_test_split(features_all, target, random_state = 1, test_size = .2)

# ########## LOW INCOME TESTS ##########

# ### RURAL ONLY, DEMOGRAPHIC FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li[final_data_li.URBAN==0]['BURDENC']
# # split test set
# X, X_test, y, y_test = train_test_split(final_data_li[final_data_li.URBAN==0][features_demo_rural], target, random_state = 1, test_size = .2)

# ### URBAN ONLY, DEMOGRAPHIC FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li[final_data_li.URBAN==1]['BURDENC']
# # split test set
# X, X_test, y, y_test = train_test_split(final_data_li[final_data_li.URBAN==1][features_demo_urban], target, random_state = 1, test_size = .2)

# ### BOTH URBAN AND RURAL, DEMOGRAPHIC FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li['BURDENC']
# # split test set
# X, X_test, y, y_test = train_test_split(final_data_li[features_demo_all], target, random_state = 1, test_size = .2)

# ### RURAL ONLY, HOUSING FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li[final_data_li.URBAN==0]['BURDENC']
# # split test set
# X, X_test, y, y_test = train_test_split(final_data_li[final_data_li.URBAN==0][features_house_li], target, random_state = 1, test_size = .2)

# ### URBAN ONLY, HOUSING FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li[final_data_li.URBAN==1]['BURDENC']
# # split test set
# X, X_test, y, y_test = train_test_split(final_data_li[final_data_li.URBAN==1][features_house_li], target, random_state = 1, test_size = .2)

# ### BOTH URBAN AND RURAL, HOUSING FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li['BURDENC']
# # split test set
# X, X_test, y, y_test = train_test_split(final_data_li[features_house_li+['URBAN']], target, random_state = 1, test_size = .2)

# ### RURAL ONLY, ALL FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li[final_data_li.URBAN==0]['BURDENC']
# # split test set
# X, X_test, y, y_test = train_test_split(features_all_rural_li, target, random_state = 1, test_size = .2)

# ### URBAN ONLY, ALL FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li[final_data_li.URBAN==1]['BURDENC']
# # split test set
# X, X_test, y, y_test = train_test_split(features_all_urban_li, target, random_state = 1, test_size = .2)

# ### BOTH URBAN AND RURAL, ALL FEATURES, LOW INCOME, ALL REGIONS
# target = final_data_li['BURDENC']
# # split test set
# X, X_test, y, y_test = train_test_split(features_all_li, target, random_state = 1, test_size = .2)



# split between train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 1, test_size = 0.2)

rf_tree = RandomForestClassifier()
rf_tree.fit(X_train, y_train)

# param_dist_rf = {'n_estimators': randint(10, 100),
#               'max_leaf_nodes': randint(3, 100),
#               'max_features': ["auto"],
#               'max_depth': randint(1, 10),
#               'min_samples_leaf': randint(1, 30),
#               'min_samples_split': randint(2, 20)}
param_dist_rf = {'n_estimators': randint(1, 5000),
              'max_leaf_nodes': randint(3, 100),
              'max_features': ["auto"],
              'max_depth': randint(1, 50),
              'min_samples_leaf': randint(1, 40),
              'min_samples_split': randint(2, 40)}

rnd_search_rf = RandomizedSearchCV(rf_tree, param_distributions=param_dist_rf, cv=10, n_iter=5) ### SHOULD BE 200

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
plt.savefig('figs/feature_importance_classifier.png',dpi=300)