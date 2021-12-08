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

data = pd.read_csv('cleaned_data_v2.csv').drop(columns=['Unnamed: 0','Unnamed: 0.1'])

# use one hot encoding to convert categorical variables into series of binary variables
non_binary_cat_vars = ['ADEQUACY','BLD','COOKFUEL','DIVISION','HEATFUEL','HEATTYPE','HHCITSHP','HHRACE','HOTWATER','TENURE']

cat_df = data[non_binary_cat_vars].copy(deep=True)

enc = OneHotEncoder()
enc.fit(cat_df)
onehotlabels = enc.transform(cat_df).toarray()

one_hot_df = pd.DataFrame(onehotlabels, columns=enc.get_feature_names_out())
for i in one_hot_df.columns:
    if one_hot_df[i].name[-3:] == 'nan':
        one_hot_df = one_hot_df.drop(columns=[one_hot_df[i].name])

# recombine dataframes
final_data = pd.concat([data.drop(columns=non_binary_cat_vars).fillna(-1),one_hot_df], axis=1)

# calculate response column
final_data['ELECBURDEN'] = (final_data['ELECAMT']/final_data['HINCP'])*12
final_data['ELECBURDEN'].replace(np.inf,np.nan,inplace=True)
final_data = final_data.dropna(subset=['ELECBURDEN'])

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

features = final_data.drop(columns=['ELECBURDEN','ELECAMT','HINCP'])
target = final_data['ELECBURDEN']

# split test set
X, X_test, y, y_test = train_test_split(features, target, random_state = 1, test_size = .2)

# split between train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 1, test_size = 0.25)

rf_tree = RandomForestRegressor()
rf_tree.fit(X_train, y_train)

param_dist_rf = {'n_estimators': randint(10, 100),
              'max_leaf_nodes': randint(3, 100),
              'max_features': ["auto"],
              'max_depth': randint(1, 10),
              'min_samples_leaf': randint(1, 30),
              'min_samples_split': randint(2, 20)}

rnd_search_rf = RandomizedSearchCV(rf_tree, param_distributions=param_dist_rf, cv=10, n_iter=200) ### SHOULD BE 200

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


# One of the nice builT in features of the RandomForestClassifier is that you are able to get features importances
# for all trees in the random forest. This allows you to not only calculate the average importance of each feature,
# but also the standard deviation of the importance of each feature. Below, these are calculated and shown visually.


importances = rf_tree.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_tree.estimators_],
              axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f, X.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.savefig('feature_importance.png',dpi=300)









