#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:51:05 2021

@author: jake
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

raw_data = pd.read_csv('ahs2019n.csv')
print('Data read')

# predictors = ['ACPRIMARY','ADEQUACY','BLD','CONDO','COOKFUEL','DISHH','DIVISION','ELECAMT','FIREPLACE',\
#               'HEATFUEL','HEATTYPE','HHAGE','HHCITSHP','HHGRAD','HHMOVE','HHRACE','HINCP',\
#                   'HOTWATER','HUDSUB','MAINTAMT','MARKETVAL','MILHH','NUMCARE','NUMELDERS','NUMERRND',\
#                       'NUMPEOPLE','NUMYNGKIDS','NUMOLDKIDS','OMB13CBSA','PERPOVLVL','RATINGNH','RATINGHS',\
#                           'RENTCNTRL','SOLAR','STORIES','TENURE','TOTROOMS','UNITSIZE','WATERAMT','YRBUILT']

predictors = ['ACPRIMARY','BLD','COOKFUEL','DIVISION','FIREPLACE','HEATFUEL','HEATTYPE','HHRACE','HOTWATER',\
              'NUMELDERS','NUMYNGKIDS','NUMOLDKIDS','OMB13CBSA','SOLAR','TENURE','UNITSIZE','YRBUILT',\
                  'HINCP','OTHERAMT','OILAMT','GASAMT','ELECAMT','NUMPEOPLE']
    
data = raw_data[predictors]

# str_cols = ['ACPRIMARY','ADEQUACY','BLD','CONDO','COOKFUEL','DISHH','DIVISION','FIREPLACE',\
#               'HEATFUEL','HEATTYPE','HHCITSHP','HHGRAD','HHRACE','HOTWATER','HUDSUB',\
#                   'MILHH','NUMCARE','NUMERRND','OMB13CBSA','RENTCNTRL','SOLAR','TENURE','UNITSIZE']

str_cols = ['ACPRIMARY','BLD','COOKFUEL','DIVISION','FIREPLACE','HEATFUEL','HEATTYPE','HHRACE','HOTWATER',\
                'OMB13CBSA','SOLAR','TENURE','UNITSIZE']
    
for col in str_cols:
    try:
        for i in range(len(data)):
            data.loc[i,col] = int(data.loc[i,col].strip("\'"))
    except:
        print('Data already converted.')
    print(col,'column complete.')


#replace -6s (not applicable) and -9s (not available) with nan
data.replace(-6, np.nan, inplace=True)
data.replace(-9, np.nan, inplace=True)

# drop rows where income and/or electricity bill are not reported or n/a,
# and where electricity bill is included with rent or other bill (unable to calculated burden)
data.ELECAMT.replace(2,np.nan,inplace=True)
data.OTHERAMT.replace(2,np.nan,inplace=True)
data.OILAMT.replace(2,np.nan,inplace=True)
data.GASAMT.replace(2,np.nan,inplace=True)
data = data.dropna(subset=['HINCP','ELECAMT','OTHERAMT','OILAMT','GASAMT'])

### Cleaning columns for needed variables
#replace 2s with 0s in CONDO, DISHH, RENTCNTRL, SOLAR (change binary coding from 1/2 to 0/1 with 0 being No)
#data.CONDO.replace(2,0,inplace=True)
#data.DISHH.replace(2,0,inplace=True)
#data.RENTCNTRL.replace(2,0,inplace=True)
data.SOLAR.replace(2,0,inplace=True)
#replace 5s with 0s in COOKFUEL (0 being no cooking fuel)
data.COOKFUEL.replace(5,0,inplace=True)
#replace ELECAMT nan's and (1,2,3) with 0
data.ELECAMT.replace([1,3],0,inplace=True)
data.OILAMT.replace([1,3],0,inplace=True)
data.GASAMT.replace([1,3],0,inplace=True)
data.OTHERAMT.replace([1,3],0,inplace=True)
#HUDSUB replace 3 and nan with 0, and replace 2 with 1
#data.HUDSUB.replace([3,np.nan],0,inplace=True)
#data.HUDSUB.replace(2,1,inplace=True)
#FIREPLACE replace 4's with 0's and (1,2,3) with 1
data.FIREPLACE.replace([2,3],1,inplace=True)
data.FIREPLACE.replace(4,0,inplace=True)
#MILHH replace 6 with 0's (no one in military) and (1,2,3,4,5) with 1 (at least one person active or veteran)
#data.MILHH.replace(6,0,inplace=True)
#data.MILHH.replace([2,3,4,5],1,inplace=True)
#NUMCARE replace (2,3) with 1 (at least one person with this disability) and 1's with 0's (no one has it)
#data.NUMCARE.replace(1,0,inplace=True)
#data.NUMCARE.replace([2,3],1,inplace=True)
#NUMERRND replace (2,3) with 1 (at least one person with difficulty doing errands) and 1's with 0's (no one has it)
#data.NUMERRND.replace(1,0,inplace=True)
#data.NUMERRND.replace([2,3],1,inplace=True)
#TENURE replace 3's with 0's (occupied without payment)
data.TENURE.replace(3,0,inplace=True)
#OMB13CBSA replace 99999 with rural (0) and all others with urban (1)
data['METRO'] = data.OMB13CBSA.values.tolist()
data.OMB13CBSA.replace(99999,0,inplace=True)
data.loc[data.OMB13CBSA > 1, 'OMB13CBSA'] = 1
data = data.rename(columns={'OMB13CBSA':'URBAN'})
#ACPRIMARY replace 12 (no AC) with 0
data.ACPRIMARY.replace(12,0,inplace=True)
#HEATFUEL replace 10 (no heat) with 0
data.HEATFUEL.replace(10,0,inplace=True)
#HEATTYPE replace 13 (no heat) with 0 and 14 with 13 to move next value up
data.HEATTYPE.replace(13,0,inplace=True)
data.HEATTYPE.replace(14,13,inplace=True)
#HOTWATER replace 7 (no hot running water) with 0
data.HOTWATER.replace(7,0,inplace=True)

data.to_csv('cleaned_data.csv')


######################################

# numerical_cols = ['BLD','NUMELDERS','NUMYNGKIDS','NUMOLDKIDS','UNITSIZE','YRBUILT']
    
# corr_cols = data[numerical_cols]
# corr = corr_cols.corr()
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask)] = True

# # Construct a heatmap of the correlation matrix
# # Note "annot = True" shows the correlation coefficient in each cell
# fig, ax = plt.subplots(figsize=(15,10))
# sns.heatmap(corr, ax=ax,
#         xticklabels=corr.columns,
#         yticklabels=corr.columns,
#         mask = mask, cmap="YlGnBu",
#         annot=True)
# plt.savefig('num_corr_matrix.png',dpi=300)



cleaned_data = pd.read_csv('cleaned_data.csv')

count=[]
for col in cleaned_data.columns:
    count.append(cleaned_data[col].isna().sum())
    
df = pd.DataFrame(count, index=cleaned_data.columns)

cleaned_data = cleaned_data.dropna(subset=['SOLAR','UNITSIZE'])

count2=[]
for col in cleaned_data.columns:
    count2.append(cleaned_data[col].isna().sum())
    
df2 = pd.DataFrame(count2, index=cleaned_data.columns)

cleaned_data['BURDEN'] = ((cleaned_data.ELECAMT+\
                            cleaned_data.OILAMT+\
                            cleaned_data.GASAMT+\
                            cleaned_data.OTHERAMT)\
                            *12)/cleaned_data.HINCP

cleaned_data.to_csv('cleaned_data_v2.csv')

###############################################################################


from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('cleaned_data_v2.csv').drop(columns=['Unnamed: 0','Unnamed: 0.1'])

data.drop(data.loc[data['BLD']==10].index, inplace=True)
data = data.reset_index()

data.ACPRIMARY.replace(4,3,inplace=True)
data.ACPRIMARY.replace(5,4,inplace=True)
data.ACPRIMARY.replace(6,5,inplace=True)
data.ACPRIMARY.replace(7,6,inplace=True)
data.ACPRIMARY.replace([8,9,10,11],7,inplace=True)
data.HHRACE.replace([7,8,9,10,11,12,13,14,15,18,20],6,inplace=True)

data.drop(columns='index').to_csv('final_cleaned_data.csv')

# use one hot encoding to convert categorical variables into series of binary variables
non_binary_cat_vars = ['ACPRIMARY','BLD','COOKFUEL','DIVISION','HEATFUEL','HEATTYPE','HHRACE','HOTWATER','TENURE']

cat_df = data[non_binary_cat_vars].copy(deep=True)

enc = OneHotEncoder()
enc.fit(cat_df)
onehotlabels = enc.transform(cat_df).toarray()
one_hot_df = pd.DataFrame(onehotlabels, columns=enc.get_feature_names_out())

# recombine dataframes
final_data = pd.concat([one_hot_df,data.drop(columns=non_binary_cat_vars)], axis=1)

final_data.to_csv('final_cleaned_dummy_data.csv')
