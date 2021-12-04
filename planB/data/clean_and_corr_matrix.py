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

predictors = ['ACPRIMARY','ADEQUACY','BLD','CONDO','COOKFUEL','DISHH','DIVISION','ELECAMT','FIREPLACE',\
              'HEATFUEL','HEATTYPE','HHAGE','HHCITSHP','HHGRAD','HHMOVE','HHRACE','HINCP',\
                  'HOTWATER','HUDSUB','MAINTAMT','MARKETVAL','MILHH','NUMCARE','NUMELDERS','NUMERRND',\
                      'NUMPEOPLE','NUMYNGKIDS','NUMOLDKIDS','OMB13CBSA','PERPOVLVL','RATINGNH','RATINGHS',\
                          'RENTCNTRL','SOLAR','STORIES','TENURE','TOTROOMS','UNITSIZE','WATERAMT','YRBUILT']
    
data = raw_data[predictors]
#data.replace("'-6'",np.nan, inplace=True)
#data.replace("'-9'",np.nan, inplace=True)

str_cols = ['ACPRIMARY','ADEQUACY','BLD','CONDO','COOKFUEL','DISHH','DIVISION','FIREPLACE',\
              'HEATFUEL','HEATTYPE','HHCITSHP','HHGRAD','HHRACE','HOTWATER','HUDSUB',\
                  'MILHH','NUMCARE','NUMERRND','OMB13CBSA','RENTCNTRL','SOLAR','TENURE','UNITSIZE']
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


### Cleaning columns for needed variables
#replace 2s with 0s in CONDO, DISHH, RENTCNTRL, SOLAR (change binary coding from 1/2 to 0/1 with 0 being No)
data.CONDO.replace(2,0,inplace=True)
data.DISHH.replace(2,0,inplace=True)
data.RENTCNTRL.replace(2,0,inplace=True)
data.SOLAR.replace(2,0,inplace=True)
#replace 5s with 0s in COOKFUEL (0 being no cooking fuel)
data.COOKFUEL.replace(5,0,inplace=True)
#replace ELECAMT nan's and (1,2,3) with 0
data.ELECAMT.replace([1,2,3,np.nan],0,inplace=True)
data.WATERAMT.replace([1,2,3,np.nan],0,inplace=True)
#HUDSUB replace 3 and nan with 0, and replace 2 with 1
data.HUDSUB.replace([3,np.nan],0,inplace=True)
data.HUDSUB.replace(2,1,inplace=True)
#FIREPLACE replace 4's with 0's and (1,2,3) with 1
data.FIREPLACE.replace([2,3],1,inplace=True)
data.FIREPLACE.replace(4,0,inplace=True)
#MILHH replace 6 with 0's (no one in military) and (1,2,3,4,5) with 1 (at least one person active or veteran)
data.MILHH.replace(6,0,inplace=True)
data.MILHH.replace([2,3,4,5],1,inplace=True)
#NUMCARE replace (2,3) with 1 (at least one person with this disability) and 1's with 0's (no one has it)
data.NUMCARE.replace(1,0,inplace=True)
data.NUMCARE.replace([2,3],1,inplace=True)
#NUMERRND replace (2,3) with 1 (at least one person with difficulty doing errands) and 1's with 0's (no one has it)
data.NUMERRND.replace(1,0,inplace=True)
data.NUMERRND.replace([2,3],1,inplace=True)
#TENURE replace 3's with 0's (occupied without payment)
data.TENURE.replace(3,0,inplace=True)
#OMB13CBSA replace 99999 with rural (0) and all others with urban (1)
data.OMB13CBSA.replace(99999,0,inplace=True)
data.loc[data.OMB13CBSA > 1, 'OMB13CBSA'] = 1

data.to_csv('cleaned_data.csv')


######################################

numerical_cols = ['BLD','ELECAMT','HHAGE','HHGRAD','HHMOVE','HINCP','MAINTAMT','MARKETVAL','NUMELDERS','NUMPEOPLE','NUMYNGKIDS',\
                      'NUMOLDKIDS','PERPOVLVL','RATINGNH','RATINGHS','STORIES','TOTROOMS','UNITSIZE','WATERAMT','YRBUILT']
    
corr_cols = data[numerical_cols]
corr = corr_cols.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

# Construct a heatmap of the correlation matrix
# Note "annot = True" shows the correlation coefficient in each cell
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(corr, ax=ax,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        mask = mask, cmap="YlGnBu",
        annot=True)
plt.savefig('num_corr_matrix.png',dpi=300)



