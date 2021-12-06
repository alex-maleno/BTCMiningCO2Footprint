#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 18:03:43 2021

@author: jake
"""

import pandas as pd

cleaned_data = pd.read_csv('cleaned_data.csv')

count=[]
for col in cleaned_data.columns:
    count.append(cleaned_data[col].isna().sum())
    
df = pd.DataFrame(count, index=cleaned_data.columns)

cleaned_data = cleaned_data.drop(columns=['RENTCNTRL'])
cleaned_data = cleaned_data.dropna(subset=['SOLAR','NUMCARE','NUMERRND','DISHH','RATINGHS','RATINGNH','UNITSIZE'])

count2=[]
for col in cleaned_data.columns:
    count2.append(cleaned_data[col].isna().sum())
    
df2 = pd.DataFrame(count2, index=cleaned_data.columns)

cleaned_data = cleaned_data.drop(columns=['MARKETVAL','MAINTAMT'])

cleaned_data.to_csv('cleaned_data_v2.csv')