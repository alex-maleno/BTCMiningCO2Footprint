#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 19:53:21 2021

@author: jake
"""

import pandas as pd
import numpy as np
import datetime

# define range of dates to extract
start_date = '2018-09-20'
end_date = '2021-11-18'
date_range = pd.date_range(start=start_date,end=end_date, freq='H')

# initialize lists of generation types and regions to loop through raw data files
gen_types = ['wind','solar','petroleum','other','nuclear','natural_gas','hydro','coal']
regions = ['California','Carolinas','Central','Florida','Mid-Atlantic','Midwest','New_England',\
           'New_York','Northwest','Southeast','Southwest','Tennessee','Texas','United_States_Lower_48']

# check earliest record for each dataset
#earliest_date = []

# list of empty dataframes to populate generation mix data with
regional_dfs = [pd.DataFrame(index=date_range,columns=['local_time']+gen_types) for i in range(len(regions))]

# two for loops for extracting files for each region and each generation type
for r in range(len(regions)):
    for g in gen_types:
        # not all regions have every generation type, so a try/except is used to avoid errors
        try:
            temp_df = pd.read_csv('net_gen/raw_data/{}/Net_generation_from_{}_for_{}_(region)_hourly_-_UTC_time.csv'\
                        .format(regions[r],g,regions[r]), skiprows=5, header=None).rename(columns={0:'Time',1:'MWh'})

            # convert time column to pandas datetime object for easier indexing
            temp_df['UTC'] = pd.to_datetime(temp_df['Time'])
            temp_df = temp_df.set_index('UTC')
#            earliest_date.append(temp_df.iloc[-1].name)
            
            for i in range(len(regional_dfs[r])):
                try:
                    regional_dfs[r].iloc[i][g] = temp_df[temp_df['Time']==date_range[i].strftime('%m/%-d/%Y %HH')]['MWh'].values.tolist()[0]
                except:
                    print('Time not found for {} {}:'.format(regions[r],g),date_range[i].strftime('%m/%d/%Y %HH'))
            
        # if the data csv is not found, set generation equal to zero    
        except:
            #regional_dfs[r][g] = 0
            print('Data not found for {} generation in {}'.format(g,regions[r]))
            
        print('{} {} complete.'.format(regions[r],g))
    
    # fill any missing values with zeroes
    regional_dfs[r] = regional_dfs[r].fillna(0)
    regional_dfs[r].index.name='utc_time'

#print('Latest Start Date to a Dataset:',np.max(earliest_date))

# regional_dfs now contains all generation data. just need to convert UTC to local time, which is approximate because
# regions do not coincide perfectly with time zones. Set US data "local time" to just be UTC time.
for r in range(len(regions)):
    times = regional_dfs[r].index.to_list()
    eastern, central, mountain, pacific = [], [], [], []
    for i in range(len(times)):
        eastern.append(times[i]-datetime.timedelta(hours=5))
        central.append(times[i]-datetime.timedelta(hours=6))
        mountain.append(times[i]-datetime.timedelta(hours=7))
        pacific.append(times[i]-datetime.timedelta(hours=8))
    if (regions[r] == 'California') or (regions[r] == 'Northwest'):
        regional_dfs[r]['local_time'] = pacific
    elif (regions[r] == 'Southwest'):
        regional_dfs[r]['local_time'] = mountain
    elif (regions[r] == 'Texas') or (regions[r] == 'Central') or (regions[r] == 'Midwest') or (regions[r] == 'Tennessee'):
        regional_dfs[r]['local_time'] = central
    elif (regions[r] == 'United_States_Lower_48'):
        regional_dfs[r]['local_time'] = times
    else:
        regional_dfs[r]['local_time'] = eastern
    
for r in range(len(regions)):
    regional_dfs[r] = regional_dfs[r].fillna(0)
    regional_dfs[r].to_csv('net_gen/by_region/{}_all_gen.csv'.format(regions[r]))    
    
###############################################################################

# create single csv file with all data
columns = ['{}_{}'.format(r,g) for r in regions for g in gen_types]
combined_df = pd.DataFrame(index=date_range, columns=columns)
for r in range(len(regions)):
    for g in gen_types:
        combined_df['{}_{}'.format(regions[r],g)] = regional_dfs[r][g].values.tolist()
combined_df = combined_df.fillna(0)
combined_df.to_csv('net_gen/combined_data.csv')

###############################################################################
    
# create "tidy" dataset with one entry per row
tidy = []
for r in range(len(regions)):
    if (regions[r] == 'California') or (regions[r] == 'Northwest') or (regions[r] == 'Southwest'):
        interconnection = 'western'
    elif (regions[r] == 'Texas'):
        interconnection = 'texas'
    elif (regions[r] == 'United_States_Lower_48'):
        interconnection = 'US'
    else:
        interconnection = 'eastern'
    for g in gen_types:
        for i in range(len(date_range)):
            tidy.append((regional_dfs[r].index[i],regional_dfs[r]['local_time'][i],regions[r],interconnection,g,regional_dfs[r][g][i]))
tidy_df = pd.DataFrame(tidy, columns = ['utc_time','local_time','region','interconnection','gen_type','MWh'])
tidy_df = tidy_df.set_index('utc_time')
tidy_df.to_csv('net_gen/tidy_net_gen_data.csv')












    