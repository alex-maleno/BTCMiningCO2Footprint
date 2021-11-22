#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 20:35:32 2021

@author: jake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Year to plot for full-year time series (string for datetime indexing)
year = '2020'

# Eastern data
car = pd.read_csv('net_gen/by_region/Carolinas_all_gen.csv',index_col=0, parse_dates=True)
cen = pd.read_csv('net_gen/by_region/Central_all_gen.csv',index_col=0, parse_dates=True)
flo = pd.read_csv('net_gen/by_region/Florida_all_gen.csv',index_col=0, parse_dates=True)
matl = pd.read_csv('net_gen/by_region/Mid-Atlantic_all_gen.csv',index_col=0, parse_dates=True)
midw = pd.read_csv('net_gen/by_region/Midwest_all_gen.csv',index_col=0, parse_dates=True)
neng = pd.read_csv('net_gen/by_region/New_England_all_gen.csv',index_col=0, parse_dates=True)
ny = pd.read_csv('net_gen/by_region/New_York_all_gen.csv',index_col=0, parse_dates=True)
se = pd.read_csv('net_gen/by_region/Southeast_all_gen.csv',index_col=0, parse_dates=True)
ten = pd.read_csv('net_gen/by_region/Tennessee_all_gen.csv',index_col=0, parse_dates=True)
# Western data
ca = pd.read_csv('net_gen/by_region/California_all_gen.csv',index_col=0, parse_dates=True)
nw = pd.read_csv('net_gen/by_region/Northwest_all_gen.csv',index_col=0, parse_dates=True)
sw = pd.read_csv('net_gen/by_region/Southwest_all_gen.csv',index_col=0, parse_dates=True)
# ERCOT data
tex = pd.read_csv('net_gen/by_region/Texas_all_gen.csv',index_col=0, parse_dates=True)
# US data
usa = pd.read_csv('net_gen/by_region/United_States_Lower_48_all_gen.csv',index_col=0, parse_dates=True)

# Convert regional data to interconnection level (western and eastern)
times = usa.index.to_list()
est, pst = [], []
for i in range(len(times)):
    est.append(times[i]-datetime.timedelta(hours=5))
    pst.append(times[i]-datetime.timedelta(hours=8))
western = pd.DataFrame(index=usa.index, columns=usa.columns)
eastern = pd.DataFrame(index=usa.index, columns=usa.columns)
cols = western.columns[1:].tolist()
for col in cols:
    western[col]=ca[col]+nw[col]+sw[col]
    eastern[col]=car[col]+cen[col]+flo[col]+matl[col]+midw[col]+neng[col]+ny[col]+se[col]+ten[col]
western['local_time']=pst
eastern['local_time']=est

# create dataframes for each energy source to plot
regions = ['east','west','texas','usa']
gen_dfs = [eastern,western,tex,usa]
final_wind, final_solar, final_petroleum, final_other, final_nuclear, final_natural_gas, final_hydro, final_coal = [pd.DataFrame(columns=regions) for i in range(8)]
gen_list = [final_wind, final_solar, final_petroleum, final_other, final_nuclear,  final_natural_gas, final_hydro, final_coal]
for i in range(len(gen_list)):
    for r in range(len(regions)):
        gen_list[i][regions[r]] = gen_dfs[r][cols[i]][year].values.tolist()
        
        
# First Figure: 4-panel stacked area plots of a single year of generation (2020)
# Panels are east, west, ERCOT, and total US

roll_wind, roll_solar, roll_petroleum, roll_other, roll_nuclear, roll_natural_gas, roll_hydro, roll_coal = [[np.nan,np.nan,np.nan,np.nan] for i in range(8)]
for i in range(4):
    roll_wind[i] = final_wind.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
    roll_solar[i] = final_solar.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
    roll_petroleum[i] = final_petroleum.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
    roll_other[i] = final_other.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
    roll_nuclear[i] = final_nuclear.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
    roll_natural_gas[i] = final_natural_gas.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
    roll_hydro[i] = final_hydro.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
    roll_coal[i] = final_coal.iloc[:,i].rolling(24,min_periods=1).mean().tolist()

### CREATE FIGURE ###
plt.rcParams.update(plt.rcParamsDefault)
fig = plt.figure(constrained_layout=False, figsize = (14,10))
#set constrained_layout=False and use wspace and hspace params to set amount of width/height reserved for space between subplots
gs = fig.add_gridspec(4,1, height_ratios = [1,1,1,1], hspace=.32)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])
axlist = [ax1,ax2,ax3,ax4]
lw = 6
plt.style.use('seaborn-white')
ax4.plot([],[],color='orange', label='Solar', linewidth=lw,alpha=1)
ax4.plot([],[],color='c', label='Wind', linewidth=lw,alpha=1)
ax4.plot([],[],color='b', label='Hydro', linewidth=lw,alpha=1)
ax4.plot([],[],color='m', label='Natural Gas', linewidth=lw,alpha=1)
ax4.plot([],[],color='k', label='Coal', linewidth=lw,alpha=1)
ax4.plot([],[],color='r', label='Petroleum', linewidth=lw,alpha=1)
ax4.plot([],[],color='green', label='Nuclear', linewidth=lw,alpha=1)
ax4.plot([],[],color='grey', label='Other', linewidth=lw,alpha=1)
ax4.legend(loc='center', bbox_to_anchor=(.5,-0.55), ncol=4, prop=dict(weight='bold', size=16))
# Daily rolling mean data plot
ax1.stackplot(range(8784),roll_other[0],roll_nuclear[0],roll_petroleum[0],roll_coal[0],roll_natural_gas[0],roll_hydro[0],roll_wind[0],roll_solar[0],\
                colors=['grey','green','r','k','m','b','c','orange'], alpha=1)
ax2.stackplot(range(8784),roll_other[1],roll_nuclear[1],roll_petroleum[1],roll_coal[1],roll_natural_gas[1],roll_hydro[1],roll_wind[1],roll_solar[1],\
                colors=['grey','green','r','k','m','b','c','orange'], alpha=1)
ax3.stackplot(range(8784),roll_other[2],roll_nuclear[2],roll_petroleum[2],roll_coal[2],roll_natural_gas[2],roll_hydro[2],roll_wind[2],roll_solar[2],\
                colors=['grey','green','r','k','m','b','c','orange'], alpha=1)
ax4.stackplot(range(8784),roll_other[3],roll_nuclear[3],roll_petroleum[3],roll_coal[3],roll_natural_gas[3],roll_hydro[3],roll_wind[3],roll_solar[3],\
                colors=['grey','green','r','k','m','b','c','orange'], alpha=1)
lfs=13
fy = ['Eastern Interconnection','Western Interconnection','ERCOT (Texas)','United States Lower 48']
ax1.set_yticks([0,100000,200000,300000,400000])
ax1.set_yticklabels(['0','100','200','300','400'], fontsize=lfs, fontweight='bold')
ax2.set_yticks([0,25000,50000,75000,100000])
ax2.set_yticklabels(['0','25','50','75','100'], fontsize=lfs, fontweight='bold')
ax3.set_yticks([0,20000,40000,60000])
ax3.set_yticklabels(['0','20','40','60'], fontsize=lfs, fontweight='bold')
ax4.set_yticks([0,100000,200000,300000,400000,500000])
ax4.set_yticklabels(['0','100','200','300','400','500'], fontsize=lfs, fontweight='bold')
for a in range(4):
    axlist[a].set_xlim(0,8784)
    axlist[a].set_xticks([0,744,1416,2160,2880,3624,4344,5088,5832,6552,7296,8016,8784])
    axlist[a].set_xticklabels(['Jan 1','Feb 1','Mar 1','Apr 1','May 1','Jun 1','Jul 1','Aug 1','Sep 1','Oct 1','Nov 1','Dec 1','Dec 31'], fontsize=lfs, fontweight='bold', rotation =15)
    axlist[a].set_ylabel('Demand (GW)', fontsize=14, fontweight='bold')
    axlist[a].annotate('{}'.format(str(fy[a])),(1300,axlist[a].get_ylim()[1]*.85),fontsize=14, fontweight='bold')

fig.suptitle('Stacked Area Plot for 2020',fontsize=20, fontweight='bold',y=0.916)
plt.savefig('figs/stacked_area_2020.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.rcParams.update(plt.rcParamsDefault)



### Next Figure: Focus on one week in WECC in August (9th?)
fig, ax = plt.subplots(constrained_layout=False, figsize = (6,6))
plt.style.use('seaborn-white')
ax.plot([],[],color='orange', label='Solar', linewidth=lw,alpha=1)
ax.plot([],[],color='c', label='Wind', linewidth=lw,alpha=1)
ax.plot([],[],color='b', label='Hydro', linewidth=lw,alpha=1)
ax.plot([],[],color='m', label='Natural Gas', linewidth=lw,alpha=1)
ax.plot([],[],color='k', label='Coal', linewidth=lw,alpha=1)
ax.plot([],[],color='r', label='Petroleum', linewidth=lw,alpha=1)
ax.plot([],[],color='green', label='Nuclear', linewidth=lw,alpha=1)
ax.plot([],[],color='grey', label='Other', linewidth=lw,alpha=1)
ax.legend(loc='center', bbox_to_anchor=(1.2,.55), prop=dict(weight='bold', size=12))
# ax.stackplot(range(5280,5448),final_other.iloc[:,0][5280:5448],final_nuclear.iloc[:,0][5280:5448],final_petroleum.iloc[:,0][5280:5448],final_coal.iloc[:,0][5280:5448],final_natural_gas.iloc[:,0][5280:5448],final_hydro.iloc[:,0][5280:5448],final_wind.iloc[:,0][5280:5448],final_solar.iloc[:,0][5280:5448],\
#                 colors=['grey','green','r','k','m','b','c','orange'], alpha=1)
ax.stackplot(range(5280,5448),final_other.iloc[:,1][5280:5448],final_nuclear.iloc[:,1][5280:5448],final_petroleum.iloc[:,1][5280:5448],final_coal.iloc[:,1][5280:5448],final_natural_gas.iloc[:,1][5280:5448],final_hydro.iloc[:,1][5280:5448],final_wind.iloc[:,1][5280:5448],final_solar.iloc[:,1][5280:5448],\
                colors=['grey','green','r','k','m','b','c','orange'], alpha=1)
# ax.stackplot(range(5280,5448),final_other.iloc[:,2][5280:5448],final_nuclear.iloc[:,2][5280:5448],final_petroleum.iloc[:,2][5280:5448],final_coal.iloc[:,2][5280:5448],final_natural_gas.iloc[:,2][5280:5448],final_hydro.iloc[:,2][5280:5448],final_wind.iloc[:,2][5280:5448],final_solar.iloc[:,2][5280:5448],\
#                 colors=['grey','green','r','k','m','b','c','orange'], alpha=1)
# ax.stackplot(range(5280,5448),final_other.iloc[:,3][5280:5448],final_nuclear.iloc[:,3][5280:5448],final_petroleum.iloc[:,3][5280:5448],final_coal.iloc[:,3][5280:5448],final_natural_gas.iloc[:,3][5280:5448],final_hydro.iloc[:,3][5280:5448],final_wind.iloc[:,3][5280:5448],final_solar.iloc[:,3][5280:5448],\
#                 colors=['grey','green','r','k','m','b','c','orange'], alpha=1)
ax.set_xlim(5280,5447)
ax.set_ylim(0,135000)
ax.set_title('Stacked Area Plot - Western Interconnection - 2020', fontweight='bold', fontsize=16)
ax.set_yticks([0,25000,50000,75000,100000,125000])
ax.set_yticklabels(['0','25','50','75','100','125'], fontsize=10, fontweight='bold')
ax.set_xticks([5280,5304,5328,5352,5376,5400,5424,5447])
ax.set_xticklabels(['August 9th','August 10th','August 11th','August 12th','August 13th','August 14th','August 15th','August 16th'], fontweight='bold',fontsize=10, rotation=30)
ax.set_ylabel('Demand (GW)', fontsize=14, fontweight='bold')
plt.savefig('figs/stacked_area_week.png', bbox_inches='tight', dpi=300)