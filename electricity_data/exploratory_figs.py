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

# roll_wind, roll_solar, roll_petroleum, roll_other, roll_nuclear, roll_natural_gas, roll_hydro, roll_coal = [[np.nan,np.nan,np.nan,np.nan] for i in range(8)]
# for i in range(7):
#     roll_must_run[i] = final_must_run.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
#     roll_imports[i] = final_imports.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
#     roll_fossil[i] = final_fossil.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
#     roll_hydro[i] = final_hydro.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
#     roll_wind[i] = final_wind.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
#     roll_solar[i] = final_solar.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
#     roll_bat_dis[i] = final_bat_dis.iloc[:,i].rolling(24,min_periods=1).mean().tolist()
#     roll_load[i] = final_load.iloc[:,i].rolling(24,min_periods=1).mean().tolist()

# ### CREATE FIGURE ###
# plt.rcParams.update(plt.rcParamsDefault)
# fig = plt.figure(constrained_layout=False, figsize = (12,15))
# #set constrained_layout=False and use wspace and hspace params to set amount of width/height reserved for space between subplots
# gs = fig.add_gridspec(7,1, height_ratios = [1,1,1,1,1,1,1], hspace=.32)
# ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1])
# ax3 = fig.add_subplot(gs[2])
# ax4 = fig.add_subplot(gs[3])
# ax5 = fig.add_subplot(gs[4])
# ax6 = fig.add_subplot(gs[5])
# ax7 = fig.add_subplot(gs[6])
# axlist = [ax1,ax2,ax3,ax4,ax5,ax6,ax7]
# lw = 6
# plt.style.use('seaborn-white')
# ax7.plot([],[],color='m', label='Battery', linewidth=lw,alpha=1)
# ax7.plot([],[],color='orange', label='Solar', linewidth=lw,alpha=1)
# ax7.plot([],[],color='c', label='Wind', linewidth=lw,alpha=1)
# ax7.plot([],[],color='b', label='Hydro', linewidth=lw,alpha=1)
# ax7.plot([],[],color='r', label='Fossil', linewidth=lw,alpha=1)
# ax7.plot([],[],color='g', label='Imports', linewidth=lw,alpha=1)
# ax7.plot([],[],color='grey', label='Must run', linewidth=lw,alpha=1)
# ax7.legend(loc='center', bbox_to_anchor=(.5,-0.55), ncol=4, prop=dict(weight='bold', size=16))
# # Daily rolling mean data plot
# ax1.stackplot(range(8736),roll_must_run[0],roll_imports[0],roll_fossil[0],roll_hydro[0],roll_wind[0],roll_solar[0],roll_bat_dis[0],\
#                colors=['grey','g','r','b','c','orange','m'], alpha=1)
# ax2.stackplot(range(8736),roll_must_run[1],roll_imports[1],roll_fossil[1],roll_hydro[1],roll_wind[1],roll_solar[1],roll_bat_dis[1],\
#                colors=['grey','g','r','b','c','orange','m'], alpha=1)
# ax3.stackplot(range(8736),roll_must_run[2],roll_imports[2],roll_fossil[2],roll_hydro[2],roll_wind[2],roll_solar[2],roll_bat_dis[2],\
#                colors=['grey','g','r','b','c','orange','m'], alpha=1)
# ax4.stackplot(range(8736),roll_must_run[3],roll_imports[3],roll_fossil[3],roll_hydro[3],roll_wind[3],roll_solar[3],roll_bat_dis[3],\
#                colors=['grey','g','r','b','c','orange','m'], alpha=1)
# ax5.stackplot(range(8736),roll_must_run[4],roll_imports[4],roll_fossil[4],roll_hydro[4],roll_wind[4],roll_solar[4],roll_bat_dis[4],\
#                colors=['grey','g','r','b','c','orange','m'], alpha=1)
# ax6.stackplot(range(8736),roll_must_run[5],roll_imports[5],roll_fossil[5],roll_hydro[5],roll_wind[5],roll_solar[5],roll_bat_dis[5],\
#                colors=['grey','g','r','b','c','orange','m'], alpha=1)
# ax7.stackplot(range(8736),roll_must_run[6],roll_imports[6],roll_fossil[6],roll_hydro[6],roll_wind[6],roll_solar[6],roll_bat_dis[6],\
#                colors=['grey','g','r','b','c','orange','m'], alpha=1)
# lfs=13
# for a in range(7):
#     axlist[a].set_xlim(0,8735)
#     axlist[a].set_ylim(0,np.max([roll_load[i] for i in range(4)])*1.1)
#     axlist[a].set_xticks([0,744,1416,2160,2880,3624,4344,5088,5832,6552,7296,8016,8735])
#     axlist[a].set_xticklabels(['Jan 1','Feb 1','Mar 1','Apr 1','May 1','Jun 1','Jul 1','Aug 1','Sep 1','Oct 1','Nov 1','Dec 1','Dec 31'], fontsize=lfs, fontweight='bold', rotation =15)
#     axlist[a].set_yticks([0,10000,20000,30000,40000])
#     axlist[a].set_yticklabels(['0','10','20','30','40'], fontsize=lfs, fontweight='bold')
#     axlist[a].set_ylabel('Demand (GW)', fontsize=14, fontweight='bold')
#     axlist[a].annotate('#{} - 2050'.format(str(fy[a])),(100,np.max([roll_load[i] for i in range(4)])*.88),fontsize=20, fontweight='bold')

# fig.suptitle('Stacked Area Plot for 2020',fontsize=20, fontweight='bold',y=0.90)
# plt.savefig('figs/stacked_area_2020.png', bbox_inches='tight', dpi=300)
# plt.clf()
# plt.rcParams.update(plt.rcParamsDefault)
