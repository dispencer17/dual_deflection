# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:16:41 2022

@author: dispe
"""

import FEA_plot_functions
from FEA_plot_functions import *
from scipy import optimize
from scipy import odr
import numpy as np
import pandas as pd
import re
import os
from pandas.core.frame import DataFrame
from converters import UnitConverters
import config
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import matplotlib
import seaborn as sns



def main():
  
  
  #plot style
  sns.set_style('ticks')
  plt.rcParams.update({"xtick.bottom" : True, "ytick.left" : True})
  xlabel1 = 'Distance from Probe ' + u'(\u03bcm)'
  xlabel2 = 'Distance along Probe ' + u'(\u03bcm)'
  ylabel = 'Z Strain'
  legends = ['CNT 3.9 MPa', 'CNT 72 MPa', 'CNT 1700 MPa', 'Si 1650 MPa']
  
  
  feaData, directory = grabData(config.fea_data_file)
  # x,y = getXY(feaData, 'ALP', 'Low', 'L')
  # plt.plot(x,y)
  X1, Y1, DyDx1 = plotAll(feaData, oneGraph = True, oneSide = '_R', legend = True, mLoc=['Tip'], modulus=['Low','Med', 'Hi', 'Si'], 
                          xlabel=xlabel1, ylabel=ylabel, legends=legends)
  # plotAll(feaData, oneGraph = True, oneSide = '_L', legend = True, mLoc=['Middle'], modulus=['Low','Med', 'Hi', 'Si'], points_to_drop=3)
  # plotAll(feaData, oneGraph = True, oneSide = '_L', legend = True, mLoc=['Surface'], modulus=['Low','Med', 'Hi', 'Si'])
  # plotAll(feaData, oneGraph = True, oneSide = '_L', legend = True, mLoc=['APL'], modulus=['Low','Med', 'Hi', 'Si'], points_to_drop=3)
  X2, Y2, DyDx2 =  plotAll(feaData, oneGraph = True, oneSide = '_R', legend = True, mLoc=['APL'], modulus=['Low','Med', 'Hi', 'Si'], 
                           points_to_drop=2, points_to_truncate=2, xlabel=xlabel2, ylabel=ylabel, legends=legends)
  pos_slope = np.greater_equal(DyDx2, 0)
  ful_point = np.greater_equal(Y2, 0)
  max_percent_after_fp = []
  # plt.figure()
  k = 0
  dy2_diff = np.abs(np.array(DyDx2)-k)
  for i in range(len(X2)):
    # plt.plot(X2[i][:-1], DyDx2[i])
    pos_slope_index = np.where(pos_slope[i] == False)
    fp_index = np.where(ful_point[i] == True)
    zero_slope_index = np.argmin(dy2_diff[i])
    
    max_point_after_fp = np.max(np.abs(Y2[i][fp_index[0][0]:]))
    max_point_index_fp= np.argmax(np.abs(Y2[i][fp_index[0][0]:])) + fp_index[0][0]
    max_point = np.max(np.abs(Y2[i]))
    max_point_index = np.argmax(np.abs(Y2[i]))
    max_percent_after_fp.append(max_point_after_fp/max_point)
    
    if np.size(pos_slope_index) > 0: 
      plt.plot(X2[i][pos_slope_index[0][0]], Y2[i][pos_slope_index[0][0]], 'ko')
    plt.plot(X2[i][fp_index[0][0]], Y2[i][fp_index[0][0]], 'mo')
    plt.plot(X2[i][max_point_index_fp], Y2[i][max_point_index_fp], 'g.')
    # plt.plot(X2[i][index], Y2[i][index], 'm*')
  # plt.xlabel('Microns')
  # plt.ylabel('Z Strain')
  
  print('stop')
main()