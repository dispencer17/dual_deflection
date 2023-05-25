# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:20:58 2022

@author: dispe
"""


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
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

def grabData(excelFileName = ''):
  directory = config.data_directory + "/" + config.metadata_directory
  filename = config.fea_data_file + ".xlsx"
  if excelFileName != '':
    filename = excelFileName + ".xlsx"

  feaData = pd.read_excel(directory + "/" + filename, header=0)
  feaData = feaData.sort_index()  
   
  return feaData, directory

def getXY(data, MSite, modulus, side):
  x_parameters = [MSite, 'dis', side]
  y_parameters = [MSite, modulus, side]
  x_address = '_'.join(x_parameters)
  y_address = '_'.join(y_parameters)
  x = data[x_address]
  y = data[y_address]
  return x,y

def plotAll(data, oneGraph=False, oneSide='_L', legend=False,  mLoc='All', modulus='All', legend_loc = 'upper left', 
            points_to_drop = 0, points_to_truncate=0, xlabel = '', ylabel = '', legends = []):
  """
  Plots all of the FEA data included in the spreadsheet. See config.py for directory and file name. 
  Parameters allow selective plotting. 

  Parameters
  ----------
  data : Pandas DataFrame
    THE DATAFRAME OF THE FEA DATA EXCEL SPREADSHEET.
    
  oneGraph : Bool, optional
    TRUE IF YOU WANT ALL THE DATA ON ONE GRAPH. FALSE IF YOU WANT ONE LINE PER GRAPH. The default is False.
    
  oneSide : string, optional
    '_L' WILL SHOW DATA FROM LEFT SIDE, '_R' WILL SHOW DATA FROM RIGHT SIDE, OTHERWISE WILL SHOW RIGHT AND LEFT. The default is '_L'.
    
  legend : Bool, optional
    IF TRUE, A LEGEND WILL BE PLACED NEXT TO THE GRAPH. The default is False.
       
  mLoc : string or array of strings, optional
    MEASUREMENT LOCATION. OPTIONS ARE AN ARRAY OF 'Tip', 'Middle', 'Surface', AND 'APL', IN ANY ORDER. The default is 'All'.
    AN ARRAY OF ALL MODULI OPTIONS AND THE DEFAULT 'All' WILL PRODUCE THE SAME RESULT.
      
  modulus : string or array of strings, optional
    MODULUS OF MATERIAL. OPTIONS ARE AN ARRAY OF 'Low', 'Med', 'Hi', AND 'Si', IN ANY ORDER. The default is 'All'.
    AN ARRAY OF ALL MODULI OPTIONS AND THE DEFAULT 'All' WILL PRODUCE THE SAME RESULT.
    
  legend_loc : string
    THE LOCATION OF THE LEGEND. INPUT FOR THE 'loc' PARAMETER IN plt.legend. 
    OPTIONS ARE 'upper center', 'lower center', 'center left', 'center right', 'center', 'lower left', 'lower right', 'lower center', 'upper center', and 'best'
    The default is 'upper left'.
    
  points_to_drop : int
    THE NUMBER OF POINTS TO DROP AT THE BEGINNING OF THE GRAPH. DOES NOT NORMALIZE DISTANCES. The default is 0.
    
  points_to_truncate : int
    THE NUMBER OF POINTS TO DROP AT THE END OF THE GRAPH. DOES NOT NORMALIZE DISTANCES. The default is 0.  
    
  xlabel : string
    THE LABEL OF THE X AXIS. IF STRING IS BLANK, NO LABEL IS ADDED. The default is ''.
    
  ylabel : string
    THE LABEL OF THE Y AXIS. IF STRING IS BLANK, NO LABEL IS ADDED. The default is ''.
    
  legends : list
    THE HANDLES OF THE LEGEND. The default is []. 

  Returns
  -------
  None.

  """
  keys = data.keys()
  correct_keys = np.ones(np.shape(keys)).astype(bool)
  if mLoc != 'All':
    correct_keys_temp = np.zeros(np.shape(keys)).astype(bool)
    for l in mLoc:
      loc_keys = keys.str.contains(l)
      correct_keys_temp = correct_keys_temp + loc_keys
    correct_keys = correct_keys_temp
    keys = keys[correct_keys]
  if modulus != 'All':
    correct_keys_temp = np.zeros(np.shape(keys)).astype(bool)
    for m in modulus:
      modulus_keys = keys.str.contains(m)
      correct_keys_temp = correct_keys_temp + modulus_keys
    correct_keys = correct_keys_temp
  dis_keys = keys.str.contains('dis')
  correct_keys = correct_keys + dis_keys
  keys = keys[correct_keys]
  dis_keys = keys.str.contains('dis')
  dis_keys_index = np.where(dis_keys)[0]
  n = dis_keys_index[1]-dis_keys_index[0]
  if oneSide == '_L' or oneSide == '_R':
    correct_keys = keys.str.endswith(oneSide)
    keys = keys[correct_keys]
  X = []
  Y = []
  DyDx = []
  q = 1
  y_addresses = []
  ns = np.arange(n)
  # if oneGraph:
  #   fig, ax = plt.figure()

  r = int(keys.size)
  for i in range(0, r, n):
    x_address = keys[i]
    for j in ns:
      if j == 0:
        continue
      y_address = keys[i+j]
      y_addresses.append(y_address)
      x = data[x_address]
      y = data[y_address]
      if points_to_drop != 0:
        x = x.drop(x.index[0:points_to_drop])
        y = y.drop(y.index[0:points_to_drop])
      if points_to_truncate != 0:
        x = x.drop(x.index[-points_to_truncate:])
        y = y.drop(y.index[-points_to_truncate:])
      if not oneGraph:
        plt.figure(figsize=(10,6))
        plt.title(y_address)
      if oneGraph and q == 1:
        plt.figure(figsize=(10,6))
        q = -1
      if y_address.__contains__('Si'):
        plt.plot(x, y, '--', label=legends[j-1])
      else:
        plt.plot(x, y, label=legends[j-1])
      if mLoc[0] == 'APL':
        plt.xlim(0,1000)
      X.append(x.reset_index(drop = True))
      Y.append(y.reset_index(drop = True)) 
      DyDx.append(np.diff(y)/np.diff(x))
      # plt.plot(x[:-1], DyDx[-1], label='slope')
  if xlabel != '':
    plt.xlabel(xlabel, fontsize=config.axisFontSize)
  if ylabel != '':
    plt.ylabel(ylabel, fontsize=config.axisFontSize)
  plt.rc('xtick', labelsize=config.axisFontSize) 
  plt.rc('ytick', labelsize=config.axisFontSize) 
  if legend:
    plt.legend(legends)
    plt.legend(fontsize=12)
    # plt.legend(bbox_to_anchor=(1.01, 1), loc=legend_loc, borderaxespad=0)
  plt.tight_layout()
  return X, Y, DyDx
    