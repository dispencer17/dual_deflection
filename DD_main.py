# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:08:27 2022

@author: dispe
"""


import DD_functions
from DD_functions import *
import scipy.stats as stats
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
from statistics import mean
from statistics import stdev

def main():
    
  #FIRST: Check the config file to make sure it matches your desired outputs and calculations
  
  #plot style
  sns.set_style('ticks')
  plt.rcParams.update({"xtick.bottom" : True, "ytick.left" : True})
  plt.rc('xtick', labelsize=config.axisFontSize) 
  plt.rc('ytick', labelsize=config.axisFontSize)
  #run program
  if config.readExcel:
    samples, sampleNamesString, directory = grabSamples(config.excelName, [])
  else:
    samples, sampleNamesString, directory = grabSamples('', config.sampleNames)
  #Set PDF name and opens pdf file to save all graphs in.
  if config.trackerFiles != "All posts":
    if config.saveEachPostPlot:
      pdfName = config.trackerFiles + f" Fitted Deflection Graphs at {config.truncationValue} microns.pdf"
    else:
      pdfName = config.trackerFiles + f" Overview Graphs at {config.truncationValue} microns.pdf"
    config.pdf = mpdf.PdfPages(directory + "/" + pdfName)
  else:
    if config.saveEachPostPlot:
      pdfName = sampleNamesString + f"Fitted Deflection Graphs at {config.truncationValue} microns.pdf"
    else:
      pdfName = sampleNamesString + f"Overview Graphs at {config.truncationValue} microns.pdf"
    config.pdf = mpdf.PdfPages(directory + "/" + pdfName)
  #Loop through sample objects and analyze each tested post  
  allPosts = []
  for sample in samples:
    posts = analyzePosts(sample)
    sample.setAvgModAndStats()
    # saveSampleLevelGraphs(posts)
    allPosts.extend(posts)
    
  xySampleData = {}
  xxyySampleData = {}
  #The dictionary of all the parameters you want in your output file.
  """ if config.numOfLineFits == 'Multi' or config.numOfLineFits == 'Both':
    data = {'Sample': [], 'Row': [], 'Post': [], 'Initial_Deflection_class': [], 'Failure_class': [],
          'After_failure_class': [], 'Modulus': [], 'Modulus_Uncertainty': [], 
          'Percent_Error': [], 'Slope_Percent_Error': [],
          'Ultimate_Strength': [], 'CNT_Diameter': [], 'Sample_Average_CNT_Diameter': [], 'Post_Length': [], 
          'Effective_Length': [], 'Wire_Effective_Length': [], 'Failure_Mode': [], 'Growth_Time': [], 'Infiltration_Time': [], 'min_samples_split': [],
          'min_samples_leaf': [], 'max_depth': [], 'max_bins': [], 'Slope_segment': [], 'Problem_Post': []}
    #Populate output data
    for p in allPosts:
      if p.modulus != 0:
        if config.classifyDeflections:
          p.curveClassification = userClassifyDeflections(p.getAddressString())
        data['Sample'].append(p.sample.name)
        data['Row'].append(p.row)
        data['Post'].append(p.post)
        data['Initial_Deflection_class'].append(p.curveClassification['Initial_deflection'])
        data['Failure_class'].append(p.curveClassification['Failure'])
        data['After_failure_class'].append(p.curveClassification['After_failure'])
        data['Modulus'].append(p.modulus)
        data['Modulus_Uncertainty'].append(p.modUncertainty)
        data['Percent_Error'].append(p.modUncertainty/p.modulus)
        data['Slope_Percent_Error'].append(p.slopeError/p.goodSlope)
        data['Ultimate_Strength'].append(p.ultStrength)
        data['CNT_Diameter'].append(p.avgCNTDiameter)
        data['Sample_Average_CNT_Diameter'].append(p.sample.avgCNTDiameter)
        data['Post_Length'].append(p.length)
        data['Effective_Length'].append(p.effectiveLength*10**6)
        data['Wire_Effective_Length'].append(p.wEffLength*10**6)
        data['Failure_Mode'].append(p.failureMode)
        data['Growth_Time'].append(p.sample.growthTime)
        data['Infiltration_Time'].append(p.sample.infiltrationTime)
        data['min_samples_split'].append(p.fitParameters[0])
        data['min_samples_leaf'].append(p.fitParameters[1])
        data['max_depth'].append(p.fitParameters[2])
        data['max_bins'].append(p.fitParameters[3])
        data['Slope_segment'].append(p.segment)
        data['Problem_Post'].append(p.problemPost)
        
        if not p.sample.name in xySampleData:
          xyData = {'Location': [], 'Post_deflection': [], 
                    'Wire_deflection': [], 'Force': [],'Infiltration_status': []}
          # xyData['Post_deflection'].extend(p.td.post.values.tolist())
          # xyData['Wire_deflection'].extend(p.tdWire.values.tolist())
          xyData['Post_deflection'].extend(p.td['tdPost'].to_numpy())
          xyData['Wire_deflection'].extend(p.td['tdWire'].to_numpy())
          xyData['Location'].extend([p.getRowPostString()]*len(p.td.post.values.tolist()))
          if p.sample.infiltrationTime == 0:
            xyData['Infiltration_status'].extend(['Uninfiltrated']*len(p.td.post.values.tolist()))
          else:
            xyData['Infiltration_status'].extend(['Infiltrated']*len(p.td.post.values.tolist()))
          xySampleData[p.sample.name] = xyData
        elif p.sample.name in xySampleData:
          # xyData['Post_deflection'].extend(p.td.post.values.tolist())
          # xyData['Wire_deflection'].extend(p.tdWire.values.tolist())
          xyData['Post_deflection'].extend(p.td['tdPost'].to_numpy())
          xyData['Wire_deflection'].extend(p.td['tdWire'].to_numpy())
          xyData['Location'].extend([p.getRowPostString()]*len(p.td.post.values.tolist()))
          if p.sample.infiltrationTime == 0:
            xyData['Infiltration_status'].extend(['Uninfiltrated']*len(p.td.post.values.tolist()))
          else:
            xyData['Infiltration_status'].extend(['Infiltrated']*len(p.td.post.values.tolist())) """
        
  if config.numOfLineFits == 'One':
    data = {'Sample': [], 'Row': [], 'Post': [], 'Modulus': [], 'Modulus_Uncertainty': [], 
          'Percent_Error': [], 'Slope': [], 'Slope_Percent_Error': [], 'Ultimate_Strength': [], 'CNT_Diameter': [],
          'Sample_Average_CNT_Diameter': [], 'Post_Length': [], 'Effective_Length': [], 'Wire_Effective_Length': [],
          'Wire_Stiffness': [], 'Failure_Mode': [], 'Growth_Time': [], 'Infiltration_Time': [], 'Slope_Initial_Guess': [], 
          'Intercept_Initial_Guess': [],'Initial_Points_Dropped': [], 'Total_Data_Points': [], 'Problem_Post': []}
    cntDiamsByInfTime = {0: [], 15: [], 30: []}
    
    #Populate output data
    for p in allPosts:
      if p.modulus != 0:
        if config.classifyDeflections:
          p.curveClassification = userClassifyDeflections(p.getAddressString())
        # if config.minimumDataPoints != 'none':
        #   if len(p.x) < config.minimumDataPoints:
        #     continue
        data['Sample'].append(p.sample.name)
        data['Row'].append(p.row)
        data['Post'].append(p.post)
        data['Modulus'].append(p.modulus)
        data['Modulus_Uncertainty'].append(p.modUncertainty)
        data['Percent_Error'].append(p.modUncertainty/p.modulus)
        data['Slope'].append(p.goodSlope)
        data['Slope_Percent_Error'].append(p.slopeError/p.goodSlope)
        data['Ultimate_Strength'].append(p.ultStrength)
        data['CNT_Diameter'].append(p.avgCNTDiameter)
        data['Sample_Average_CNT_Diameter'].append(p.sample.avgCNTDiameter)
        data['Post_Length'].append(p.length*10**6)
        data['Effective_Length'].append(p.effectiveLength*10**6)
        data['Wire_Effective_Length'].append(p.wEffLength*10**6)
        data['Wire_Stiffness'].append(p.wStiffness)
        data['Failure_Mode'].append(p.failureMode)
        data['Growth_Time'].append(p.sample.growthTime)
        data['Infiltration_Time'].append(p.sample.infiltrationTime)
        data['Slope_Initial_Guess'].append(p.initialGuesses[0])
        data['Intercept_Initial_Guess'].append(p.initialGuesses[1])
        data['Initial_Points_Dropped'].append(p.droppedPoints)
        data['Total_Data_Points'].append(len(p.x))
        data['Problem_Post'].append(p.problemPost)
        
        cntDiamsByInfTime[p.sample.infiltrationTime].extend(p.individualCNTDiameters) 
        
        if not p.sample.name in xySampleData:
          xyData = {'Location': [], 'Post_deflection': [],
                    'Wire_deflection': [], 'Force': [],'Infiltration_status': [],  'Sample': []}
          fitData = {'Location': [], 'xx': [], 
                    'yy': [], 'yyForce': [],'Infiltration_status': [],  'Sample': []}
          if config.offsetToZero and not config.plotFullDeflection:
            xyData['Post_deflection'].extend(p.x)
            xyData['Wire_deflection'].extend(p.y)
          elif not config.offsetToZero and not config.plotFullDeflection:
            xyData['Post_deflection'].extend(p.tdNoOffset['post'].to_numpy())
            xyData['Wire_deflection'].extend(p.tdNoOffset['wire'].to_numpy())
          elif config.plotFullDeflection:
            p.resetDeflectionData()
            p.dropInitialPoints(p.droppedPoints)
            config.truncatePostDeflection = True
            temp = config.truncationValue
            config.truncationValue = config.fullDeflTruncationValue
            p.truncateData()
            p.calcForce()
            config.truncationValue = temp
            xyData['Post_deflection'].extend(p.td.post.to_numpy())
            xyData['Wire_deflection'].extend(p.td.wire.to_numpy())
           
          xyData['Force'].extend(p.force[0].to_numpy()*10e6)
          xyData['Location'].extend([p.getRowPostString()]*len(p.td.post.values.tolist()))
          fitData['xx'].extend(p.xfit)
          fitData['yy'].extend(p.yfit)
          fitData['yyForce'].extend(p.yfitForce*10e6)
          fitData['Location'].extend([p.getRowPostString()]*len(p.xfit))

          if p.sample.infiltrationTime == 0:
            xyData['Infiltration_status'].extend(['Uninfiltrated']*len(p.td.post.values.tolist()))
            fitData['Infiltration_status'].extend(['Uninfiltrated']*len(p.xfit))
          else:
            xyData['Infiltration_status'].extend(['Infiltrated']*len(p.td.post.values.tolist()))
            fitData['Infiltration_status'].extend(['Infiltrated']*len(p.xfit))
          xyData['Sample'].extend([p.sample.name]*len(p.td.post.values.tolist()))
          fitData['Sample'].extend([p.sample.name]*len(p.xfit))
          xySampleData[p.sample.name] = xyData
          xxyySampleData[p.sample.name] = fitData

        elif p.sample.name in xySampleData:
          if config.offsetToZero and not config.plotFullDeflection:
            xyData['Post_deflection'].extend(p.x)
            xyData['Wire_deflection'].extend(p.y)
          elif not config.offsetToZero and not config.plotFullDeflection:
            xyData['Post_deflection'].extend(p.tdNoOffset['post'].to_numpy())
            xyData['Wire_deflection'].extend(p.tdNoOffset['wire'].to_numpy())
          elif config.plotFullDeflection:
            p.resetDeflectionData()
            p.dropInitialPoints(p.droppedPoints)
            config.truncatePostDeflection = True
            temp = config.truncationValue
            config.truncationValue = config.fullDeflTruncationValue
            p.truncateData()
            p.calcForce()
            config.truncationValue = temp
            xyData['Post_deflection'].extend(p.td.post.to_numpy())
            xyData['Wire_deflection'].extend(p.td.wire.to_numpy())
          elif not config.offsetToZero and config.plotFullDeflection:
            p.resetDeflectionData()
            p.dropInitialPoints(p.droppedPoints)
            config.truncatePostDeflection = True
            temp = config.truncationValue
            config.truncationValue = config.fullDeflTruncationValue
            p.truncateData()
            p.calcForce()
            config.truncationValue = temp
            xyData['Post_deflection'].extend(p.tdNoOffset.post.to_numpy())
            xyData['Wire_deflection'].extend(p.tdNoOffset.wire.to_numpy())
          

          xyData['Force'].extend(p.force[0].to_numpy()*10e6)
          xyData['Location'].extend([p.getRowPostString()]*len(p.td.post.values.tolist()))
          fitData['xx'].extend(p.xfit)
          fitData['yy'].extend(p.yfit)
          fitData['yyForce'].extend(p.yfitForce*10e6)
          fitData['Location'].extend([p.getRowPostString()]*len(p.xfit))
          xyData['Sample'].extend([p.sample.name]*len(p.td.post.values.tolist()))
          fitData['Sample'].extend([p.sample.name]*len(p.xfit))

          if p.sample.infiltrationTime == 0:
            xyData['Infiltration_status'].extend(['Uninfiltrated']*len(p.td.post.values.tolist()))
            fitData['Infiltration_status'].extend(['Uninfiltrated']*len(p.xfit))
          else:
            xyData['Infiltration_status'].extend(['Infiltrated']*len(p.td.post.values.tolist()))
            fitData['Infiltration_status'].extend(['Infiltrated']*len(p.xfit))

      # savePostLevelGraphs(p)
      # fig = p.plots.get("Force Deflection plot")
      # plt.show(fig)
  #Set directory and file name of output file
  if config.trackerFiles != "All posts":
    filename = config.trackerFiles + f" output at {config.truncationValue} microns.xlsx"
  else:
    filename = sampleNamesString + f"output at {config.truncationValue} microns.xlsx"
  #Convert output data to a pandas dataframe and save as a excel file.
  df = pd.DataFrame(data)

  averageData = {'CNT_Diameter': [], 'CNT_Diam_std': [], 'CNT_Diam_Quantity': [], 'CNT_Diam_sem': [], 'CNT_Diam_cv': [],
                  'Modulus': [], 'Mod_std': [], 'Mod_Quantity':[], 'Mod_sem': [], 'Mod_cv': [],
                  'Infiltration_Time': [], 'Parametric': [], 'adStat': [], 'adCritVal': []}
  sampleData = {'Sample': [],
                 'Number_of_posts': [], 'Average_Modulus': [], 'std_Modulus':[], 'cv_Modulus': [], 'sem_Modulus': [],
                 'CNT_Diam_Quantity': [], 'Average_CNT_Diameter': [], 'std_Diameter': [], 'cv_Diameter': [], 'sem_CNT_Diameter': [], 
                 'Infiltration_Time': []}
  moduliByInfTime = {0: [], 15: [], 30: []}
  cntDiamSampleAveragesByInfTime = {0: [], 15: [], 30: []}
  for sample in samples:
    sampleData['Sample'].append(sample.name)
    sampleData['Number_of_posts'].append(len(sample.testedPosts))
    sampleData['Average_Modulus'].append(sample.avgModulus)
    sampleData['std_Modulus'].append(sample.stdModulus)
    sampleData['cv_Modulus'].append(sample.cvModulus)
    sampleData['sem_Modulus'].append(sample.semModulus)
    sampleData['CNT_Diam_Quantity'].append(sample.quantCNTDiameter)
    sampleData['Average_CNT_Diameter'].append(sample.avgCNTDiameter)
    sampleData['std_Diameter'].append(sample.stdCNTDiameter)
    sampleData['cv_Diameter'].append(sample.cvCNTDiameter)
    sampleData['sem_CNT_Diameter'].append(sample.semCNTDiameter)
    sampleData['Infiltration_Time'].append(sample.infiltrationTime)
  sdf = pd.DataFrame(sampleData)
  sdf.sort_values(by=['Sample'])

  infiltrationTimes = df.Infiltration_Time.unique()
  for time in infiltrationTimes:
    adf = df.query("Infiltration_Time == @time")
    averageData['CNT_Diameter'].append(np.mean(cntDiamsByInfTime[time]))
    averageData['CNT_Diam_std'].append(np.std(cntDiamsByInfTime[time]))
    averageData['CNT_Diam_Quantity'].append(len(cntDiamsByInfTime[time]))
    averageData['CNT_Diam_sem'].append(stats.sem(cntDiamsByInfTime[time]))
    averageData['CNT_Diam_cv'].append(np.std(cntDiamsByInfTime[time])/np.mean(cntDiamsByInfTime[time]))
    averageData['Modulus'].append(adf.Modulus.mean())
    averageData['Mod_std'].append(adf.Modulus.std())
    averageData['Mod_Quantity'].append(adf.shape[0])
    averageData['Mod_sem'].append(adf.Modulus.sem())
    averageData['Mod_cv'].append(adf.Modulus.std()/adf.Modulus.mean())
    averageData['Infiltration_Time'].append(time)
    res = stats.anderson(adf['Modulus'], dist='norm')
    averageData['adStat'].append(res.statistic)
    averageData['adCritVal'].append(res.critical_values[2])
    if res.statistic <= res.critical_values[2]:
      averageData['Parametric'].append(True)
    else:
      averageData['Parametric'].append(False)
    moduliByInfTime[time].append(adf.Modulus.to_numpy())
    cntDiamSampleAveragesByInfTime[time].append(adf.CNT_Diameter.to_numpy())
    # fig, ax = plt.subplots(figsize=(10,6))
    # sns.histplot(data=adf,x='Modulus', log_scale=True, kde=(True))
    # plt.title(time, fontsize=config.titleFontSize)
  adf = pd.DataFrame(averageData)
  adf.sort_values(by=['Infiltration_Time'])  
  df = df.sort_values(by=['Sample'])
  
  # sns.set_style('ticks')cntDiamsByInfTime
  

  #statistics
  # fig, ax = plt.subplots(figsize=(10,6))
  # sns.histplot(data=df,x='CNT_Diameter', y='Modulus', log_scale=(False, True))
  # fig, ax = plt.subplots(figsize=(10,6))
  # sns.histplot(data=df,x='CNT_Diameter', kde=(True))
  # fig, ax = plt.subplots(figsize=(10,6))
  # sns.histplot(data=df,x='Modulus', log_scale=True, kde=(True))
  # res = stats.anderson(df['Modulus'], dist='logistic')
  # res.fit_result.plot()
  # print('Test statistic:', res.statistic)
  # print('Critical values:', res.critical_values)
  # print('Significance levels:', res.significance_level)
  
  # kwRes = stats.kruskal(moduliByInfTime[0][0], moduliByInfTime[15][0], moduliByInfTime[30][0])
  # print('kw res for Moduni: ', kwRes)
  
  # kwRes = stats.kruskal(cntDiamsByInfTime[0][0], cntDiamsByInfTime[15][0], cntDiamsByInfTime[30][0])
  # print('kw res for CNT Diams: ', kwRes)
  
  
  
  # Graph and save plots of output data to PDF file.
  
  if config.plotOverviewGraphs:
    # # Average CNT diameter vs Modulus
    # xAxis = 'Sample_Average_CNT_Diameter'
    # yAxis = 'Modulus'
    # hue = 'Sample'
    # legend = True
    # xlabel = 'Sample Average CNT Diameter (nm)'
    # ylabel = 'Modulus (Pa)'
    # title = 'Modulus vs Average CNT Diameter'
    # ylog = True
    # overviewGraphs(df, xAxis, yAxis, hue=hue, legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    
    # # Modulus vs Post length  
    # xAxis = 'Post_Length'
    # yAxis = 'Modulus'
    # hue = 'Infiltration_Recipe'
    # style = 'Sample'
    # legend = True
    # xlabel = 'Post Length '+ u'(\u03bcm)'
    # ylabel = 'Modulus (Pa)'
    # title = 'Modulus vs Post_Length'
    # ylog = True
    # overviewGraphs(df, xAxis, yAxis, hue=hue, style=style, legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    
    # # Modulus vs Effective_Post_Length
    # xAxis = 'Effective_Length'
    # yAxis = 'Modulus'
    # hue = 'Infiltration_Recipe'
    # style = 'Sample'
    # legend = True
    # xlabel = 'Effective Post Length '+ u'(\u03bcm)'
    # ylabel = 'Modulus (Pa)'
    # title = 'Modulus vs Effective_Length'
    # ylog = True
    # overviewGraphs(df, xAxis, yAxis, hue=hue, style=style, legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    
    # # Modulus vs Wire_Effective_Length
    # xAxis = 'Wire_Effective_Length'
    # yAxis = 'Modulus'
    # hue = 'Infiltration_Recipe'
    # style = 'Sample'
    # legend = True
    # xlabel = 'Wire Length '+ u'(\u03bcm)'
    # ylabel = 'Modulus (Pa)'
    # title = 'Modulus vs Wire_Effective_Length'
    # ylog = True
    # overviewGraphs(df, xAxis, yAxis, hue=hue, style=style, legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
  
    # Modulus vs CNT Diameter 
    xAxis = 'CNT_Diameter'
    yAxis = 'Modulus'
    hue = 'Infiltration_Time'
    style = 'Sample'
    legend = True
    truncateLegend = True
    config.legendIn = True
    xlabel = 'CNT Diameter (nm)'
    ylabel = 'Modulus (Pa)'
    title = 'Modulus vs CNT Diameter'
    ylog = True
    fontSize = 20
    overviewGraphs(df, xAxis, yAxis, hue=hue, style=style, legend=legend, truncateLegend=truncateLegend,
                  xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    
    # # Average Moulus vs CNT Diameter 
    # xAxis = 'CNT_Diameter'
    # xErr = 'CNT_Diam_std'
    # yAxis = 'Modulus'
    # yErr = 'Mod_std'
    # hue = 'Infiltration_Time'
    # legend = True
    # truncateLegend = False
    # xlabel = 'CNT Diameter (nm)'
    # ylabel = 'Modulus (Pa)'
    # title = 'Modulus vs CNT Diameter'
    # ylog = True
    # errorBars = True
    # overviewGraphs(adf, xAxis, yAxis, xErr=xErr, yErr=yErr, errorBars=errorBars, hue=hue, legend=legend, truncateLegend=truncateLegend,
    #                xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    

    
    # Modulus vs Infiltration time violin and swarm plot
    # df = df.sort_values(by=['Infiltration_Time'])
    # snsPlotType = 'violin_swarm'
    # xAxis = 'Infiltration_Time'
    # yAxis = 'Modulus'
    # hue = 'Sample'
    # style = 'Sample'
    # xlabel = 'Infiltration Time'
    # ylabel = 'Modulus (Pa)'
    # title = 'Modulus vs Infiltration time'
    # ylog = True
    # cPalette = ['Pastel2','tab10','white']
    # overviewGraphs(df, xAxis, yAxis, snsPlotType=snsPlotType, hue=hue, style=style,
    #                legend=False, colorViolinEdge=True, 
    #                markerPalette=cPalette[1], violinEdgePalette=cPalette[1], violinFillPallete=cPalette[2],
    #                xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    
    # # Modulus vs Infiltration time box
    df = df.sort_values(by=['Infiltration_Time'])
    snsPlotType = 'boxplot'
    xAxis = 'Infiltration_Time'
    yAxis = 'Modulus'
    hue = None
    legend = False
    truncateLegend = False
    xlabel = 'Infiltration Time'
    ylabel = 'Modulus (Pa)'
    title = 'Modulus vs Infiltration time'
    ylog = True
    cPalette = 'tab10'
    overviewGraphs(df, xAxis, yAxis, snsPlotType=snsPlotType, hue=hue,  markerPalette=cPalette, legend=legend,
                    xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    
    # CNT diameter vs Infiltration time box
    vertBoxes = False
    snsPlotType = 'boxplot'
    hue = None
    legend = False
    title = 'CNT Diameter vs Infiltration time'
    ylog = False
    cPalette = ['Pastel2','tab10','white']
    if vertBoxes:
      df = df.sort_values(by=['Infiltration_Time'])
      xAxis = 'Infiltration_Time'
      yAxis = 'CNT_Diameter'
      xlabel = 'Infiltration Time'
      ylabel = 'CNT Diameter (nm)'
      # overviewGraphs(df, xAxis, yAxis, snsPlotType=snsPlotType, hue=hue, markerPalette=cPalette[1], 
      #               legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    # for Horizontal boxes
    else:
      df = df.sort_values(by=['Infiltration_Time'], ascending=False)
      df = df.astype({'Infiltration_Time': str})
      yAxis = 'Infiltration_Time'
      xAxis = 'CNT_Diameter'
      ylabel = 'Infiltration Time'
      xlabel = 'CNT Diameter (nm)'
      boxOrien = 'h'
      boxOrder = ['0', '15', '30'].reverse()
      hueOrder = ['0', '15', '30']
      cmap = matplotlib.cm.get_cmap(cPalette[1])
      c1 = cmap(0)
      c2 = cmap(0.1)
      c3 = cmap(0.2)
      colors = [c3, c2, c1]
      overviewGraphs(df, xAxis, yAxis, snsPlotType=snsPlotType, boxOrien=boxOrien,
                    boxOrder=boxOrder, hue=hue, markerPalette=colors, 
                    legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
      
    # for all versions 
    
    # Modulus vs Infiltration time box
    # vertBoxes = True
    # snsPlotType = 'boxplot'
    # hue = None
    # legend = False
    # title = 'Modulus vs Infiltration time'
    # ylog = True
    # cPalette = ['Pastel2','tab10','white']
    # if vertBoxes:
    #   df = df.sort_values(by=['Infiltration_Time'])
    #   xAxis = 'Infiltration_Time'
    #   yAxis = 'Modulus'
    #   xlabel = 'Infiltration Time'
    #   ylabel = 'Modulus (Pa)'
    #   fig = overviewGraphs(df, xAxis, yAxis, snsPlotType=snsPlotType, hue=hue, markerPalette=cPalette[1], 
    #                legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    # # for Horizontal boxes
    # else:
    #   df = df.sort_values(by=['Infiltration_Time'], ascending=False)
    #   df = df.astype({'Infiltration_Time': str})
    #   yAxis = 'Infiltration_Time'
    #   xAxis = 'Modulus'
    #   ylabel = 'Infiltration Time'
    #   xlabel = 'Modulus (Pa)'
    #   boxOrien = 'h'
    #   boxOrder = ['0', '15', '30'].reverse()
    #   hueOrder = ['0', '15', '30']
    #   cmap = matplotlib.cm.get_cmap(cPalette[1])
    #   c1 = cmap(0)
    #   c2 = cmap(0.1)
    #   c3 = cmap(0.2)
    #   colors = [c3, c2, c1]
    #   overviewGraphs(df, xAxis, yAxis, snsPlotType=snsPlotType, boxOrien=boxOrien,
    #                boxOrder=boxOrder, hue=hue, markerPalette=colors, 
    #                legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    # # for all versions 
    
    
    # # Modulus vs Failure Mode
    # df = df.sort_values(by=['Failure_Mode'])
    # snsPlotType = 'swarmplot'
    # xAxis = 'Failure_Mode'
    # yAxis = 'Modulus'
    # hue = 'Infiltration_Recipe'
    # hue1 = 'Sample'
    # style = 'Sample'
    # legend = True
    # xlabel = 'Failure Mode'
    # ylabel = 'Modulus (Pa)'
    # title = 'Modulus vs Failure mode'
    # ylog = True
    # overviewGraphs(df, xAxis, yAxis, snsPlotType=snsPlotType, hue=hue, style=style, legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    # overviewGraphs(df, xAxis, yAxis, snsPlotType=snsPlotType, hue=hue1, style=style, legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
    
    # df = df.sort_values(by=['Failure_Mode'])
    # snsPlotType = 'swarmplot'
    # xAxis = 'Failure_Mode'
    # yAxis = 'Sample'
    # hue = 'Infiltration_Recipe'
    # hue1 = 'Sample'
    # style = 'Sample'
    # legend = True
    # xlabel = 'Failure Mode'
    # ylabel = 'Sample'
    # title = 'Sample vs Failure mode'
    # overviewGraphs(df, xAxis, yAxis, snsPlotType=snsPlotType, hue=hue, style=style, legend=legend, xlabel=xlabel, ylabel=ylabel, title=title, ylog=ylog)
  
  # US vs Effective Length
  # ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
  df =df.set_index('Sample')
  df.to_excel(directory + "/" + filename)
  sdf = sdf.set_index('Sample')
  sdf.to_excel(directory + '/' + config.sampleExcelFileName)
  adf = adf.set_index('Infiltration_Time')
  adf.to_excel(directory + '/' + config.infTimeExcelFileName)
  
  # NOTE: The following 4 for loops plot each sample's data and all the data on one plot
  # Loop 1: plots one sample's deflection data
  # infiltrationTime is either an infiltration time or 'all'. It will limit the sample plots to that infiltration time.
  
  if config.oneGraph:
    oneGraphData = {'Location': [], 'Post_deflection': [],
                    'Wire_deflection': [], 'Force': [],'Infiltration_status': [],  'Sample': []}

  
  if config.testMode:
    plt.close('all')
  if config.matchYscale:
      maxYValueByInfTime = {}
      for key in xySampleData:
        infTime = sdf.at[key,'Infiltration_Time']
        snsDF = pd.DataFrame(xySampleData[key])
        if infTime not in maxYValueByInfTime:
          maxYValueByInfTime[infTime] = []
        maxYValueByInfTime[infTime].append(snsDF.max(numeric_only=True).Force)
  if config.paintOverFigure:
    fig, ax = plt.subplots(figsize=(10,6))
  if config.forceColor:
    i = 0 
  for key in xySampleData:
    infTime = sdf.at[key,'Infiltration_Time']
    if infTime != config.infiltrationTime and config.infiltrationTime != 'all':
      continue
    if config.oneGraph:
      for k, values in xySampleData[key].items():
        oneGraphData[k].extend(values)
      continue
    snsDF = pd.DataFrame(xySampleData[key])
    fitDF = pd.DataFrame(xxyySampleData[key])
    if not config.paintOverFigure:
      fig, ax = plt.subplots(figsize=(10,6))
    plt.xlabel('Post deflection ' + u'(\u03bcm)', fontsize=config.titleFontSize)
    if config.forceColor:
      sns.set_palette(config.forcedColor[i])
      i = i+1 
    if config.plotDeflection:
      sns.scatterplot(x='Post_deflection', y='Wire_deflection', data=snsDF, hue=config.hue, alpha = 0.1, s=100,
                    legend=config.legend)
      plt.ylabel('Wire deflection ' + u'(\u03bcm)', fontsize=config.titleFontSize)
    if config.plotFits and not config.plotForce:
      sns.lineplot(x='xx', y='yy', data=fitDF, hue=config.hue, palette='dark', legend=config.legend)
    if config.plotFits and config.plotForce:
      sns.lineplot(x='xx', y='yyForce', data=fitDF, hue=config.hue, palette='dark', legend=config.legend)
    if config.plotForce:
      sns.scatterplot(x='Post_deflection', y='Force', data=snsDF, hue=config.hue, alpha = 0.1, s=100,
                    legend=config.legend)
      plt.ylabel('Force ' + u'(\u03bcN)', fontsize=config.titleFontSize)
    if not config.legendIn and config.legend:
      plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    if config.limxAxis:
      plt.xlim(right=config.xlimit)
    if config.matchYscale:
      plt.ylim(top=(max(maxYValueByInfTime[infTime])*1.05)) 
    if config.tightLayout:
      plt.tight_layout()
    if config.title:
      plt.title(f'Sample: {key}, Inf time:  {infTime}, Offset to zero: {config.offsetToZero}', fontsize=16)
    if not config.testMode:
      plt.show()
    if key == '202-I':
        print(f'{ax.get_ylim()}')
        print(f'{ax.get_xlim()}')
        # plt.ylim(top=117)
    config.pdf.savefig(fig) #, bbox_inches='tight')
    if config.truncatePostDeflection and not config.plotFullDeflection:
      fig.savefig(directory + f'/Sample {key} at {config.truncationValue}.svg')#, bbox_inches='tight')
    elif config.plotFullDeflection:
      fig.savefig(directory + f'/Sample {key} at {config.fullDeflTruncationValue}.svg')
    else:
      fig.savefig(directory + f'/Sample {key} all points.svg')#, bbox_inches='tight')
    if config.checkBoxDims:
      checkBoxDimensions(f'Sample {key}', fig, ax)
  
  if config.oneGraph:
    ogDF = pd.DataFrame(oneGraphData)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(x='Post_deflection', y='Force', data=ogDF, hue=config.hue, alpha = 0.1, s=100,
                    legend=config.legend)
    plt.ylabel('Force ' + u'(\u03bcN)', fontsize=config.titleFontSize)
    plt.xlabel('Post deflection ' + u'(\u03bcm)', fontsize=config.titleFontSize)
    if config.truncatePostDeflection and not config.plotFullDeflection:
      fig.savefig(directory + f'/{config.infiltrationTime} inf time at {config.truncationValue}.svg')#, bbox_inches='tight')
    elif config.plotFullDeflection:
      fig.savefig(directory + f'/{config.infiltrationTime} inf time at {config.fullDeflTruncationValue}.svg')
    config.pdf.savefig(fig)
    # if config.truncatePostDeflection and not config.plotFullDeflection:
    #   fig.savefig(directory + f'/Sample {key} at {config.truncationValue}.svg')#, bbox_inches='tight')
    # elif config.plotFullDeflection:
    #   fig.savefig(directory + f'/Sample {key} at {config.fullDeflTruncationValue}.svg')
    # else:
    #   fig.savefig(directory + f'/Sample {key} all points.svg')
  # keys = ['200-U', '201-I', '202-I', '203-I']
  # keysI = ['188-I', '195-I', '199-I', '190-I', '201-I','202-I', '203-I']
  # keysU = ['198-U', '194-U', '200-U', ]
  # keys = keysI + keysU
  # cmap = matplotlib.cm.get_cmap('tab10')  
  
  # fig, ax = plt.subplots(figsize=(10,6))
  # plt.xlabel('Post deflection ' + u'(\u03bcm)', fontsize=config.titleFontSize)
  # plt.ylabel('Wire deflection ' + u'(\u03bcm)', fontsize=config.titleFontSize)
  # for key in keysI:
  #   snsDF = pd.DataFrame(xySampleData[key])
  #   s1 = sns.scatterplot('Post_deflection', 'Wire_deflection', data=snsDF,
  #                   alpha = 0.4, legend=False, ax=ax, c=cmap(0), label='Infiltrated')
  #   plt.title('Sample ' + key, fontsize=16)
  #   # i = i+1
  # for key in keysU:
  #   snsDF = pd.DataFrame(xySampleData[key])
  #   s2 = sns.scatterplot('Post_deflection', 'Wire_deflection', data=snsDF,
  #                   alpha = 0.4, legend=False, ax=ax, c=cmap(1), label='Uninfiltrated')
  #   plt.title('Sample ' + key, fontsize=16)
  #   # i = i+1
  # # plt.legend(labels=keys, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
  # handles, labels = ax.get_legend_handles_labels()
  # plt.legend(handles=[handles[0], handles[-1]], labels=[labels[0], labels[-1]], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
  # plt.tight_layout()
  
  # fig, ax = plt.subplots(figsize=(10,6))
  # plt.xlabel('Post deflection ' + u'(\u03bcm)', fontsize=config.titleFontSize)
  # plt.ylabel('Wire deflection ' + u'(\u03bcm)', fontsize=config.titleFontSize)
  # i=0
  # for key in keys:
  #   snsDF = pd.DataFrame(xySampleData[key])
  #   sns.scatterplot('Post_deflection', 'Wire_deflection', data=snsDF,
  #                   alpha = 0.5, legend=False, ax=ax, c=cmap(i/10))
  #   # plt.title('Sample ' + key, fontsize=16)
  #   i = i+1
  # plt.legend(labels=keys, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    
  
  
  
  
  # config.pdf.savefig(fig, bbox_inches='tight')
  # fig.savefig(directory + f'/Sample {key} at {config.truncationValue}.svg', bbox_inches='tight')
  config.pdf.close()
  
  
main()