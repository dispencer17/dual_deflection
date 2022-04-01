# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:08:27 2022

@author: dispe
"""


import DD_functions
from DD_functions import *
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
import matplotlib
import seaborn as sns

def main():
    
  #FIRST: Check the config file to make sure 
  
  #plot style
  sns.set()
  plt.rcParams.update({"xtick.bottom" : True, "ytick.left" : True})
  #run program
  if config.readExcel:
    samples, sampleNamesString, directory = grabSamples(config.excelName, [])
  else:
    samples, sampleNamesString, directory = grabSamples('', sampleNames)
  #Set PDF name and opens pdf file to save all graphs in.
  if config.trackerFiles != "All posts":
    if config.saveEachPostPlot:
      pdfName = config.trackerFiles + f" Fitted Deflection Graphs at {config.truncationValue} microns.pdf"
    else:
      pdfName = config.trackerFiles + f" Overview Graphs at f{config.truncationValue} microns.pdf"
    config.pdf = mpdf.PdfPages(directory + "/" + pdfName)
  else:
    if config.saveEachPostPlot:
      pdfName = sampleNamesString + f"Fitted Deflection Graphs at {config.truncationValue} microns.pdf"
    else:
      pdfName = sampleNamesString + f"Overview Graphs at f{config.truncationValue} microns.pdf"
    config.pdf = mpdf.PdfPages(directory + "/" + pdfName)
  #Loop through sample objects and analyze each tested post  
  allPosts = []
  for sample in samples:
    posts = analyzePosts(sample)
    # saveSampleLevelGraphs(posts)
    allPosts.extend(posts)
    
  xySampleData = {}
  #The dictionary of all the parameters you want in your output file.
  if config.numOfLineFits == 'Multi' or config.numOfLineFits == 'Both':
    data = {'Sample': [], 'Row': [], 'Post': [], 'Initial_Deflection_class': [], 'Failure_class': [],
          'After_failure_class': [], 'Modulus': [], 'Modulus_Uncertainty': [], 
          'Percent_Error': [], 'Slope_Percent_Error': [],
          'Ultimate_Strength': [], 'CNT_Diameter': [], 'Sample_Average_CNT_Diameter': [], 
          'Effective_Length': [], 'Failure_Mode': [], 'Infiltration_Recipe': [], 'min_samples_split': [],
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
        data['Effective_Length'].append(p.effectiveLength*10**6)
        data['Failure_Mode'].append(p.failureMode)
        data['Infiltration_Recipe'].append(p.sample.infiltrationRecipe)
        data['min_samples_split'].append(p.fitParameters[0])
        data['min_samples_leaf'].append(p.fitParameters[1])
        data['max_depth'].append(p.fitParameters[2])
        data['max_bins'].append(p.fitParameters[3])
        data['Slope_segment'].append(p.segment)
        data['Problem_Post'].append(p.problemPost)
        
        if not p.sample.name in xySampleData:
          xyData = {'Location': [], 'Post_deflection': [], 'Wire_deflection': []}
          xyData['Post_deflection'].extend(p.tdPost.values.tolist())
          xyData['Wire_deflection'].extend(p.tdWire.values.tolist())
          xyData['Location'].extend([p.getRowPostString()]*len(p.tdPost.values.tolist()))
          xySampleData[p.sample.name] = xyData
        elif p.sample.name in xySampleData:
          xyData['Post_deflection'].extend(p.tdPost.values.tolist())
          xyData['Wire_deflection'].extend(p.tdWire.values.tolist())
          xyData['Location'].extend([p.getRowPostString()]*len(p.tdPost.values.tolist()))
        
  elif config.numOfLineFits == 'One':
    data = {'Sample': [], 'Row': [], 'Post': [], 'Modulus': [], 'Modulus_Uncertainty': [], 
          'Percent_Error': [], 'Slope_Percent_Error': [], 'Ultimate_Strength': [], 'CNT_Diameter': [],
          'Sample_Average_CNT_Diameter': [], 'Effective_Length': [], 'Failure_Mode': [],
          'Infiltration_Recipe': [], 'Slope_Initial_Guess': [], 
          'Intercept_Initial_Guess': [],'Initial_Points_Dropped': [], 'Problem_Post': []}
    #Populate output data
    for p in allPosts:
      if p.modulus != 0:
        if config.classifyDeflections:
          p.curveClassification = userClassifyDeflections(p.getAddressString())
        data['Sample'].append(p.sample.name)
        data['Row'].append(p.row)
        data['Post'].append(p.post)
        data['Modulus'].append(p.modulus)
        data['Modulus_Uncertainty'].append(p.modUncertainty)
        data['Percent_Error'].append(p.modUncertainty/p.modulus)
        data['Slope_Percent_Error'].append(p.slopeError/p.goodSlope)
        data['Ultimate_Strength'].append(p.ultStrength)
        data['CNT_Diameter'].append(p.avgCNTDiameter)
        data['Sample_Average_CNT_Diameter'].append(p.sample.avgCNTDiameter)
        data['Effective_Length'].append(p.effectiveLength*10**6)
        data['Failure_Mode'].append(p.failureMode)
        data['Infiltration_Recipe'].append(p.sample.infiltrationRecipe)
        data['Slope_Initial_Guess'].append(p.initialGuesses[0])
        data['Intercept_Initial_Guess'].append(p.initialGuesses[1])
        data['Initial_Points_Dropped'].append(p.droppedPoints)
        data['Problem_Post'].append(p.problemPost)
        
        if not p.sample.name in xySampleData:
          xyData = {'Location': [], 'Post_deflection': [], 'Wire_deflection': []}
          xyData['Post_deflection'].extend(p.tdPost.values.tolist())
          xyData['Wire_deflection'].extend(p.tdWire.values.tolist())
          xyData['Location'].extend([p.getRowPostString()]*len(p.tdPost.values.tolist()))
          xySampleData[p.sample.name] = xyData
        elif p.sample.name in xySampleData:
          xyData['Post_deflection'].extend(p.tdPost.values.tolist())
          xyData['Wire_deflection'].extend(p.tdWire.values.tolist())
          xyData['Location'].extend([p.getRowPostString()]*len(p.tdPost.values.tolist()))


  
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

  df =df.set_index('Sample')
  df.to_excel(directory + "/" + filename)
  
  
  #Graph and save plots of output data to PDF file.
  fig, ax = plt.subplots(figsize=(10,6))
  ax.set_yscale('log')
  sns.scatterplot('Sample_Average_CNT_Diameter', 'Modulus', data=df, hue='Sample')
  plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
  plt.xlabel('Sample Average CNT Diameter (nm)')
  plt.ylabel('Modulus (Pa)')
  plt.tight_layout()
  plt.title('Modulus vs Average CNT Diameter', fontsize=16)
  plt.show()
  config.pdf.savefig(fig, bbox_inches='tight')
  fig.savefig(directory + f'/Mod vs Avg CNT diam at {config.truncationValue}.svg', bbox_inches='tight')
  
  fig, ax = plt.subplots(figsize=(10,6))
  ax.set_yscale('log')
  sns.scatterplot('CNT_Diameter', 'Modulus', data=df, hue='Sample', style='Sample')
  plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
  plt.xlabel('CNT Diameter (nm)')
  plt.ylabel('Modulus (Pa)')
  plt.tight_layout()
  plt.title('Modulus vs CNT Diameter', fontsize=16)
  plt.show()
  config.pdf.savefig(fig, bbox_inches='tight')
  fig.savefig(directory + f'/Mod vs CNT diam at {config.truncationValue}.svg', bbox_inches='tight')
  
  fig, ax = plt.subplots(figsize=(10,6))
  ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
  # ax.ticklabel_format(axis='x', style='scientific', scilimits=(-6, -6))
  sns.scatterplot('Effective_Length', 'Ultimate_Strength', data=df, hue='Sample')
  plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
  plt.xlabel('Effective Length ' + u'(\u03bcm)')
  plt.ylabel('Ultimate Strength (Pa)')
  plt.tight_layout()
  plt.title('US vs Effective Length', fontsize=16)
  plt.show()
  config.pdf.savefig(fig, bbox_inches='tight')
  
  for key in xySampleData:
    snsDF = pd.DataFrame(xySampleData[key])
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot('Post_deflection', 'Wire_deflection', data=snsDF, hue='Location', alpha = 0.65, style='Location')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.xlabel('Post deflection ' + u'(\u03bcm)')
    plt.ylabel('Wire deflection ' + u'(\u03bcm)')
    plt.tight_layout()
    plt.title('Sample ' + key, fontsize=16)
    plt.show()
    config.pdf.savefig(fig, bbox_inches='tight')
    fig.savefig(directory + f'/Sample {key} at {config.truncationValue}.svg', bbox_inches='tight')
  

  config.pdf.close()    
  
main()