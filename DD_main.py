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
    
  #Test mode will run the entire program with no input from user.
  testMode = True
  #Do you want to read in previously processed posts from an excel file?
  readExcel = True
  excelName = 'All posts'
  #Declare which tracker files use want to use, i.e. Bad posts, Good posts, or All posts
  #This will change which directory the tracker files are pulled from. 
  trackerFiles = "All posts"
  # Delare sample name or names you want to do in this run. If you are reading in an excel file, sampleNames is not used. 
  # ['188-I', '190-I', '194-U', '195-I', '198-U', '199-I', '200-U', '201-I', '202-I', '203-I']
  sampleNames = ['188-I', '190-I', '194-U', '195-I', '198-U', '199-I', '200-U', '201-I', '202-I', '203-I']
  #If true, the deflection curve and linear fits of each post will be saved to the pdf
  saveEachPostPlot = True
  #Do you want a 'Multi', 'One', or 'Both'? Both is just for graph comparison. Most calculations won't work.
  numOfLineFits = 'One'
  #Do you want to classify each deflection curve?
  #If true, your input will be required even if testMode = true
  #Will default to False if numOfLineFits == 'One'
  classifyDeflections = False
  if numOfLineFits == 'One':
    classifyDeflections = False
  #The initial deflection region ends when failure occurs. 
  initialDeflections = {1: 'Negative_curvature', 2: 'Straight', 3: 'Positive_curvature'}
  # Failure is denoted by a change in adjacent slopes of 2.5, 
  # or unusual jumps in the post defleciton of 10 microns or more.
  failures = {1: 'No_failure', 2: 'Jump_failure', 3: 'Gradual_failure'}
  # After failure behavior doesn not contribute to 
  afterFailures = {1: 'Softening', 2: 'Constant', 3: 'Stiffening', 4: 'N/A'}
  curveSubsections = {'Initial_deflection': initialDeflections, 'Failure': failures, 'After_failure': afterFailures}
  #Do you want to truncate the post deflection curves at some value of microns?
  truncatePostDeflection = True
  truncationValue = 400
  
  #plot style
  sns.set()
  plt.rcParams.update({"xtick.bottom" : True, "ytick.left" : True})
  #run program
  if readExcel:
    samples, sampleNamesString, directory = grabSamples(excelName, [])
  else:
    samples, sampleNamesString, directory = grabSamples('', sampleNames)
  #Set PDF name and opens pdf file to save all graphs in.
  if trackerFiles != "All posts":
    if saveEachPostPlot:
      pdfName = trackerFiles + f" Fitted Deflection Graphs at {truncationValue} microns.pdf"
    else:
      pdfName = trackerFiles + f" Overview Graphs at f{truncationValue} microns.pdf"
    pdf = mpdf.PdfPages(directory + "/" + pdfName)
  else:
    if saveEachPostPlot:
      pdfName = sampleNamesString + f"Fitted Deflection Graphs at {truncationValue} microns.pdf"
    else:
      pdfName = sampleNamesString + f"Overview Graphs at f{truncationValue} microns.pdf"
    pdf = mpdf.PdfPages(directory + "/" + pdfName)
  #Loop through sample objects and analyze each tested post  
  allPosts = []
  for sample in samples:
    posts = analyzePosts(sample)
    # saveSampleLevelGraphs(posts)
    allPosts.extend(posts)
    
  xySampleData = {}
  #The dictionary of all the parameters you want in your output file.
  if numOfLineFits == 'Multi' or numOfLineFits == 'Both':
    data = {'Sample': [], 'Row': [], 'Post': [], 'Initial_Deflection_class': [], 'Failure_class': [],
          'After_failure_class': [], 'Modulus': [], 'Modulus_Uncertainty': [], 
          'Percent_Error': [], 'Slope_Percent_Error': [],
          'Ultimate_Strength': [], 'CNT_Diameter': [], 'Sample_Average_CNT_Diameter': [], 
          'Effective_Length': [], 'Failure_Mode': [], 'Infiltration_Recipe': [], 'min_samples_split': [],
          'min_samples_leaf': [], 'max_depth': [], 'max_bins': [], 'Slope_segment': [], 'Problem_Post': []}
    #Populate output data
    for p in allPosts:
      if p.modulus != 0:
        if classifyDeflections:
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
        
  elif numOfLineFits == 'One':
    data = {'Sample': [], 'Row': [], 'Post': [], 'Modulus': [], 'Modulus_Uncertainty': [], 
          'Percent_Error': [], 'Slope_Percent_Error': [], 'Ultimate_Strength': [], 'CNT_Diameter': [],
          'Sample_Average_CNT_Diameter': [], 'Effective_Length': [], 'Failure_Mode': [],
          'Infiltration_Recipe': [], 'Slope_Initial_Guess': [], 
          'Intercept_Initial_Guess': [], 'Problem_Post': []}
    #Populate output data
    for p in allPosts:
      if p.modulus != 0:
        if classifyDeflections:
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
  if trackerFiles != "All posts":
    filename = trackerFiles + f" output at {truncationValue} microns.xlsx"
  else:
    filename = sampleNamesString + f"output at {truncationValue} microns.xlsx"
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
  pdf.savefig(fig, bbox_inches='tight')
  
  fig, ax = plt.subplots(figsize=(10,6))
  ax.set_yscale('log')
  sns.scatterplot('CNT_Diameter', 'Modulus', data=df, hue='Sample', style='Sample')
  plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
  plt.xlabel('CNT Diameter (nm)')
  plt.ylabel('Modulus (Pa)')
  plt.tight_layout()
  plt.title('Modulus vs CNT Diameter', fontsize=16)
  plt.show()
  pdf.savefig(fig, bbox_inches='tight')
  
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
  pdf.savefig(fig, bbox_inches='tight')
  
  for key in xySampleData:
    snsDF = pd.DataFrame(xySampleData[key])
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot('Post_deflection', 'Wire_deflection', data=snsDF, hue='Location', alpha = 0.65, style='location')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.xlabel('Post deflection ' + u'(\u03bcm)')
    plt.ylabel('Wire deflection ' + u'(\u03bcm)')
    plt.tight_layout()
    plt.title('Sample ' + key, fontsize=16)
    plt.show()
    pdf.savefig(fig, bbox_inches='tight')
    fig.savefig(directory + f'/Sample {key}.png', bbox_inches='tight')
  

  pdf.close()    
  
main()