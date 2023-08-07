from scipy import optimize
from scipy import odr
import scipy.stats as stats
import numpy as np
import pandas as pd
import re
import os
from pandas.core.frame import DataFrame
from converters import UnitConverters
import config
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.ticker import NullFormatter, FixedLocator
import matplotlib
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# import pwlf
# from GPyOpt.methods import BayesianOptimization

import pprint
import sys
from easygui import *
# from sklearn.linear_model import *
# from sklearn.linear_model import LinearRegression
# from lineartree import LinearTreeRegressor
# from sklearn.datasets import make_regression
from statistics import mean
from statistics import stdev
import statsmodels.api as sm
import matplotlib.backends.backend_pdf as mpdf



class Sample:
  def __init__(self, name):
    self.sampleCharacteristics = None
    self.name = name
    self.posts = np.zeros((config.dim_rows, config.dim_columns)).tolist()
    self.testedPosts = []
    self.plots = {}
    self.avgModulus = 0.0
    self.stdModulus = 0.0
    self.cvModulus = 0.0
    self.semModulus = 0.0
    self.failureMode = ''
    self.quantCNTDiameter = 0
    self.avgCNTDiameter = 0.0
    self.stdCNTDiameter = 0.0
    self.cvCNTDiameter = 0.0
    self.semCNTDiameter = 0.0
    self.growthTime = 0.0
    self.infiltrationTime = 0.0
    self.infiltrationRecipe = ''
    self.inSEMdata = False
    self.inTrackerData = False
    self.inSampleData = False
    self.firstInitialization = True
    self.checkMode = False
    
    # read and load tracker data for posts
    self.loadTrackerData()
    self.loadMetadata()
    self.setTestedPosts()
    self.setAvgModAndStats()
    self.setFailureMode()
    self.setAvgCNTDiamAndStats()
    self.setInfilData()
    self.firstInitialization = False
  
  def loadMetadata(self):
  # Read and load metadata
    directory = config.data_directory + "/" + config.metadata_directory
    filenames = sorted(os.listdir(directory))

    for file in filenames:
      if file in config.SampleCharacteristicFiles:
        self.loadSampleMetadata(directory + "/" + file)
      elif file in config.DdTestConditionFiles:
        self.loadDdTestConditions(directory + "/" + file)
      elif file in config.SemDataFiles:
        self.loadPostSemData(directory + "/" + file)
    
  def updateWithExcelData(self, row):
    initGuesses = np.zeros(2).tolist()
    fitParameters = np.zeros(4).tolist()
    p = self.posts[row['Row']][row['Post']]
    # if self.name == '198-U' and p.row == 4:
    #   print("!!!")
    if config.numOfLineFits == 'Multi' or config.numOfLineFits == 'Both':
      fitParameters[0] = row['min_samples_split']
      fitParameters[1] = row['min_samples_leaf']
      fitParameters[2] = row['max_depth']
      fitParameters[3] = row['max_bins']
      segment = row['Slope_segment']
      curveClassification = row['Curve_class']
      p.fitParameters = fitParameters
      p.segment = segment
      p.curveClassification = curveClassification
    elif config.numOfLineFits == 'One':
      initGuesses[0] = row['Slope_Initial_Guess']
      initGuesses[1] = row['Intercept_Initial_Guess']
      p.droppedPoints = row['Initial_Points_Dropped']
      p.initialGuesses = initGuesses
    p.inExcel = True
    
  def zeroOffset(self, trackerData: DataFrame):
    firstRow = trackerData.values[0]
    transformer = lambda row: row - firstRow
    return trackerData.transform(transformer)

  def calculateTrackerDistance(self, trackerData: DataFrame):
    distanceColumn = np.sqrt(trackerData.get("x")**2 + trackerData.get("y")**2)
    return trackerData.assign(totalDistanceTraveled=distanceColumn)
    
  def savePlot(self, key, fig):
    self.plots[key] = fig
    
  def setTestedPosts(self):
    self.testedPosts.clear()
    if self.inTrackerData:
      for i in range(config.dim_rows):
        for j in range(config.dim_columns):
          post = self.posts[i][j]
          if post != 0.0:
            if config.readExcel:
              if post.inExcel or self.firstInitialization:
                self.testedPosts.append(post)
            else:
              self.testedPosts.append(post)

              
  def setAvgModAndStats(self):
    moduli = []
    if self.inTrackerData:
      for p in self.testedPosts:
        moduli.append(p.modulus)
      moduli = [i for i in moduli if i != 0]
      if len(moduli) == 0:
        self.avgModulus = 0.0
        self.stdModulus = 0.0
        self.cvModulus = 0.0
        self.semModulus = 0.0
      else:  
        self.avgModulus = mean(moduli)
        self.stdModulus = stdev(moduli)
        self.cvModulus = self.stdModulus/self.avgModulus
        self.semModulus = stats.sem(moduli)
        

  def setFailureMode(self):
    modes = []
    if self.inSEMdata:
      for p in self.testedPosts:
        fmode = p.semData.get("failure_mode")
        if isinstance(fmode, str):
            modes.append(fmode)
      self.failureMode = modes[0]
    
  def setAvgCNTDiamAndStats(self):
    averageDiams = []
    diams = []
    if self.inSEMdata:
      for p in self.testedPosts:
        averageDiam = p.semData.get('Average')
        diams.extend(p.individualCNTDiameters)
        if isinstance(averageDiam, float) and not np.isnan(averageDiam) and averageDiam != 0:
          averageDiams.append(averageDiam)
      if len(averageDiams) == 0:
        self.avgCNTDiameter = 0.0
        self.stdCNTDiameter = 0.0
        self.cvCNTDiameter = 0.0
        self.quantCNTDiameter = 0
        self.semCNTDiameter = 0.0
      else:
        self.quantCNTDiameter = len(diams)  
        self.avgCNTDiameter = mean(diams)
        self.stdCNTDiameter = stdev(diams)
        self.cvCNTDiameter = self.stdCNTDiameter/self.avgCNTDiameter
        self.semCNTDiameter = stats.sem(diams)
    
      
  def setInfilData(self):
    if self.inSampleData:
      self.infiltrationRecipe = self.sampleCharacteristics.get('infiltration_recipe').iloc[0]
      self.infiltrationTime = self.sampleCharacteristics.get('infiltration_time').iloc[0]
  
  def setGrowthTime(self):
    if self.inSampleData:
      self.growthTime = self.sampleCharacteristics.get('growth_time').iloc[0]
      
  def loadTrackerData(self, filename = ''):
    if config.trackerFiles == config.tracker_bad_posts:
      problemPost = True
      directory = config.data_directory + "/" + config.tracker_directory + "/" + config.tracker_bad_posts
    elif config.trackerFiles == config.tracker_good_posts:
      problemPost = False
      directory = config.data_directory + "/" + config.tracker_directory + "/" + config.tracker_good_posts
    elif config.trackerFiles == 'All posts':
      problemPost = False
      directory = config.data_directory + "/" + config.tracker_directory
    filenames = sorted(os.listdir(directory))
    pattern = self.name + "_row-(\d*)_post-(\d*).*(ref|wire)"
    if filename != '': pattern = filename
    for file in filenames:
      match = re.match(pattern, file)
      if match:
        row = int(match.group(1))
        post = int(match.group(2))
        dataType = match.group(3)

        readData = self.readTrackerDataFile("./" + directory + "/" + file)

        if (self.posts[row][post] == 0):
          self.posts[row][post] = Post(self, row, post, problemPost)
          

        offsetData = self.zeroOffset(readData)
        if (dataType == "ref"):
          data = self.calculateTrackerDistance(offsetData)
          self.posts[row][post].refData = self.calculateTrackerDistance(offsetData)
          p = self.posts[row][post]
          
        if (dataType == "wire"):
          self.posts[row][post].wireData = self.calculateTrackerDistance(offsetData)

        self.posts[row][post].calcTdPost()
        self.posts[row][post].sample = self

  def loadSampleMetadata(self, path):
    readData = self.readMetadataFile(path)
    self.sampleCharacteristics = readData.filter(axis=0, like=self.name)
    if self.sampleCharacteristics.empty:
      print(f"Sample {self.name} is not in Sample metadata.")
    else:
      self.inSampleData = True

  def loadDdTestConditions(self, path):
    readData = self.readMetadataFile(path)
    for r in readData.itertuples():
      row = r._asdict()
      if row['Index'] == self.name:
        self.inTrackerData = True
        rowNum = int(row.get("row_num"))
        postNum = int(row.get("post_num"))
        post = self.posts[rowNum][postNum]
        if post == 0.0:
          post = Post(self, rowNum, postNum)
        post.setDdTestConditions(row)
        self.posts[rowNum][postNum] = post
    if not self.inTrackerData:
      print(f"Sample {self.name} is not in Tracker output.")
  
  def loadPostSemData(self, path):
    readData = self.readMetadataFile(path)
    for r in readData.itertuples():
      row = r._asdict()
      if row['Index'] == self.name:
        self.inSEMdata = True
        rowNum = int(row.get("row_num"))
        postNum = int(row.get("post_num"))
        post = self.posts[rowNum][postNum]
        if post == 0.0:
          post = Post(self, rowNum, postNum)
        post.setSemData(row)
        self.posts[rowNum][postNum] = post
    if not self.inSEMdata:
      print(f"Sample {self.name} is not in SEM metadata.")

  def readMetadataFile(self, path):
    if path is None:
      raise Exception("path parameter is missing for reading metadata file")
    
    unitRow = pd.read_excel(path, nrows=1, skiprows=0).to_numpy()[0]
    converters = {}

    for i in range(len(unitRow)):
      converters[i] = UnitConverters[unitRow[i]] if UnitConverters.get(unitRow[i]) else UnitConverters["default"]

    readData = pd.read_excel(path, header=0, converters=converters, skiprows=lambda x: x in [1], index_col=0)

    #filter NaN indexes
    readData = readData[readData.index.notna()]
    #filter NaN columns
    readData = readData.dropna(axis=1, how="all")

    return readData

  def readTrackerDataFile(self, path):
    if path is None:
      raise Exception("path parameter is missing for reading tracker data file")

    readData = pd.read_table(path, skiprows=1)
    return readData

class Metadata:
  sampleCharacteristics: DataFrame
  infiltrationConditions: DataFrame
  

class Post:
  
  def __init__(self, sample, row, post, problemPost=False):
    self.sample = sample
    self.row = 0
    self.post = 0
    self.goodSlope = 0.0
    self.slopeError = 0.0
    self.length = 0.0
    self.effectiveLength = 0.0
    self.wEffLength = 0.0
    self.wStiffness = 0.0
    self.modulus = 0.0
    self.averageMod = 0.0
    self.modUncertainty = 0.0
    self.percentModUncertainty = 0.0
    self.ultStrength = 0.0
    self.failureMode = ''
    self.individualCNTDiameters = []
    self.avgCNTDiameter = 0.0
    self.fitParameters = [0.1, 3, 2, 120]
    self.initialGuesses = [1, 2]
    self.droppedPoints = 0
    self.segment = 0
    self.plots = {}
    self.wireData: DataFrame = DataFrame()
    self.refData: DataFrame = DataFrame()
    self.ddTestConditions = {}
    self.semData = {}
    self.td: DataFrame = DataFrame()
    self.tdRaw: DataFrame = DataFrame()
    self.tdNoOffset: DataFrame = DataFrame()
    self.tdWire: DataFrame = DataFrame()
    self.tdPost: DataFrame = DataFrame()
    self.force: DataFrame = DataFrame()
    self.stress: DataFrame = DataFrame()
    self.x = []
    self.y = []
    self.sx = 5 # uncertainty in post deflection measurement in um
    self.sy = 5 # uncertainty in wire deflection measurement in um
    self.xfit = []
    self.yfit = []
    self.yfitForce = []
    self.fitOutput = {'slope': [], 'stdError': [], 'endIndex': [], 'endXValue': [],
                      'endYValue': [], 'redChiSqr': [], 'xx': [], 'yy': []}
    self.dynamicFitResults: DataFrame = DataFrame()
    self.metaData = {}
    self.row = row
    self.post = post
    self.problemPost = problemPost
    self.curveClassification =  {'Initial_deflection': [], 'Failure': [], 'After_failure': []}
    self.inExcel = False
    self.dynamicallyUpdatingXandY = config.dynamicPostDeflectionRange

  def resetDeflectionData(self):
    # self.sample.loadTrackerData()
    self.td = self.tdRaw
    self.tdNoOffset = self.tdRaw
    self.zeroOffset()
    
  def zeroOffset(self):
    firstRow = self.td.values[0]
    transformer = lambda row: row - firstRow
    self.td = self.td.transform(transformer)
    # self.setXandY()  

  def setSample(self, sample: Sample):
    self.sample = sample
    
  def setGoodSlope(self, goodSlope):
    self.goodSlope = goodSlope
    
  def setSlopeStdError(self, stdErr):
    self.slopeError = stdErr

  def setDdTestConditions(self, data):
    self.ddTestConditions = data
    self.setLength()
    self.setEffectiveLength()
    self.setWireEffectiveLength()
  
  def setLength(self):
    pLength = self.ddTestConditions.get('slope_corrected_p_height')
    if isinstance(pLength, float):
      self.length = pLength
  
  def setEffectiveLength(self):
    eLength = self.ddTestConditions.get("p_effective_length")
    if isinstance(eLength, float):
      self.effectiveLength = eLength
      
  def setWireEffectiveLength(self):
    wELength = self.ddTestConditions.get("w_effective_length")
    if isinstance(wELength, float):
      self.wEffLength = wELength
    
  def setSemData(self, data):
    self.semData = data
    self.setIndCNTDiams()
    self.setFailureMode()
    self.setAvgCNTDiamAndStats()

  def setIndCNTDiams(self):
    for i in range(1, len(self.semData)):
      d = self.semData[f'diameter_{i}']
      if(np.isnan(d) or d == -1 or d == 0.0):
        break
      self.individualCNTDiameters.append(d)

  def setModulus(self, modulus):
    self.modulus = modulus
      
  def setFailureMode(self):
    mode = self.semData.get("failure_mode")
    if isinstance(mode, str):
      self.failureMode = mode
  
  def setAvgCNTDiamAndStats(self):
    diam = self.semData.get('Average')
    if isinstance(diam, float):
      self.avgCNTDiameter = diam

  def storePlot(self, key, fig, ax):
    self.plots[key] = {'fig': fig, 'ax': ax}

  def savePlots(self):
    directory = config.data_directory + "/" + config.output_directory
    address = self.getAddressString()
    for key, figs in self.plots.items():
      fig = figs['fig']
      if config.savePostPlotAsSVG:
        fig.savefig(directory + f'/{address}_{key} at {config.truncationValue}.svg')
      if config.savePostPlotOnPDF:
        config.pdf.savefig(fig)

  def calcTdPost(self):
    if self.refData.empty and self.wireData.empty:
      return
    if not self.wireData.empty:  
      self.tdWire = self.wireData["totalDistanceTraveled"]
    else: 
      return
    data = abs(self.refData["totalDistanceTraveled"] - self.wireData["totalDistanceTraveled"])
    data = data.dropna()
    self.tdPost = data
    self.tdRaw = DataFrame().assign(post=self.tdPost, wire=self.tdWire)
    self.td = self.tdRaw
    self.tdNoOffset = self.tdRaw
    # self.dropInitialPoints(self.droppedPoints)
    # self.truncateData() 
  
  def setXandY(self):
    self.x = self.td.post.to_numpy()
    self.y = self.td.wire.to_numpy()

  def dynamicallyUpdateXandY(self, n):
    initialLength = self.td[self.td.post <= config.minDynamicDelfectionRange].index[-1]
    if initialLength+n > self.td.shape[0]:
      self.dynamicallyUpdatingXandY = False
      return
    self.tdPost = self.td.post.iloc[0:(initialLength+n)]
    self.x = self.tdPost.to_numpy()
    self.tdWire = self.td.wire.iloc[0:(initialLength+n)]
    self.y = self.tdWire.to_numpy()

  def updateFitResults(self):
    self.dynamicFitResults = pd.DataFrame(self.fitOutput)
    i = self.dynamicFitResults.idxmin(numeric_only=True).redChiSqr
    self.fitOutput = self.dynamicFitResults.loc[i].to_dict()# removes arrays
    ei = self.fitOutput['endIndex']
    self.tdPost = self.td.post.iloc[0:ei+1]
    self.x = self.tdPost.to_numpy()
    self.tdWire = self.td.wire.iloc[0:ei+1]
    self.y = self.tdWire.to_numpy()
    self.td = DataFrame().assign(post=self.tdPost, wire=self.tdWire)
    self.xfit = self.fitOutput['xx']
    self.yfit = self.fitOutput['yy']

  def truncateData(self):
    if config.truncatePostDeflection == True:
      self.td = self.td[self.td.post <= config.truncationValue]
      self.td.reset_index(drop=True, inplace=True)
      #NOTE: see what Rob thinks of these graphs vs the truncate then offset graphs
      self.tdNoOffset = self.tdNoOffset[self.tdNoOffset.post <= (config.truncationValue + self.tdNoOffset.post[0])]
      self.tdNoOffset.reset_index(drop=True, inplace=True)
      self.setXandY()
  
  def dropInitialPoints(self, numPointsToDrop):
    self.td = self.td.truncate(before = numPointsToDrop)
    self.td.reset_index(drop=True, inplace=True)
    self.tdNoOffset = self.tdNoOffset.truncate(before = numPointsToDrop)
    self.tdNoOffset.reset_index(drop=True, inplace=True)
    self.droppedPoints = numPointsToDrop
    self.zeroOffset()
    self.setXandY()
  
  def calculations(self):
    self.calcModulus()
    self.calcModUncertainty()
    self.calcwStiffness()
    self.calcForce()
    self.calcStress()

  def calcwStiffness(self):
    Ew = self.ddTestConditions.get("w_modulus")
    dw = self.ddTestConditions.get("w_diameter")
    lw = self.ddTestConditions.get("w_effective_length")
    self.wStiffness = (Ew*3*np.pi*(dw**4))/(64*(lw**3))

  def calcForce(self):
    k = self.wStiffness 
    yw = self.td['wire']
    self.force = pd.DataFrame(yw.to_numpy()/10**6 * k)
    self.yfitForce = (self.yfit/10**6 * k)
  
  def calcModulus(self):
    A = self.calcA()
    R = self.goodSlope * A
    Ew = self.ddTestConditions.get("w_modulus")
    Ep = R * Ew
    self.modulus = Ep 
  
  def calcStress(self):
    L = self.ddTestConditions.get("p_effective_length")
    dp = self.ddTestConditions.get("p_diameter")
    c = dp/2
    I = (np.pi * dp**4)/64
    self.stress = (L*c)/I * self.force
    
  def calcA(self):
    lw = self.ddTestConditions.get("w_effective_length")
    lp = self.ddTestConditions.get("p_effective_length")
    dw = self.ddTestConditions.get("w_diameter")
    dp = self.ddTestConditions.get("p_diameter")
    
    A = (dw**4 * lp**3)/(dp**4 * lw**3)
    return A
  
  def calcPostLengthUncertainty(self):
    dVertDis = self.ddTestConditions.get("p_height_uncertainty")
    dNomWireContHeight = self.ddTestConditions.get("wire_contact_height_uncertainty")
    dPl = np.sqrt(dVertDis**2 + dNomWireContHeight**2)
    return dPl
    
  def calcWireLengthUncertainty(self):
    dWireLengthUncert = self.ddTestConditions.get("w_length_uncertainty")
    dWireLengthOffsetUncert = self.ddTestConditions.get("w_length_offset_uncertainty")
    dWl = np.sqrt(dWireLengthUncert**2 + dWireLengthOffsetUncert**2)
    return dWl
    
  def calcDeltaA(self, A):
    lw = self.ddTestConditions.get("w_effective_length")
    lp = self.ddTestConditions.get("p_effective_length")
    dw = self.ddTestConditions.get("w_diameter")
    dp = self.ddTestConditions.get("p_diameter")
    
    dlw = self.calcWireLengthUncertainty()
    dlp = self.calcPostLengthUncertainty()
    ddw = self.ddTestConditions.get("w_diam_uncertainty")
    ddp = self.ddTestConditions.get("p_diam_uncertainty")
    
    dA = np.sqrt((4*(ddw/dw))**2 + (3*(dlp/lp))**2 + (4*(ddp/dp))**2 + (3*(dlw/lw))**2)*A
    return dA
  
  def calcDeltaR(self, R, A, dA):
    m = self.goodSlope
    dm = self.slopeError
    dR = np.sqrt((dA/A)**2 + (dm/m)**2)*R
    return dR
  
  def calcModUncertainty(self):
    A = self.calcA()
    R = self.goodSlope * A
    Ew = self.ddTestConditions.get("w_modulus")
    Ep = self.modulus
    
    dA = self.calcDeltaA(A)
    dR = self.calcDeltaR(R, A, dA)
    dEw = self.ddTestConditions.get("w_modulus_uncertainty")
    
    dEp = np.sqrt((dR/R)**2 + (dEw/Ew)**2)*Ep
    self.modUncertainty = dEp
    self.percentModUncertainty = dEp/Ep
    
  def getAddressString(self):
    return f"{self.sample.name}_row-{self.row}_post-{self.post}"

  def getRowPostString(self):
    return f"r{self.row}:p{self.post}"

def int_or_float(s):
  try:
    return int(s)
  except ValueError:
    return float(s)

def makeDictKeyValueStringsList(dictionary):
  strList = []
  for k, v in dictionary.items():
    strList.append(str(k) + ': ' + str(v))
  return strList

def makeDictValueStringsList(dictionary):
  strList = []
  for v in dictionary.values():
    strList.append(str(v))
  return strList

def makeClassificationLists():
  curveSubsectionsList = {}
  for subsection in config.curveSubsections:
    lst = makeDictValueStringsList(config.curveSubsections[subsection])
    curveSubsectionsList[subsection] = lst
  return curveSubsectionsList

def userFitCheck():
    title = "Linear Regression Check"
    msg = "Is the fit sufficient?"
    choices = ["Yes","No","Stop program"]
    reply = buttonbox(msg, title, choices=choices)
    return reply

def userAutoOrCheck(sample):
    # global checkMode
    title = "AutoMode or user check"
    msg = f"Check each individual plot of {sample.name}?"
    choices = ["Yes","No","Stop program"]
    reply = buttonbox(msg, title, choices=choices)
    if reply == "Yes":
      sample.checkMode = True
      return False
    elif reply == "No":
      sample.checkMode = False
      return True
    elif reply == "Stop program":
        sys.exit(0)
    return reply

def userUpdateFitParameters(parameters):
  if config.numOfLineFits == 'Multi' or config.numOfLineFits == 'Both':
    title = "Enter new Linear Tree parameters"
    msg = f"Previous parameters are: {parameters[0]}, {parameters[1]}, {parameters[2]}, and {parameters[3]}, respectively. Bounds are > 6, > 3, >1, and 10 < bins < 120, respectively."
    fieldNames = ["min_samples_split","min_samples_leaf","max_depth","max_bins"]
    fieldValues =  multenterbox(msg, title, fieldNames)
    while 1:
        errmsg = ""
        for i, name in enumerate(fieldNames):
            if fieldValues[i].strip() == "":
              errmsg += "{} is a required field.\n\n".format(name)
        if errmsg == "":
            break # no problems found
        fieldValues = multenterbox(errmsg, title, fieldNames, fieldValues)
        if fieldValues is None:
            break
    return list(map(int_or_float, fieldValues))
  elif config.numOfLineFits == 'One':
    title = "Enter new ODR initial guesses"
    msg = f"Previous gueses are: Slope: {parameters[0]} and Intercept: {parameters[1]}."
    fieldNames = ["Slope_initial_guess", "Intercept_initial_guess"]
    fieldValues =  multenterbox(msg, title, fieldNames)
    while 1:
        errmsg = ""
        for i, name in enumerate(fieldNames):
            if fieldValues[i].strip() == "":
              errmsg += "{} is a required field.\n\n".format(name)
        if errmsg == "":
            break # no problems found
        fieldValues = multenterbox(errmsg, title, fieldNames, fieldValues)
        if fieldValues is None:
            break
    return list(map(int_or_float, fieldValues))

def userClassifyDeflections(name):
  curveClassified = False
  subsectionClassified = False
  subsectionClassifications = {}
  while not curveClassified:
    curveSubsectionsList = makeClassificationLists()
    for subsection in curveSubsectionsList:
      if subsection == 'After_failure' and subsectionClassifications['Failure'] == 'No_failure':
        subsectionClassifications[subsection] = 'N/A'
        continue
      subsectionClassified = False
      while not subsectionClassified:
        title = f"Classify {subsection} curve"
        msg = f"How does the {subsection} curve of " + name + " behave?"
        choices = curveSubsectionsList[subsection]
        choices.append("Add new classification")
        choices.append("STOP PROGRAM")
        classification = choicebox(msg, title, choices)
        if classification == "Add new classification":
          curveClassified = False
          subsectionClassified = False
          curveSubsectionsList = userAddClassification(subsection, choices)
        elif classification =='STOP PROGRAM':
          sys.exit(0)
        else:
          curveClassified = True
          subsectionClassified = True
      subsectionClassifications[subsection] = classification 
  return subsectionClassifications
  
def userAddClassification(subsection, choices):
  title = f"Add a new {subsection} classification"
  glue = '\n'
  msg = glue.join(["Type the classification you want to add. The latest classification is:",
                   f"{choices[-3]}.",
                   "If you do not want to add a new classification, type 'None'."])
  newClassification = enterbox(msg, title)
  if newClassification == 'None':
    pass
  else:
    config.curveSubsections[subsection][len(config.curveSubsections[subsection]) + 1] = newClassification
  return makeClassificationLists()

def userPickLinearSegment(slopes):
    title = "Pick the linear segment to use"
    msg = "Starting from the left most segment, i.e. 0, choose the slope to use for calculations."
    choices = range(len(slopes))
    segment = choicebox(msg, title, choices)
    return segment

def userMarkAsProblemPost():
  title = "Mark as problem post"
  msg = "Is the data unusual, poorly fit, or requires further inspection?"
  pp = ynbox(msg, title)
  return pp

def userInterface(ltParameters, initialGuesses, problemPost, segment, post):
  if config.numOfLineFits == 'Multi' or config.numOfLineFits == 'Both':
    if post.sample.checkMode == True:  
      reply = userFitCheck()
      if reply == "Yes":
        segment = int(userPickLinearSegment(post.fitOutput['slope']))
        problemPost = userMarkAsProblemPost()
        return True, segment, ltParameters, problemPost
      elif reply == "No":
        dropInitialPoints(post)
        ltParameters = userUpdateFitParameters(ltParameters)
        return False, segment, ltParameters, False
      elif reply == "Stop program":
        sys.exit(0)
    elif post.sample.checkMode == False:
        return True, segment, ltParameters, False
  elif config.numOfLineFits == 'One':
    if post.sample.checkMode == True:  
      reply = userFitCheck()
      # reply = 'Yes'
      if reply == "Yes":
        problemPost = userMarkAsProblemPost()
        return True, segment, initialGuesses, problemPost
      elif reply == "No":
        dropInitialPoints(post)
        #initialGuesses = userUpdateFitParameters(initialGuesses)
        return False, segment, initialGuesses, False
      elif reply == "Stop program":
        sys.exit(0)
    elif post.sample.checkMode == False:
      return True, segment, initialGuesses, False

def dropInitialPoints(post):
  if config.readExcel and not post.sample.checkMode:
    numPointsToDrop = post.droppedPoints
  else:
    numPointsToDrop = userDropInitialPoints(post)
  post.dropInitialPoints(numPointsToDrop)


def userDropInitialPoints(post):
  title = "Drop initial points"
  msg = "Choose how many points to drop from the beggining. Only input integers or 'reset'."
  points = enterbox(msg, title)
  if points == 'reset':
    post.resetDeflectionData()
    return 0
  else:
    return int(points)



# NOTE: Main data pipeline starts here

def analyzePosts(sample):
  goodFit = False
  testedPosts = sample.testedPosts
  print(sample.name)
  while not goodFit:
    if config.testMode:
      goodFit = True
    else: 
      goodFit = userAutoOrCheck(sample)
    for post in testedPosts:
      if post.td.size != 0:
        dropInitialPoints(post)
        post.truncateData() 
        findGoodFit(post)
    if config.minimumDataPoints != 'none':
      testedPosts = [post for post in testedPosts if len(post.x) >= config.minimumDataPoints]
      sample.testedPosts = testedPosts

  return testedPosts
                
def findGoodFit(post):
  goodFit = False
  while not goodFit:
    goodFit = fitPlotCalcAndSave(post)

def fitPlotCalcAndSave(post):
  ltParameters = post.fitParameters
  initialGuesses = post.initialGuesses
  dynamicFitResults = {}
  if config.numOfLineFits == 'One': 
    if config.dynamicPostDeflectionRange:
      n = 0
      while post.dynamicallyUpdatingXandY:
        n = n+1
        post.dynamicallyUpdateXandY(n)
        if post.dynamicallyUpdatingXandY:
          oneLineODRfit(post)
      post.updateFitResults()
    else:
      oneLineODRfit(post)
      post.updateFitResults()
    if config.plotMode:
      if config.specificPost == post.getAddressString() or config.specificPost == 'none':
        # fig = plotForceDeflectionGraph(post)
        fig = plotDeflectionGraph(post)
      if post.sample.checkMode:
        plt.show()
    goodFit, segment, initialGuesses, problemPost = userInterface(ltParameters, initialGuesses, post.problemPost, post.segment, post)
    post.fitOutput['endIndex']= post.fitOutput['endIndex']-post.droppedPoints
  # elif config.numOfLineFits == 'Multi':
  #   slopes, stdErrors, endXs = multiLineLTfit(x, y, post, ltParameters)
  #   if config.plotMode:
  #     fig = plotDeflectionGraph(x, y, xx, yy, post)
  #     plt.show()
  #   goodFit, segment, ltParameters, problemPost = userInterface(slopes, ltParameters, initialGuesses, post.problemPost, post.segment, post)
  #   endXs[segment] = endXs[segment]-post.droppedPoints
  # elif config.numOfLineFits == 'Both':
  #   xx, yy, slopesODR, stdErrorsODR, endXsODR = oneLineODRfit(x, y, post)
  #   slopesLT, stdErrorsLT, endXsLT = multiLineLTfit(x, y, post, ltParameters)
  #   goodFit, segment, ltParameters, problemPost = userInterface(slopes, ltParameters, initialGuesses, post.problemPost, post.segment, post)
  #   slopes = [*slopesODR, *slopesLT]
  #   endXs[segment] = endXsLT[segment]-post.droppedPoints
  #   stdErrors = stdErrorsLT
  
    # if(post.sample.name == '202-I' and post.post == 22 and post.row == 4):
    #   print(post.sample.name)
    #   directory = config.data_directory + "/" + config.output_directory
    #   fig.savefig(directory + '/202-I_r4_p22.svg', bbox_inches='tight')
    # post.savePlot('Deflection plot with fits', fig)
  
  goodSlope = post.fitOutput['slope']
  stdError = post.fitOutput['stdError']
  post.setGoodSlope(goodSlope)
  post.setSlopeStdError(stdError)
  post.problemPost = problemPost
  post.calculations()
  # post.ultStrength = post.force.to_numpy()[post.fitOutput['endIndex']-1][0]
  # x = post.td['tdPost'].to_numpy()
  # y = post.td['tdWire'].to_numpy() 
  if goodFit:
    if (config.saveEachPostPlot and config.plotMode) or config.specificPost == post.getAddressString():
      plotForceDeflectionGraph(post)
      plt.ylim(-5.669513036154814, 116.75735602527222)
      plt.xlim(-0.7874829987535152, 10.496276705116763)
      # post.plots["Force Deflection plot"]['ax'].set_ylim(top=115)
      post.savePlots()
    post.fitParameters = ltParameters
    post.segment = segment
    post.initialGuesses = initialGuesses
  return goodFit

def oneLineODRfit(post):
  def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]

  linear = odr.Model(f)
  mydata = odr.RealData(post.x, post.y, sx=post.sx, sy=post.sy)
  myodr = odr.ODR(mydata, linear, beta0=[post.initialGuesses[0], post.initialGuesses[1]])
  myoutput = myodr.run()
  # myoutput.pprint()
  xx = np.array([post.x[0], post.x[-1]])
  yy = f(myoutput.beta, xx)
  post.xfit=xx
  post.yfit=yy
  # if config.plotMode and not config.testmode:
  #   plt.scatter(post.x, post.y)
  #   plt.show()
  post.fitOutput['slope'].append(myoutput.beta[0])
  post.fitOutput['stdError'].append(myoutput.sd_beta[0])
  post.fitOutput['endIndex'].append(len(post.x)-1)
  post.fitOutput['endXValue'].append(post.x[-1])
  post.fitOutput['endYValue'].append(post.y[-1])
  post.fitOutput['redChiSqr'].append(myoutput.res_var)
  post.fitOutput['xx'].append(xx)
  post.fitOutput['yy'].append(yy)

# NOTE: Main data pipeline ends here

# NOTE: Multi-line functions
""" def multiLineLTfit(x, y, post, ltParameters):
  x = x.reshape(-1,1)
  lt = LinearTreeRegressor(
       base_estimator = LinearRegression(),
       min_samples_split= ltParameters[0],
       min_samples_leaf = ltParameters[1],
       max_depth = ltParameters[2],
       max_bins = ltParameters[3]
   ).fit(x, y)
  slopes, stdErrors, endXs = loopThroughLeaves(lt, x, y, post)
  return slopes, stdErrors, endXs
  
def calcStandardErrorOfSlope(X, y, model):
  stdErrors = []
  N = len(X)
  p = len(X[0]) + 1  # plus one because LinearRegression adds an intercept term
  
  X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
  X_with_intercept[:, 0] = 1
  X_with_intercept[:, 1:p] = X
  
  y_hat = model.predict(X)
  residuals = y - y_hat
  residual_sum_of_squares = residuals.T @ residuals
  sigma_squared_hat = residual_sum_of_squares / (N - p)
  var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
  for p_ in range(p):
    standard_error = var_beta_hat[p_, p_] ** 0.5
    stdErrors.append(standard_error)
    #print(f"SE(beta_hat[{p_}]): {standard_error}")
  return stdErrors[1]
  
def loopThroughLeaves(lt, x, y, post):
  startX = 0 
  endX = 0
  slopes = []
  stdErrors = []
  endXs = []
  leaves = lt.summary(only_leaves=(True))
  # print(lt.n_features_in_)
  for m in leaves.values():
    model = m.get('models')
    slope = model.coef_[0]
    slopes.append(slope)
    startX = endX
    endX += m.get('samples')
    endXs.append(endX)
    if config.plotMode:
      plt.plot(x[startX:endX], lt.predict(x[startX:endX]))
    stdError = calcStandardErrorOfSlope(x[startX:endX], y[startX:endX], model)
    stdErrors.append(stdError)
    # if not config.numOfLineFits:
    #   break
  return slopes, stdErrors, endXs """


def isFigEmpty(figure):
  """
  Return whether the figure contains no Artists (other than the default
  background patch).
  """
  contained_artists = figure.get_children()
  return len(contained_artists) <= 1

def checkBoxDimensions(plotName, fig, ax):
  bbox = ax.get_window_extent()
  print(f'{plotName} H: {bbox.height/fig.dpi}')
  print(f'{plotName} W: {bbox.width/fig.dpi}')


def plotForceDeflectionGraph(post, legend=False):
  f = post.force.values*10e6
  d = post.tdPost.values.reshape(-1,1)
  
  fig, ax = plt.subplots(figsize=(10,6))
  plt.scatter(d, f, s=100, c='black', alpha=(0.65))
  plt.xlabel('Post deflection ' + u'(\u03bcm)', fontsize=config.titleFontSize);plt.ylabel('Force ' + u'(\u03bcN)', fontsize=config.titleFontSize)
  if config.title:
    plt.title(f'{post.getAddressString()} {post.sample.infiltrationRecipe}', fontsize=config.titleFontSize)
  if config.plotFit:
    plt.plot(post.xfit, post.yfitForce*10e6, linewidth=3, alpha=0.85)
  if legend:
    plt.legend([post.fitOutput['slope'], post.fitOutput['redChiSqr']])
  post.storePlot('Force Deflection plot', fig, ax)
  if config.checkBoxDims:
    checkBoxDimensions(post.getAddressString(), fig, ax)
  return fig
    
def plotDeflectionGraph(post, legend=False):
  fig, ax = plt.subplots(figsize=(10,6))
  plt.scatter(post.x, post.y, s=100, c='black', alpha=(0.65))
  plt.xlabel('Post deflection ' + u'(\u03bcm)', fontsize=config.titleFontSize); plt.ylabel('Wire deflection ' + u'(\u03bcm)', fontsize=config.axisFontSize)
  if config.title:  
    plt.title(f'{post.getAddressString()} {post.sample.infiltrationRecipe}', fontsize=config.titleFontSize)
  if config.plotFit:
    plt.plot(post.xfit, post.yfit, linewidth=3, alpha=0.85)
  if legend:
    plt.legend([post.fitOutput['slope'], post.fitOutput['redChiSqr']])
  post.storePlot('Deflection plot', fig, ax)
  if config.checkBoxDims:
      checkBoxDimensions(post.getAddressString(), fig, ax)
  return fig
    
# def collectPostInfo(posts):
#   slopes, moduli, wcHeights, failureModes, rpAddresses = collectPostInfo(posts)
#   slopes = []
#   moduli = []
#   wcHeights = []
#   failureModes = []
#   rpAddresses = []
#   for p in posts:
#     if p.goodSlope != 0:
#       slopes.append(p.goodSlope)
#       moduli.append(p.modulus)
#       wcHeights.append(p.ddTestConditions.get('slope_corrected_wire_contact_height'))
#       failureModes.append(p.semData.get("failure_mode"))
#       rpAddresses.append(p.getRowPostString())
#   return slopes, moduli, wcHeights, failureModes, rpAddresses

def overviewGraphs(df, xAxis, yAxis,  xlabel, ylabel, title, xErr=None, yErr=None, errorBars=False, snsPlotType='scatterplot', titlesize=16,
                   hue=None, style=None, markerSize=200, tickFontSize=16, legend=False, 
                   truncateLegend=False, colorViolinEdge=False, savefig=True, ylog=False, 
                   boxOrien = 'v', boxOrder=None, hueOrder=None, titleBool=False,
                   markerPalette='tab10', violinEdgePalette='tab10', violinFillPallete='Pastel2'):
  fig, ax = plt.subplots(figsize=(10,6))
  if ylog:
    ax.set_yscale('log')
  if snsPlotType == 'scatterplot':
    sns.scatterplot(x=xAxis, y=yAxis, data=df, hue=hue, palette=markerPalette, style=style, s=markerSize, legend=legend)
    if errorBars:
      cmap = cm.get_cmap(markerPalette)
      df = df.sort_values(by=[xAxis])
      cols = [cmap(0.05), cmap(0.15), cmap(0.25)]
      ax.errorbar(df[xAxis], df[yAxis], xerr=df[xErr], yerr=df[yErr], fmt='', zorder=-1, ls='none',  ecolor=cols)
      # plt.errorbar(df[xAxis], df[yAxis], xerr=df[xErr], yerr=df[yErr], lolims=True, fmt='', zorder=-1, ls='none')
  elif snsPlotType == 'pointplot':
    sns.pointplot(x=xAxis, y=yAxis, data=df, hue=hue, join=False, ci='sd')
  elif  snsPlotType == 'catplot':
    sns.catplot(x=xAxis, y=yAxis, data=df, hue=hue, palette=markerPalette)
    plt.xticks(rotation=60)
  elif snsPlotType == 'boxplot':
    config.legendIn = True
    sns.boxplot(x=xAxis, y=yAxis, data=df, hue=hue, palette=markerPalette, 
                orient=boxOrien, order=boxOrder)
  elif snsPlotType == 'swarmplot':
    sns.swarmplot(x=xAxis, y=yAxis, data=df, hue=hue, palette=markerPalette)
    #plt.xticks(rotation=60)
  elif snsPlotType == 'violin_swarm':
    if ylog:
      dflog = df.copy()
      dflog[yAxis] = np.log10(dflog[yAxis])
    if violinFillPallete == 'white':
      sns.violinplot(x=xAxis, y=yAxis, data=dflog, inner='quartile', color='white')
    else:
      sns.violinplot(x=xAxis, y=yAxis, data=dflog, inner='quartile', palette=violinFillPallete)
    if colorViolinEdge:
      for i in range(len(ax.collections)):
        ax.collections[i].set_edgecolor(sns.color_palette(violinEdgePalette)[i])
    sns.swarmplot(x=xAxis, y=yAxis, data=dflog, hue=hue, palette=markerPalette)
    plt.legend([],[], frameon=False)
    if ylog:
      ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
      ax.yaxis.set_minor_formatter(NullFormatter())
      ymin, ymax = ax.get_ylim()
      tick_range = np.arange(np.floor(ymin), ymax)
      ax.yaxis.set_ticks(tick_range)
      ax.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)
  elif snsPlotType == 'box_swarm':
    sns.boxplot(x=xAxis, y=yAxis, data=df, palette=markerPalette, orient=boxOrien, order=boxOrder)
    sns.swarmplot(x=xAxis, y=yAxis, data=df, color=".25")
    
  if legend:
    plt.legend()
    if not config.legendIn and config.legend:
      plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    
    if truncateLegend:
      handles, labels = ax.get_legend_handles_labels()
      index_infiltration_title = labels.index('Infiltration_Time')
      index_sample_title = labels.index('Sample')
      ax.legend(handles[index_infiltration_title+1:index_sample_title], labels[index_infiltration_title+1:index_sample_title], title='Infiltration Times (s)', 
              bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
      ax.legend(handles[index_infiltration_title+1:index_sample_title], labels[index_infiltration_title+1:index_sample_title], title='Infiltration Times (s)',
                fontsize=16, title_fontsize=16)
      # ax.legend(handles[index_infiltration_title+1:index_sample_title], labels[index_infiltration_title+1:index_sample_title], title='Infiltration Times (s)', 
              # bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
  if ylog:
    yticks = ax.get_yticks()
    values = np.log10(yticks)
    ylabels = [f'1e{int(val)}' for val in values]
    ax.set_yticklabels(ylabels, fontsize=config.tickFontSize)
  plt.xlabel(xlabel, fontsize=config.titleFontSize)
  plt.ylabel(ylabel, fontsize=config.titleFontSize)
  plt.rc('xtick', labelsize=config.tickFontSize) 
  plt.rc('ytick', labelsize=config.tickFontSize) 
  # plt.tight_layout()
  if config.checkBoxDims:
    checkBoxDimensions(snsPlotType, fig, ax)
  # plt.show()
  if titleBool:
    plt.title(title, fontsize=config.titleFontSize)
  # plt.show()
  if savefig:
    config.pdf.savefig(fig)#, bbox_inches='tight')
    directory = config.data_directory + "/" + config.output_directory
    fig.savefig(directory + f'/{title} at {config.truncationValue}.svg')#, bbox_inches='tight')
  return fig
  
def saveSampleLevelGraphs(posts):
  slopesVsPosts = plt.figure(figsize=(10,6))
  plt.scatter(rpAddresses, slopes, s=20, c='black')
  plt.title(posts[0].sample.name)
  plt.xlabel('Address'); plt.ylabel('Deflection slope')
  plt.ylim(bottom = 0, top = max(slopes) + 2)
  posts[0].sample.savePlot('Slopes vs Posts', slopesVsPosts)
  
  modsVsPosts = plt.figure(figsize=(10,6))
  plt.scatter(rpAddresses, moduli, s=20, c='black')
  plt.title(posts[0].sample.name)
  plt.xlabel('Address'); plt.ylabel('Modulus')
  # plt.ylim(bottom = 0)
  posts[0].sample.savePlot('Modulus vs Posts', modsVsPosts)
   
  # slopesVsWcheight = plt.figure(figsize=(10,6))
  # plt.scatter(wcHeights, slopes, s=20, c='black')
  # plt.title(posts[0].sample.name)
  # plt.xlabel('Effective Post Height (um)'); plt.ylabel('Deflection slope')
  # plt.ylim(bottom = 0, top = max(slopes) + 2)
  # plt.ticklabel_format(axis='x', style='sci', scilimits=(-6,-6))
  # posts[0].sample.savePlot('Slopes vs Effective Post Height', slopesVsWcheight)
  
  # modsVsWcheight = plt.figure(figsize=(10,6))
  # plt.scatter(wcHeights, moduli, s=20, c='black')
  # plt.title(posts[0].sample.name)
  # plt.xlabel('Effective Post Height (um)'); plt.ylabel('Modulus')
  # plt.ylim(bottom = 0)
  # plt.ticklabel_format(axis='x', style='sci', scilimits=(-6,-6))
  # posts[0].sample.savePlot('Modulus vs Effective Post Height', modsVsWcheight)

def grabSamples(excelFileName = '', sampleNames=[]):
  directory = config.data_directory + "/" + config.output_directory
  filename = config.trackerFiles + ".xlsx"
  if excelFileName != '':
    filename = excelFileName + ".xlsx"
  samples = []
  sampleNamesString = ""
  
  if config.readExcel:    
    readData = pd.read_excel(directory + "/" + filename, header=0, index_col=0)
    readData = readData.sort_index()  
    for r in readData.itertuples():
      row = r._asdict()
      name = row['Index']
      if name in sampleNames:
        #use the last sample object added to the list
        samples[-1].updateWithExcelData(row)
      else:
        sample = Sample(name)
        sample.updateWithExcelData(row)
        samples.append(sample)
        sampleNames.append(name)
        sampleNamesString = f"{row['Index']} " + sampleNamesString
    
    for sample in samples:
      sample.setTestedPosts()
  else:
    for name in sampleNames:
      samples.append(Sample(name))
      sampleNamesString = f"{name} " + sampleNamesString
    
  return samples, sampleNamesString, directory
