dim_columns = 100
dim_rows = 100
data_directory = "Data"
tracker_directory = "Tracker output"
tracker_bad_posts = "Bad posts"
tracker_good_posts = "Good posts"
metadata_directory = "Metadata"
media_directory = "Videos and images"
output_directory = "Program output"
fea_data_file = "FEA Results_20230513"
#Sample level excel output filename
sampleExcelFileName = 'Sample averages with outliers.xlsx'
#Averages per infiltration time
infTimeExcelFileName = 'Infiltration time averages with outliers.xlsx'
# metadata_filenames = [ "Carbon_recipes.xlsx", "Posts_characteristics.xlsx", "Posts_SEM_data.xlsx", "Samples.xls" ]

SampleCharacteristicFiles = [ "Samples.xlsx" ]
InfiltrationConditionFiles = [ "Carbon_recipes.xlsx" ]
DdTestConditionFiles = [ "DD_test_conditions.xlsx" ]
SemDataFiles = [ "Posts_SEM_data.xlsx" ]

global testMode
global plotMode
global readExcel
global excelName
global trackerFiles
global pdf
global saveEachPostPlot
global classifyDeflections
global curveSubsections
global truncatePostDeflection
global truncationValue
global numOfLineFits
global axisFontSize
global titleFontSize

#Test mode will run the entire program with no input from user.
testMode = True
#Plot mode will plot the deflection curves of each post
plotMode = True
#If saveEachPostPlot, the deflection curve and linear fits of each post will be saved to the pdf
saveEachPostPlot = False
savePostPlotOnPDF = True
savePostPlotAsSVG = True
#If saveSpecificPost, the deflection of a specific post will be saved to pdf and as svg
saveSpecificPost = True
if saveSpecificPost:
  specificPost = '202-I_row-4_post-22'
  saveEachPostPlot = False
else:
  specificPost = 'none'

#Plot fit will plot the linear fit over the deflection data points on each post deflection graph. 
#Has no effect if plotMode is False.
plotFit = True
#Do you want to read in previously processed posts from an excel file?
readExcel = True
excelName = 'Final output at 10 microns with outliers'
#Declare which tracker files use want to use, i.e. Bad posts, Good posts, or All posts
#This will change which directory the tracker files are pulled from. 
trackerFiles = "All posts"
# Delare sample name or names you want to do in this run. If you are reading in an excel file, sampleNames is not used. 
# ['188-I', '190-I', '194-U', '195-I', '198-U', '199-I', '200-U', '201-I', '202-I', '203-I']
sampleNames = ['188-I', '194-U', '195-I', '198-U', '199-I', '200-U', '201-I', '202-I', '203-I']

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
# Do you want to truncate the post deflection curves at some value of microns?
truncatePostDeflection = True
truncationValue = 10
# minimumDataPoints is an integer or 'none'. Does not plot or include in the excel deflections with less than 
minimumDataPoints = 8
# Do you want to use a dynamic deflection range starting at 10 um and up to minimize reduced chi squared?
# Will default to False if truncatePostDeflection == True
# minDynamicDelfectionRange will set the minimum distance to consider for linear fits
dynamicPostDeflectionRange = True
if truncatePostDeflection:
  dynamicPostDeflectionRange = False
  minDynamicDelfectionRange = 10
# Plot font sizes
axisFontSize = 18
tickFontSize = 18
titleFontSize = 20

#Sample plots characteristics
paintOverFigure = False  #Iteratively plot samples on the same figure. Best used with forceColor.
oneGraph = False #collects all data into single data frame and plots on one graph. Allows easy hue coloring. Best with specific infiltration levels
offsetToZero = True
forceColor = False
if forceColor:
  # forcedColor = ['Purples', 'bone', 'Reds', 'Greens']
  forcedColor = ['Purples', 'bone', 'Reds', 'Greens']
legend = True
title = False
plotFits = False
plotDeflection = False
plotFullDeflection = False
fullDeflTruncationValue = 50
plotForce = True
hue = 'Location' #'Location' or 'Sample'
legendIn = False
tightLayout = False
checkBoxDims = False
matchYscale = False #matches y axis scale for graphs of the same infiltration level. Scales to the largest.
limxAxis = False
if limxAxis:
  xlimit = 55
#   infiltrationTime is either an infiltration time or 'all'.
#   It will limit the sample plots to that infiltration time.
infiltrationTime = 15

#Plot Overview graphs
plotOverviewGraphs = True