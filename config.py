dim_columns = 100
dim_rows = 100
data_directory = "Data"
tracker_directory = "Tracker output"
tracker_bad_posts = "Bad posts"
tracker_good_posts = "Good posts"
metadata_directory = "Metadata"
media_directory = "Videos and images"
output_directory = "Program output"
# metadata_filenames = [ "Carbon_recipes.xlsx", "Posts_characteristics.xlsx", "Posts_SEM_data.xlsx", "Samples.xls" ]

SampleCharacteristicFiles = [ "Samples.xlsx" ]
InfiltrationConditionFiles = [ "Carbon_recipes.xlsx" ]
DdTestConditionFiles = [ "DD_test_conditions.xlsx" ]
SemDataFiles = [ "Posts_SEM_data.xlsx" ]

global testMode
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

#Test mode will run the entire program with no input from user.
testMode = True
#Do you want to read in previously processed posts from an excel file?
readExcel = True
excelName = 'Final output at 10 microns'
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
truncationValue = 10
