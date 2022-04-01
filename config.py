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
global trackerFiles
global pdf
global saveEachPostPlot
global classifyDeflections
global curveSubsections
global truncatePostDeflection
global truncationValue
global numOfLineFits