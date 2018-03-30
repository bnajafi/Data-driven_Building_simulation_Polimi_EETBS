import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import getpass
import os

# First let's create the path to the files

userName = getpass.getuser() # this line checks for the username and chooses the corresponding directory accordingly
if userName=="behzad":
    Genomic_DataSet_directory = "C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/data/raw/"
    weatherDataSet_directory = "C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/data/external/weather/"

# here you can simply add another "if userName=="yourName"  and provide the directory in which you have included the temp and meta datasets and that of the weather data
# so that you would not need to change the script to work on your PC

# Next, we insert the names of the main dataset temp_open_utc (which includes all of the buildign consumption values) and the meta dataset (Which includes the information about different buildings
dataSetFileName = "temp_open_utc.csv"
metaDataFileName = "meta_open.csv"

# Now I simply add this to the main folder so that I would create the path to the temp and meta files, I would use os.path.join to do so
path_temporalData= os.path.join(Genomic_DataSet_directory,dataSetFileName)
path_metaData =  os.path.join(Genomic_DataSet_directory,metaDataFileName)

""" Note on creating the path to the file! 
Why do we use such a complicated syntax (os.path.join), we could simply use the following:
path_temp = Genomic_DataSet_directory+dataSetFileName 
path_metaData = Genomic_DataSet_directory
The problem is that this will only work on Windows and not on a Mac, os.path.join handles this problem 
"""
# now let's use Pandas read_csv the import the temperalData and MetaData datasets as Pandas dataframes
DF_temporalData = pd.read_csv(path_temporalData,index_col="timestamp", parse_dates=True).tz_localize("utc")
# pay attention that usling index_col argument we impose the "timestamp" column as the index and we ask pandas to parse its dates as timestamps
# using tz_localize, we are imposing the time zone to be the Coordinated Universal Time (UTC)


DF_metaData = pd.read_csv(path_metaData,index_col = "uid", parse_dates=["datastart","dataend"], dayfirst=True) 
# simirly using index_col argument we impose the "uid" column as the index and we ask pandas to parse  the datestart and dateend clumns
# as timestamps in which the day is given before the month --> dayfirst = True

# Let's have a look on the head (first 5 rows by default) of the temporalData dataframe,

DF_temporalData.head() 

# We can see that in the temporal data dataframe the index is the dates starting from Jan 1st 2010, till Jan 1st 2016
# each column instead correspond to a specific building 
# but how come most of the values of electrical consumptions in first timestamps are NaN (not a number)? the reason is that the measurement for almost all of the building was not
# started in 2010, and was also finished before the end of 2015, so how can we find the start date and end date of measurment for each building, these two dates along 
# with many other useful information are given in the metaData ! before having a look on the metadata let's see more general information about the temporalData dataset.
DF_temporalData.info()
# we can see that we have 507 columns in this dataset each corresponding to one building and 40940 rows each of which corresponds to a specfic date 
#between  2010-01-01 08:00:00+00:00 to 2016-01-01 06:00:00+00:00

# Next have a look on the metaData dataframe
DF_metaData.head()
# we can see that in this dataframe the rows are the name of the buildings and the columns are the prperties of each building,
DF_metaData.info()
# using the info we can see that this DataFrame has 507 rows (number of buildings ) and 18 columns (each of which is a prperty of the building) to see these properties:
# in order to see these properties better let's transpose 
DF_metaData.columns.tolist()

# In order to have a more facilitated visualisation of the metaData let's transpose it (change the position of columns with the indices)
DF_metaData_transposed = DF_metaData.T

DF_metaData_transposed.head(19) # this simply shows the first 18 rows
# so in the meta data trasnpsed DataFrame we have 18 rows each of which represents a property and each of the 507 columns instead corresponds to one building
# So imagine that we would choose "office_Ellie" as an example building, all the properties of this example building can be found as:
chosenBuilding = "Office_Ellie"
Series_metaData_chosenBuilding = DF_metaData_transposed[chosenBuilding]
# so we can see its date Start Date End , time zone, surface , industry, subindustry etc. and the corresponding weather file
# Now let's see how we can use the metadata information to extract the related data of this building from the temporalData dataframe
# first let's extract the time zone, measurement start and end date of the building from the corresponding metaData serie

startDate_chosenBuilding = Series_metaData_chosenBuilding["datastart"]
endDate_chosenBuilding = Series_metaData_chosenBuilding["dataend"]
timeZone_chosenBuilding = Series_metaData_chosenBuilding["timezone"]

# Now let's go back to our temporalData dataframe and extract the whole column that corresponds to our example building
Series_temporalData_chosenBuilding  = DF_temporalData[chosenBuilding]
Series_temporalData_chosenBuilding.head()
# but if we would have a look on the head we can see that for the indexes there is +00:00  which indicates that time zone is UTC
# though we now that time zone of our example building  is  America/Los_Angeles
print timeZone_chosenBuilding
# So we will need to convert the timezone of it using tz_convert

Series_temporalData_chosenBuilding_timeZoneConverted =  Series_temporalData_chosenBuilding.tz_convert(timeZone_chosenBuilding)
# now if we would print the head of this Series :
Series_temporalData_chosenBuilding_timeZoneConverted.head()
# We can see that the time zone has changed and -8:00 shows that timezone of this dataset is 8 hours bheind UTC.
# but still we can observe that the for the first rows the consumption value is NaN which is due to the fact that in 2010 the measurments were not started
# so we need to limit the rows to the ones between the start and the end date corresponding to this dataset. To do so, we can use the truncate method which simply
# truncates the rows before a certain date and after a certain date.
measuredData_chosenBuilding =  Series_temporalData_chosenBuilding_timeZoneConverted.truncate(before=startDate_chosenBuilding,after=endDate_chosenBuilding)
measuredData_chosenBuilding.head(24)
# We can see that the values are not NaN anymore.
print "Measurement of this building started on "+startDate_chosenBuilding.strftime('%Y-%m-%d') + " and ended on "+endDate_chosenBuilding
# Clearly this procedure can be similarly repeated for all buildings so let's write a function that performs the same procedure !

def extract_building_data(DF_temporalData, DF_metaData, chosenBuilding, printBuildingInformation = False):
    DF_metaData_transposed = DF_metaData.T
    Series_metaData_chosenBuilding = DF_metaData_transposed[chosenBuilding]
    startDate_chosenBuilding = Series_metaData_chosenBuilding["datastart"]
    endDate_chosenBuilding = Series_metaData_chosenBuilding["dataend"]
    timeZone_chosenBuilding = Series_metaData_chosenBuilding["timezone"]
    Series_temporalData_chosenBuilding  = DF_temporalData[chosenBuilding]
    Series_temporalData_chosenBuilding_timeZoneConverted =  Series_temporalData_chosenBuilding.tz_convert(timeZone_chosenBuilding)
    measuredData_chosenBuilding = Series_temporalData_chosenBuilding_timeZoneConverted.truncate(before=startDate_chosenBuilding,after=endDate_chosenBuilding)
    if printBuildingInformation==True:
        print "The name of chosen building is "+ chosenBuilding+  "  it's timezone is "+timeZone_chosenBuilding 
        print "Measurement of this building started on "+startDate_chosenBuilding.strftime('%Y-%m-%d') + " and ended on "+endDate_chosenBuilding.strftime('%Y-%m-%d')
    return measuredData_chosenBuilding

# So let's use this created function to find the data of our example building    
# the function strftime only converts the pandas timestamp to a formated  string ( whic is given to be year month day in our case '%Y-%m-%d'
    
ExtractedData_chosenBuilding  = extract_building_data(DF_temporalData,DF_metaData, "Office_Ellie", printBuildingInformation=True)
ExtractedData_chosenBuilding.head(24)


# Let's see an example of extracting the data corresponding to an individual building:


# Now let's extract the data for the period of '2012-12-15' till '2012-12-30'
period_start = '2012-12-15'
period_end = '2012-12-30'
DF_Office_Ellie_period = DF_Office_Ellie.truncate(before=period_start,after=period_end)
DF_Office_Ellie_period.head()    


# Let's visualise it
fig=  plt.figure()
ax = fig.add_axes()

ax = DF_Office_Ellie_period.plot()
ax.set_xlabel("Time")
ax.set_ylabel("Consumption (kWh) of "+exampleBuilding)
ax.set_title("Consumption of "+exampleBuilding)
# Next, we create a dataSet similar to the original temp with the only difference that the consumption of each timestamp of each building
# is divided by its surface (sqm)
DF_temp_hourly = DF_temp.resample('H').sum() # this is simply resampling data to be hourly and it is summing up everything in between
DF_temp_normalized = DF_temp_hourly/DF_meta["sqm"] # it is dividing the column of each building by its corresponding sqm


# Now let's find the the normalized consumption of the example building

DF_Office_Ellie_normalized = get_individual_data(DF_temp_normalized,DF_meta,"Office_Ellie")
DF_Office_Ellie_normalized_period = DF_Office_Ellie_normalized.truncate(before=period_start,after=period_end)





# Obtaining the temporal features
DF_temp_normalized_daily =  DF_temp_normalized.resample("D").sum()
DF_temp_normalized_daily.head()

# Let's see the normalized daily consumption of our example building
DF_temp_normalized_daily_example = DF_temp_normalized_daily[exampleBuilding]
# Let's visualise it
plt.close("all")
fig=  plt.figure()
ax1 = fig.add_axes()

DF_temp_normalized_daily.plot(ax=ax1)
ax.set_xlabel("Time")
ax.set_ylabel("Consumption (kWh) of "+exampleBuilding)
ax.set_title("Consumption of "+exampleBuilding)


DF_temp_normalized_stats_transposed= DF_temp_normalized_daily.describe().T 
DF_temp_normalized_stats_transposed


DF_temp_normalized_mainStats_transposed = DF_temp_normalized_stats_transposed[["mean","std","min","max"]]

NewNames_for_staticsColumns = ["BGNormalizedCons_mean","BGNormalizedCons_std","BGNormalizedCons_max","BGNormalizedCons_min"]

DF_temp_normalized_mainStats_transposed.columns = NewNames_for_staticsColumns

DF_temp_normalized_mainStats_transposed.index.name = "building_name"
DF_temp_normalized_mainStats_transposed.columns.name = "feature_name"

allBuildings_daily_temporal_features = DF_temp_normalized_mainStats_transposed
