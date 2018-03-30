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
path_temp= os.path.join(Genomic_DataSet_directory,dataSetFileName)
path_metaData =  os.path.join(Genomic_DataSet_directory,metaDataFileName)

""" Note on creating the path to the file! 
Why do we use such a complicated syntax (os.path.join), we could simply use the following:
path_temp = Genomic_DataSet_directory+dataSetFileName 
path_metaData = Genomic_DataSet_directory
The problem is that this will only work on Windows and not on a Mac, os.path.join handles this problem 
"""
# now let's use Pandas read_csv the import the temperalData and MetaData datasets as Pandas dataframes
DF_temporalData = pd.read_csv(path_temp,index_col="timestamp", parse_dates=True).tz_localize("utc")
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
exampleBuilding = "Office_Ellie"
Series_metaData_exampleBuilding = DF_metaData_transposed[exampleBuilding]
# so we can see its date Start Date End , time zone, surface , industry, subindustry etc. and the corresponding weather file
# Now let's see how we can use the metadata information to extract the related data of this building from the temporalData dataframe

DF_meta_transposed = DF_metaData.T # here we just have a transposed format of the meta dataset
DF_meta_transposed.head()
DF_meta_trasposed_exampleBuilding = DF_meta_transposed[exampleBuilding]


exampleBuilding_startDate = DF_meta_trasposed_exampleBuilding.datastart
exampleBuilding_EndDate = DF_meta_trasposed_exampleBuilding.dataend
exampleBuilding_timeZone = DF_meta_trasposed_exampleBuilding.timezone

DF_temp_exampleBuilding  = DF_temp[exampleBuilding]
DF_temp_exampleBuilding.head()
DF_temp_exampleBuilding_timeZoned =DF_temp_exampleBuilding.tz_convert(exampleBuilding_timeZone)
DF_temp_exampleBuilding_inMeasuredDuration = DF_temp_exampleBuilding_timeZoned.truncate(before=exampleBuilding_startDate,after=exampleBuilding_EndDate)
#Next function read the data of an indvidual building
DF_temp_exampleBuilding_inMeasuredDuration.head()

def get_individual_data(temp, meta, building):
    timezone = meta.T[building].timezone # it is simply first transposing the meta dataset so that the buildings would be columns and propertes would be indexes, next it find the time zone based on that
    start = meta.T[building].datastart # using the transposed meta data it is finding the measurment start date for a building
    end = meta.T[building].dataend# using the transposed meta data it is finding the measurment end date for a building
    # Next we convert the data of each building into its timezone (tz_convert
    #and then we truncate this Series between the start and end dates of its measurment
    return pd.DataFrame(temp[building].tz_convert(timezone).truncate(before=start,after=end)) 

# Let's see an example of extracting the data corresponding to an individual building:

exampleBuilding = "Office_Ellie"

DF_Office_Ellie = get_individual_data(DF_temp,DF_meta,exampleBuilding)
DF_Office_Ellie.head()

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
