import getpass
import os
import pandas as pd
import matplotlib.pyplot as plt

userName = getpass.getuser() # this line checks for the username and chooses the corresponding directory accordingly
if userName=="behzad":
    Genomic_DataSet_directory = "C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/data/raw/"
    weatherDataSet_directory = "C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/data/external/weather/"
dataSetFileName = "temp_open_utc.csv"
metaDataFileName = "meta_open.csv"
path_temporalData= os.path.join(Genomic_DataSet_directory,dataSetFileName)
path_metaData =  os.path.join(Genomic_DataSet_directory,metaDataFileName)
DF_temporalData = pd.read_csv(path_temporalData,index_col="timestamp", parse_dates=True).tz_localize("utc")
DF_metaData = pd.read_csv(path_metaData,index_col = "uid", parse_dates=["datastart","dataend"], dayfirst=True) 

#Let's paste the function we created in the previous step

def extract_building_data(DF_temporalData, DF_metaData, chosenBuilding):
    DF_metaData_transposed = DF_metaData.T
    Series_metaData_chosenBuilding = DF_metaData_transposed[chosenBuilding]
    startDate_chosenBuilding = Series_metaData_chosenBuilding["datastart"]
    endDate_chosenBuilding = Series_metaData_chosenBuilding["dataend"]
    timeZone_chosenBuilding = Series_metaData_chosenBuilding["timezone"]
    Series_temporalData_chosenBuilding  = DF_temporalData[chosenBuilding]
    Series_temporalData_chosenBuilding_timeZoneConverted =  Series_temporalData_chosenBuilding.tz_convert(timeZone_chosenBuilding)
    measuredData_chosenBuilding = Series_temporalData_chosenBuilding_timeZoneConverted.truncate(before=startDate_chosenBuilding,after=endDate_chosenBuilding)
    return measuredData_chosenBuilding

ExtractedData_chosenBuilding  = extract_building_data(DF_temporalData,DF_metaData, "Office_Ellie")
ExtractedData_chosenBuilding.head(24)

# Clearly normalized values and provide more useful means for comparing different buildings.
#In the field of buildings, one useful normalized value is the consumption of the building divided by its surface (kwh/m^2)

# to create such a series let's first calculate the surface of this buildings 
chosenBuilding = "Office_Ellie"
DF_metaData_transposed = DF_metaData.T
Series_metaData_chosenBuilding = DF_metaData_transposed[chosenBuilding] # the transposed Series has 18 rows one of which is sqm or squa
startDate_chosenBuilding = Series_metaData_chosenBuilding["datastart"]
endDate_chosenBuilding = Series_metaData_chosenBuilding["dataend"]
timeZone_chosenBuilding = Series_metaData_chosenBuilding["timezone"]
surface_chosenBuilding = Series_metaData_chosenBuilding["sqm"] # I have just addded this
Series_temporalData_chosenBuilding  = DF_temporalData[chosenBuilding]
Series_temporalData_chosenBuilding_timeZoneConverted =  Series_temporalData_chosenBuilding.tz_convert(timeZone_chosenBuilding)
measuredData_chosenBuilding = Series_temporalData_chosenBuilding_timeZoneConverted.truncate(before=startDate_chosenBuilding,after=endDate_chosenBuilding)
NormalizedData_chosenBuilding = measuredData_chosenBuilding/surface_chosenBuilding

# So we can similarly write a function to the same for any buildingg
def extract_building_normalizedData(DF_temporalData, DF_metaData, chosenBuilding):
    DF_metaData_transposed = DF_metaData.T
    Series_metaData_chosenBuilding = DF_metaData_transposed[chosenBuilding]
    startDate_chosenBuilding = Series_metaData_chosenBuilding["datastart"]
    endDate_chosenBuilding = Series_metaData_chosenBuilding["dataend"]
    timeZone_chosenBuilding = Series_metaData_chosenBuilding["timezone"]
    surface_chosenBuilding = Series_metaData_chosenBuilding["sqm"] 
    Series_temporalData_chosenBuilding  = DF_temporalData[chosenBuilding]
    Series_temporalData_chosenBuilding_timeZoneConverted =  Series_temporalData_chosenBuilding.tz_convert(timeZone_chosenBuilding)
    measuredData_chosenBuilding = Series_temporalData_chosenBuilding_timeZoneConverted.truncate(before=startDate_chosenBuilding,after=endDate_chosenBuilding)
    NormalizedData_chosenBuilding = measuredData_chosenBuilding/surface_chosenBuilding
    return NormalizedData_chosenBuilding


# Now let's find use this function to find the normalized consumption:
chosenBuilding="Office_Ellie"
normalizedData_chosenBuilding = extract_building_normalizedData(DF_temporalData, DF_metaData, chosenBuilding)
normalizedData_chosenBuilding.head(24)


#Though there is a solution to directly find the normalized consumption of all buildings in a vectorized way

DF_temporalData_normalized = DF_temporalData.resample('H').sum()/DF_metaData["sqm"] # resampel hourly just makes sure all indices are hourly  
# dividing by meta DF_metaData["sqm"] divides every column's consumption by its corresponding sqm

# Now let's resample thw whole dataset to daily 
DF_temporalData_daily_normalized = DF_temporalData_normalized.resample("D").sum()

# Next, let's apply describe on it to find the statistical properties of each building's daily normalized consumption
DF_temporalData_daily_normalized_stats= DF_temporalData_daily_normalized.describe()
# Let's next transpose it so that each building woulbe a row!
DF_temporalData_daily_normalized_stats_trasposed = DF_temporalData_daily_normalized_stats.T

# Let's just keep the std , mean , min and max
DF_temporalData_dailyNormalizedStats = DF_temporalData_daily_normalized_stats_trasposed[["mean","std","min","max"]]
# let's rename the columns
DF_temporalData_dailyNormalizedStats.columns = ["mean_normalizedCons","std_normalizedCons","min_normalizedCons","max_normalizedCons"]
DF_temporalData_dailyNormalizedStats.head()

