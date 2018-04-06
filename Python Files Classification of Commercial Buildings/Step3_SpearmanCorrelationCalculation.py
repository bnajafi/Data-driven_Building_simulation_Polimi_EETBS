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
    

# In this step, we would like to find the spearman correaltion between the weather condition and the consumption of the building 

# so the first task is to extract the weather condiiton corresponding to the  city and timestamp corresponding to each building, the external 
# data folder includes the weather data and the name of the weather data corresponding to each building is given in the meta data file.
    
# Let's first choose our example building
chosenBuilding = "Office_Ellie"
# we can first extract the building's consumption data
ExtractedData_chosenBuilding  = extract_building_data(DF_temporalData,DF_metaData,chosenBuilding )
ExtractedData_chosenBuilding.head(24)

# let's we find the name of the  corresponding weather file:
# the column of the chosen buildign in  the meta data file can be found in
metaData_chosenBuildingColumn = DF_metaData.T[chosenBuilding]
# so the weather name and timezone can be found:
weatherfilename = metaData_chosenBuildingColumn["newweatherfilename"]
chosenBuilding_timezone =  metaData_chosenBuildingColumn["timezone"]

# we can next create the path to the weather data by adding the corresponding directory and apply the building's time zone
weatherfile_path = weatherDataSet_directory+weatherfilename
weather = pd.read_csv(weatherfile_path,index_col='timestamp', parse_dates=True, na_values='-9999')
weather = weather.tz_localize(chosenBuilding_timezone,ambiguous = 'infer')
weather.head()

# Next's find the average daily temperatures:
outdoor_temp_dailyAverage =weather["TemperatureC"].resample("D").mean()
# we can next remove the timezone:

outdoor_temp_dailyAverage = outdoor_temp_dailyAverage.tz_localize(None)

# Let's convert it to a dataFrame
DF_outdoorTemp_dailyAverage = pd.DataFrame(outdoor_temp_dailyAverage);