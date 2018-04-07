import getpass
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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


# we apply the same formatting to the consumption data
ConsumptionData_chosenBuilding_dailySum = ExtractedData_chosenBuilding.resample("D").sum()
# Next we remove the timezone
ConsumptionData_chosenBuilding_dailySum=ConsumptionData_chosenBuilding_dailySum.tz_localize(None)
# Nextt we convert this Serie into a dataframe
DF_ConsumptionData_chosenBuilding_dailySum= pd.DataFrame(ConsumptionData_chosenBuilding_dailySum)
# we should now merge these two dataframes based on the index(dates) of the building consumption DF
# so as merge arguments we should choose left_index=True, right_index=True so that it would take into account 
#both indices then to choose the index tobe retained we would choose how ="left" which would only keep the indices included in the left DF which is building consumption
tempeCons_combined = pd.merge(DF_ConsumptionData_chosenBuilding_dailySum, DF_outdoorTemp_dailyAverage, right_index=True, left_index=True, how='left')

# Next we groupby the combined dataframe by th eyear and the month
tempCons_combined_grouppedbyMonth = tempeCons_combined.groupby([lambda date: date.year, lambda date: date.month])

# now let's see the groups that we have:
print tempCons_combined_grouppedbyMonth.groups
#  we observe that we hav 12 groups each corresponding to a month in 2012.

# Next we find the spearman r correlation of the grouppedby dataframe
# the syntax of spearmann r is : R_value, P_value= stats.spearmanr(x,y) , so we can choose index [0] of its results to receive the pearsonr value
# for the x we can use .iloc[:,0] which is the column of consumptions, and .iloc[:,1] which is the column of temperatures
pearsonr_tempCons_combined_grouppedbyMonth = tempCons_combined_grouppedbyMonth.apply(lambda x: stats.spearmanr(x.iloc[:,0] , x.iloc[:,1])[0]) # remmeber to import the stats from scipy package
