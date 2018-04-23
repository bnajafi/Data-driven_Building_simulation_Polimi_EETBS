# -*- coding: utf-8 -*-
# in order to utilize this file, you should install version 0.3.2 iof eemeter using the following command
#pip install eemeter==0.3.2
# if you are using canopy, you should do so in the canopy command prompt. found in tools menu
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import getpass
import seaborn as sns
import os 


import eemeter
from datetime import timedelta
#import eemeter
# models to build efficiency meters
from eemeter.importers import import_pandas
from eemeter.weather import GSODWeatherSource
from eemeter.weather import TMY3WeatherSource
#from eemeter.consumption import DatetimePeriod
from eemeter.consumption import ConsumptionData
from eemeter.weather import WeatherSourceBase
#from eemeter.consumption import ConsumptionHistory
from eemeter.models import AverageDailyTemperatureSensitivityModel
from eemeter.meter import TemperatureSensitivityParameterOptimizationMeter
from eemeter.meter import AnnualizedUsageMeter


# First we simply paste the procedure we had utilized to extract the building and weather data in the previous step
## pasted part start

userName = getpass.getuser() # this line checks for the username and chooses the corresponding directory accordingly
if userName=="behzad":
    Genomic_DataSet_directory = "C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/data/raw/"
    weatherDataSet_directory = "C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/data/external/weather/"
dataSetFileName = "temp_open_utc_complete.csv"
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
chosenBuilding = "PrimClass_Everett"
# we can first extract the building's consumption data
ExtractedData_chosenBuilding  = extract_building_data(DF_temporalData,DF_metaData,chosenBuilding )
ExtractedData_chosenBuilding.head(24)

# let's we find the name of the  corresponding weather file:
# the column of the chosen buildign in  the meta data file can be found in
metaData_chosenBuildingColumn = DF_metaData.T[chosenBuilding]
# so the weather name and timezone can be found:
weatherfilename = metaData_chosenBuildingColumn["newweatherfilename"]
chosenBuilding_timezone =  metaData_chosenBuildingColumn["timezone"]

# without function
testbuilding = "PrimClass_Everett"
test_timezone = DF_metaData.T[testbuilding].timezone
test_start = DF_metaData.T[testbuilding].datastart
test_end = DF_metaData.T[testbuilding].dataend
test_building_data = pd.DataFrame(DF_temporalData[testbuilding].tz_convert(test_timezone).truncate(before=test_start,after=test_end))

test_building_data.tail()




# we can next create the path to the weather data by adding the corresponding directory and apply the building's time zone
weatherfile_path = weatherDataSet_directory+weatherfilename
weather = pd.read_csv(weatherfile_path,index_col='timestamp', parse_dates=True, na_values='-9999')
weather = weather.tz_localize(chosenBuilding_timezone,ambiguous = 'infer')
weather.head()

# pasted part end

# Next we find the outside temeperature

outdoor_temp = pd.DataFrame(weather[[col for col in weather.columns if 'Temperature' in col]]).resample("H").mean()

DF_temp = pd.DataFrame(weather[[col for col in weather.columns if 'Temperature' in col]])
DF_temp.head()
DF_temp.describe()
DF_temp_dailyaverage = DF_temp.resample("D").mean()
DF_temp_dailyaverage = DF_temp_dailyaverage.tz_localize(None)

data = {}
for i, v in zip(DF_temp_dailyaverage.index.strftime('%Y%m%d'),list(DF_temp_dailyaverage.TemperatureC)):
    data[i] = v
    
    
# formatted weather data
class CustomDailyWeatherSource(WeatherSourceBase):
        def __init__(self, weather):
            data = {}
            #df = pd.read_csv("Weather/"+weatherfilename, index_col='timestamp', parse_dates=True, na_values='-9999')
            df_DB = pd.DataFrame(weather[[col for col in weather.columns if 'Temperature' in col]])
            #df_DB = (df_DB*(9/5))+32 #convert to F
            df_DB = df_DB[df_DB<120].resample("D").mean()
            df_DB = df_DB.tz_localize(None)
            for i, v in zip(df_DB.index.strftime('%Y%m%d'),list(df_DB.TemperatureC)):
                data[i] = v
            self.data = data
            self._internal_unit = "degC"

        def internal_unit_datetime_average_temperature(self,dt):
            return self.data.get(dt.strftime("%Y%m%d"),np.nan)  
            
class CustomDailyWeatherSource2(WeatherSourceBase):
        def __init__(self, weather):
            data = {}
            #df = pd.read_csv("Weather/"+weatherfilename, index_col='timestamp', parse_dates=True, na_values='-9999')
            df_DB = pd.DataFrame(weather[[col for col in weather.columns if 'Temperature' in col]])
            #df_DB = (df_DB*(9/5))+32 #convert to F
            df_DB = df_DB[df_DB<120].resample("D").mean()
            df_DB = df_DB.tz_localize(None)
            df_DB.columns=["TempC"]
            TempC_Series=df_DB["TempC"]    
            self.tempC = TempC_Series
            self._internal_unit = "degC"

formattedWeatherData2 =  CustomDailyWeatherSource2(weather)
formattedWeatherData =  CustomDailyWeatherSource(weather)

#df_t = pd.Series(formattedWeatherData.data)
df_t = pd.Series(formattedWeatherData.data)

df_t.index = df_t.index.to_datetime()
df_t = pd.DataFrame(df_t)
df_t.columns = ["TempC"]
    
    
model = AverageDailyTemperatureSensitivityModel(
            heating=True,
            cooling=True,
            initial_params={
                "base_daily_consumption": 0,
                "heating_slope": 0,
                "heating_balance_temperature": 10,
                "cooling_slope": 0,
                "cooling_balance_temperature": 20,
            },
            param_bounds={
                "base_daily_consumption": [0,1000000],
                "heating_slope": [0,1000],
                "heating_balance_temperature": [5,16],
                "cooling_slope": [0,1000],
                "cooling_balance_temperature": [13,24],
            })
            
            
# This weather source base !!
#ThisWeatherData= WeatherSourceBase()            

ExtractedData_chosenBuilding = pd.DataFrame(ExtractedData_chosenBuilding)
to_load = ExtractedData_chosenBuilding.resample('D').sum().tz_localize(None)
to_load.columns = ['Consumption']
to_load['StartDateTime'] = to_load.index.format('%Y-%m-%d %H:%M:%S.%f')[1:]
end=to_load.index+timedelta(days=1)
to_load['EndDateTime'] = end.format('%Y-%m-%d %H:%M:%S.%f')[1:]
to_load['UnitofMeasure'] = 'kWh'
to_load['FuelType'] = 'electricity'
to_load['ReadingType'] = 'actual'


to_load = to_load.reset_index(drop=True)

to_load.head()

consumptions = import_pandas(to_load)

param_optimization_meter = TemperatureSensitivityParameterOptimizationMeter("degC", model)


annualized_usage_meter = AnnualizedUsageMeter("degC", model)

params = param_optimization_meter.evaluate_raw(consumption_data=consumptions,  weather_source=formattedWeatherData2,energy_unit_str="kWh")["temp_sensitivity_params"]
params_list = params.to_list()
 

names1 = ['Baseload', 'HeatBalPtF', 'HeatSlope','CoolBalPtF','CoolSlope']
df_par = pd.DataFrame(params.to_list())
df_par['Parameter'] = index=names1
df_par.columns = ['Value', 'Parameter']
df_par = df_par.set_index('Parameter')


df_m = ExtractedData_chosenBuilding.resample('D').sum().tz_localize(None).join(df_t.resample('1D').mean(), how='inner')
df_m.tail()
df_m['model_Consumption'] = params_list[0] + params_list[2]*(np.where(df_m.TempC<params_list[1], params_list[1]-df_m.TempC, 0)) + params_list[4]*(np.where(df_m.TempC>params_list[3], df_m.TempC - params_list[3], 0))

df_m['modelbase'] = params_list[0]
df_m['modelheating'] = params_list[1]*(np.where(df_m.TempC<params_list[2], params_list[2]-df_m.TempC, np.nan))
df_m['modelcooling'] = params_list[3]*(np.where(df_m.TempC>params_list[4], df_m.TempC - params_list[4], np.nan))
df_par.loc['totalNMBE'] = 100*((df_m[testbuilding] - df_m.model_Consumption).sum()/((df_m[testbuilding].count()-1) * df_m[testbuilding].mean()))
df_par.loc['totalCVRMSE'] = 100*((((df_m[testbuilding] - df_m.model_Consumption)**2).sum()/(df_m[testbuilding].count()-1))**(0.5))/df_m[testbuilding].mean()

df_m.head()


df_m[[testbuilding,"model_Consumption"]].plot(figsize=(15,3))
plt.xlabel("Date")
plt.ylabel("Daily kWh Elec.")
plt.title("EEMeter Model Comparison")
plt.tight_layout()
#plt.savefig(os.path.join(repos_path,"reports/figures/eemeter/predictedvsactual_annual.png"));

sns.set_style('whitegrid')
plt.figure(figsize=(15,5))
plt.scatter(df_m.TempC, df_m[testbuilding], color='b')
plt.scatter(df_m.TempC, df_m.model_Consumption, color='g')
plt.ylabel("Electrical Consumption [kWh]")
plt.xlabel("Outdoor Air Dry Bulb Temp [Deg C]")
#plt.savefig(os.path.join(repos_path,"reports/figures/eemeter/changepointscatter_example.png"))
plt.show()

df_par



def runeemetermodel(meta, temp, building, model):
    
    timezone = meta.T[building].timezone
    start = meta.T[building].datastart
    end = meta.T[building].dataend
    building_data = pd.DataFrame(temp[building].tz_convert(timezone).truncate(before=start,after=end))
    weatherDataSet_directory = "C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/data/external/weather/"
    weatherfilename = meta.T[building].newweatherfilename
    weatherfile_path = weatherDataSet_directory+weatherfilename
    weather = pd.read_csv(weatherfile_path,index_col='timestamp', parse_dates=True, na_values='-9999')
    weather = weather.tz_localize(timezone, ambiguous = 'infer')
    
    weatherdata = CustomDailyWeatherSource2(weather)
   # df_t = pd.Series(weatherdata.data)
    df_t = pd.Series(weatherdata.tempC)
    df_t.index = df_t.index.to_datetime()
    df_t = pd.DataFrame(df_t)
    
    df_t.columns = ["TempC"]

    to_load = building_data.resample('D').sum().tz_localize(None)
    to_load.columns = ['Consumption']
    to_load['StartDateTime'] = to_load.index.format('%Y-%m-%d %H:%M:%S.%f')[1:]
    end=to_load.index+timedelta(days=1)
    to_load['EndDateTime'] = end.format('%Y-%m-%d %H:%M:%S.%f')[1:]
    to_load['UnitofMeasure'] = 'kWh'
    to_load['FuelType'] = 'electricity'
    to_load['ReadingType'] = 'actual'
    to_load = to_load.reset_index(drop=True)
    consumptions = import_pandas(to_load)
    
    param_optimization_meter = TemperatureSensitivityParameterOptimizationMeter("degC",model)
    annualized_usage_meter = AnnualizedUsageMeter("degC",model)
    params = param_optimization_meter.evaluate_raw(
                        consumption_data=consumptions,
                        weather_source=weatherdata,
                        energy_unit_str="kWh")["temp_sensitivity_params"]

    names1 = ['Baseload', 'HeatBalPtF', 'HeatSlope','CoolBalPtF','CoolSlope']
    df_par = pd.DataFrame(params.to_list())
    df_par['Parameter'] = index=names1
    df_par.columns = ['Value', 'Parameter']
    df_par = df_par.set_index('Parameter')
    
    df_m = building_data.resample('D').sum().tz_localize(None).join(df_t.resample('1D').mean(), how='inner')
    params_list = params.to_list()

    df_m['model_Consumption'] = params_list[0] + params_list[2]*(np.where(df_m.TempC<params_list[1], params_list[1]-df_m.TempC, 0)) + params_list[4]*(np.where(df_m.TempC>params_list[3], df_m.TempC - params_list[3], 0))
    df_m['modelbase'] = params_list[0]
    df_m['modelheating'] = params_list[2]*(np.where(df_m.TempC<params_list[1], params_list[1]-df_m.TempC, np.nan))
    df_m['modelcooling'] = params_list[4]*(np.where(df_m.TempC>params_list[3], df_m.TempC - params_list[3], np.nan))
    
    df_par.loc['totalNMBE'] = 100*((df_m[building] - df_m.model_Consumption).sum()/((df_m[building].count()-1) * df_m[building].mean()))
    df_par.loc['totalCVRMSE'] = 100*((((df_m[building] - df_m.model_Consumption)**2).sum()/(df_m[building].count()-1))**(0.5))/df_m[building].mean()
    
    return df_m, df_par
    
building = "PrimClass_Everett"
df_m, df_par = runeemetermodel(DF_metaData, DF_temporalData, building, model)
df_m.head()
df_par

sns.set_style('whitegrid')
plt.figure(figsize=(15,5))
plt.scatter(df_m.TempC, df_m[building], color='b')
plt.scatter(df_m.TempC, df_m.model_Consumption, color='g')
plt.ylabel("Electrical Consumption [kWh]")
plt.xlabel("Outdoor Air Dry Bulb Temp [Deg C]")
#plt.savefig(os.path.join(repos_path,"reports/figures/eemeter/changepointscatter_example.png"))
plt.show()



buildinglist = list(DF_metaData.index)

# for all buildings
temp_modelconsumption = pd.DataFrame()
temp_modelheating = pd.DataFrame()
temp_modelcooling = pd.DataFrame()
features_eemeter = pd.DataFrame()

overall_start_time = timeit.default_timer()
for building in buildinglist:
    start_time = timeit.default_timer()
    try:
        df_m, df_par = runeemetermodel(DF_metaData, DF_temporalData, building, model)
    except:
        print "Couldn't model: "+building
        
    temp_modelconsumption = pd.merge(temp_modelconsumption, pd.DataFrame({building:df_m["model_Consumption"]}), right_index=True, left_index=True, how='outer')
    temp_modelheating = pd.merge(temp_modelheating, pd.DataFrame({building:df_m["modelheating"]}), right_index=True, left_index=True, how='outer')
    temp_modelcooling = pd.merge(temp_modelcooling, pd.DataFrame({building:df_m["modelcooling"]}), right_index=True, left_index=True, how='outer')
    features_eemeter = pd.merge(features_eemeter, pd.DataFrame({building:df_par["Value"]}), right_index=True, left_index=True, how='outer')
    
    print "Calculated: "+building+" in "+str(timeit.default_timer() - start_time)+" seconds"

print "Calculated all building in "+str(timeit.default_timer() - overall_start_time)+" seconds"

temp_modelconsumption.info()
temp_modelheating.info()
temp_modelcooling.info()

features_eemeter.info()

folder_processedData =  r"C:\Users\behzad\Dropbox\3 Research Projects\2 Data for Building\BuildingDataGenomeProject\the-building-data-genome-project\ProcessesData"

temp_modelconsumption.to_csv(os.path.join(folder_processedData,"temp_eemeter_predictedtotal.csv"))
temp_modelheating.to_csv(os.path.join(folder_processedData,"temp_eemeter_predictedheating.csv"))
temp_modelcooling.to_csv(os.path.join(folder_processedData,"temp_eemeter_predictedcooling.csv"))
features_eemeter.to_csv(os.path.join(folder_processedData,"features_eemeter.csv"))


temp_modelheating = pd.read_csv(os.path.join(folder_processedData,"temp_eemeter_predictedheating.csv"), index_col='timestamp')
temp_modelcooling = pd.read_csv(os.path.join(folder_processedData,"temp_eemeter_predictedcooling.csv"), index_col='timestamp')

temp_modelheating/DF_metaData.sqm
temp_modelheating_normalized= temp_modelheating/DF_metaData.sqm
temp_modelcooling_normalized = temp_modelcooling/DF_metaData.sqm


temp_modelheating_normalized_stats= temp_modelheating_normalized.describe().T
temp_modelcooling_normalized_stats = temp_modelcooling_normalized.describe().T

temp_modelheating_normalized_stats= temp_modelheating_normalized_stats [['mean','std','min','max']]
temp_modelcooling_normalized_stats= temp_modelcooling_normalized_stats[['mean','std','min','max']]

temp_modelheating_normalized_stats.columns = ['BG_eemeter_heating_mean','BG_eemeter_heating_std','BG_eemeter_heating_min','BG_eemeter_heating_max']
temp_modelcooling_normalized_stats.columns = ['BG_eemeter_cooling_mean','BG_eemeter_cooling_std','BG_eemeter_cooling_min','BG_eemeter_cooling_max']

features_eemeter_imported = pd.read_csv(os.path.join(folder_processedData,"features_eemeter.csv"), index_col=0)
features_eemeter2 = pd.concat([features_eemeter_imported.T,temp_modelcooling_normalized_stats,temp_modelcooling_normalized_stats],axis=1)