
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DataFolderPath = "C:/Users/behzad/Dropbox/_2_Teaching Activities/_0_EETBS- On-going/git_fork_clone/Data-driven_Building_simulation_Polimi_EETBS/Data"
ConsumptionFileName = "consumption_5545.csv"
ConsumptionFilePath = DataFolderPath+"/"+ConsumptionFileName 

DF_consumption = pd.read_csv(ConsumptionFilePath,sep = ",",index_col=0) 
previousIndex= DF_consumption.index
NewparsedIndex = pd.to_datetime(previousIndex)
DF_consumption.index= NewparsedIndex
DF_consumption.head(24)
DF_JulyfirstTillthird = DF_consumption["2014-07-01 00:00:00":"2014-07-03 23:00:00"]
DF_JulyfirstTillthird.head(5)
DF_JulyfirstTillthird.describe()
# In[48]:

plt.figure()
plt.plot(DF_JulyfirstTillthird)
DF_JulyfirstTillthird.plot()
plt.xlabel('Time')
plt.ylabel('AC Power [W]')
plt.show()


# Now let's import some weather data!
weatherSourceFileName = "Austin_weather_2014.csv"
weatherSourceFilePath = DataFolderPath+"/"+weatherSourceFileName 
DF_weatherSource = pd.read_csv(weatherSourceFilePath,sep = ";",index_col=0)
DF_weatherSource.index

previousIndex_weatherSource= DF_weatherSource.index
NewparsedIndex_weatherSource = pd.to_datetime(previousIndex_weatherSource)
DF_weatherSource.index= NewparsedIndex_weatherSource

#  we usually do this
series_temperature = DF_weatherSource['temperature']

# Nut now I would prefer to have it as a dataframe with just one column, we will then see why !!
DF_temperature = DF_weatherSource[['temperature']]

# let's do the same for irradiation!!!
IrradianceSourceFileName = "irradiance_2014_gen.csv"
IrradianceSourceFilePath =  DataFolderPath+"/"+IrradianceSourceFileName 
DF_irradianceSource = pd.read_csv(IrradianceSourceFilePath, sep = ";",index_col= 1)
DF_irradianceSource.head(5)

previousIndex_irradianceSource= DF_irradianceSource.index
NewparsedIndex_irradianceSource = pd.to_datetime(previousIndex_irradianceSource)
DF_irradianceSource.index= NewparsedIndex_irradianceSource

# IF I want take just the column "gen " as a dataframe with a single column !
DF_irradiance = DF_irradianceSource[["gen"]] # to take it as a DF
DF_irradiance[DF_irradianceSource["gen"] < 0] = 0

DF_joined = DF_consumption.join([DF_temperature,DF_irradiance])
DF_joined.head(24)

# what to do with Nans 
DF_joined.dropna() #it will remove all Nans !!!!! 
