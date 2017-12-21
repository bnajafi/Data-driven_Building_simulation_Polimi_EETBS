
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

DF_temperature=DF_temperature.shift(-6)
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
DF_joined.plot()
# what to do with Nans 
DF_joined_cleaned = DF_joined.dropna() #it will remove all Nans !!!!! 

DF_joined_cleaned_chosenDates = DF_joined_cleaned['2014-08-01':'2014-08-02'] #it will remove all Nans !!!!! 

DF_joined_cleaned_chosenDates_max  = DF_joined_cleaned_chosenDates.max()
DF_joined_cleaned_chosenDates_min  = DF_joined_cleaned_chosenDates.min()
DF_joined_cleaned_chosenDates_normalized = (DF_joined_cleaned_chosenDates-DF_joined_cleaned_chosenDates_min)/(DF_joined_cleaned_chosenDates_max-DF_joined_cleaned_chosenDates_min)
plt.figure()
DF_joined_cleaned_chosenDates_normalized.plot()

# If you would like to close all figures !!!!
plt.close("all")
"""
This is what we might do if we do not know that PAndas does vector oepration on all columns !!!
temp_max=DF_joined_cleaned_chosenDates["temperature"].max()
temp_min = DF_joined_cleaned_chosenDates["temperature"].min()

DF_joined_cleaned_chosenDates["temperature normalized"] = (DF_joined_cleaned_chosenDates["temperature"]-temp_min)/(temp_max-temp_min)
plt.figure()
DF_joined_cleaned_chosenDates["temperature normalized"].plot()

gen_max=DF_joined_cleaned_chosenDates["gen"].max()
gen_min = DF_joined_cleaned_chosenDates["gen"].min()

DF_joined_cleaned_chosenDates["gen normalized"] = (DF_joined_cleaned_chosenDates["gen"]-gen_min)/(gen_max-gen_min)
DF_joined_cleaned_chosenDates["gen normalized"].plot()

AC_max=DF_joined_cleaned_chosenDates["air conditioner_5545"].max()
AC_min = DF_joined_cleaned_chosenDates["air conditioner_5545"].min()

DF_joined_cleaned_chosenDates["AC normalized"] = (DF_joined_cleaned_chosenDates["air conditioner_5545"]-AC_min)/(AC_max-AC_min)
DF_joined_cleaned_chosenDates["AC normalized"].plot(label=True)
plt.legend()
"""

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
DF_joined_cleaned_chosenDates["air conditioner_5545"].plot(ax=ax1,color="b",legend=True)
DF_joined_cleaned_chosenDates["temperature"].plot(ax=ax2,color="r",legend=True)
DF_joined_cleaned_chosenDates["gen"].plot(ax=ax3,color="g",legend=True)
DF_joined_cleaned_chosenDates.head()

DF_FinalDataSet = DF_joined_cleaned.copy()
DF_joined_cleaned_chosenDates_lagged = DF_joined_cleaned_chosenDates.copy()
DF_joined_cleaned_chosenDates_lagged["temperature -1hr"]=DF_joined_cleaned_chosenDates_lagged["temperature"].shift(1)
DF_joined_cleaned_chosenDates_lagged["temperature -2hr"]=DF_joined_cleaned_chosenDates_lagged["temperature"].shift(2)
DF_joined_cleaned_chosenDates_lagged["temperature -3hr"]=DF_joined_cleaned_chosenDates_lagged["temperature"].shift(3)
DF_joined_cleaned_chosenDates_lagged["temperature -4hr"]=DF_joined_cleaned_chosenDates_lagged["temperature"].shift(4)
DF_joined_cleaned_chosenDates_lagged["temperature -5hr"]=DF_joined_cleaned_chosenDates_lagged["temperature"].shift(5)
 
 #let's do the same for irradiation
DF_joined_cleaned_chosenDates_lagged["gen -1hr"]=DF_joined_cleaned_chosenDates_lagged["gen"].shift(1)
DF_joined_cleaned_chosenDates_lagged["gen -2hr"]=DF_joined_cleaned_chosenDates_lagged["gen"].shift(2)
DF_joined_cleaned_chosenDates_lagged["gen -3hr"]=DF_joined_cleaned_chosenDates_lagged["gen"].shift(3)
DF_joined_cleaned_chosenDates_lagged["gen -4hr"]=DF_joined_cleaned_chosenDates_lagged["gen"].shift(4)
DF_joined_cleaned_chosenDates_lagged["gen -5hr"]=DF_joined_cleaned_chosenDates_lagged["gen"].shift(5)
DF_joined_cleaned_chosenDates_lagged["gen -6hr"]=DF_joined_cleaned_chosenDates_lagged["gen"].shift(6)
DF_joined_cleaned_chosenDates_lagged.head()

DF_joined_cleaned_chosenDates_lagged.corr() # computes correlation beteeen the features


DF_joined_cleaned_chosenDates_lagged.head()
# since it is not possible to continue making new DF like this !!!!!
#DF_joined_cleaned_chosenDates_lagged_cleanedAgain= DF_joined_cleaned_chosenDates_lagged.dropna()
 
DF_joined_cleaned_chosenDates_lagged.dropna()# the original one does not change
DF_joined_cleaned_chosenDates_lagged.dropna(inplace=True)

DF_joined_cleaned_chosenDates_lagged.corr()

# Let's build our final set with our selected features !!!

DF_FinalDataSet
DF_FinalDataSet["temperature -1hr"]=DF_FinalDataSet["temperature"].shift(1)
DF_FinalDataSet["temperature -2hr"]=DF_FinalDataSet["temperature"].shift(2)
DF_FinalDataSet["temperature -3hr"]=DF_FinalDataSet["temperature"].shift(3)
DF_FinalDataSet["temperature -4hr"]=DF_FinalDataSet["temperature"].shift(4)
DF_FinalDataSet["temperature -5hr"]=DF_FinalDataSet["temperature"].shift(5)
DF_FinalDataSet["gen -5hr"]=DF_FinalDataSet["gen"].shift(5)
DF_FinalDataSet["gen -6hr"]=DF_FinalDataSet["gen"].shift(6)
DF_FinalDataSet.dropna(inplace=True)

#DF_FinalDataSet["AC cons-1hr"]=DF_FinalDataSet["air conditioner_5545"].shift(1)



def lag_column(df,column_name,lag_period=1):
    for i in range(1,lag_period+1,1):
        new_column_name = column_name+"-"+str(i)+"hr"
        df[new_column_name]=df[column_name].shift(i)
    return df
 
DF_FinalDataSet=lag_column(DF_FinalDataSet,"air conditioner_5545",24)
DF_FinalDataSet.dropna(inplace=True)

# What are we missing ?!?!

# So first let's see what we have in our index !
DF_FinalDataSet.index
DF_FinalDataSet.index.hour
result = DF_FinalDataSet.index.dayofweek
result.min()
result.max()
DF_FinalDataSet.index.month
DF_FinalDataSet.index.week

# SO first let's add these ones !!
DF_FinalDataSet['hour']=DF_FinalDataSet.index.hour
DF_FinalDataSet['day_of_week']=DF_FinalDataSet.index.dayofweek
DF_FinalDataSet['month']=DF_FinalDataSet.index.month
DF_FinalDataSet['week_of_year']=DF_FinalDataSet.index.week

def weekendDetector(day):
    weekendLabel = 0
    if (day == 5 or day==6):
        weekendLabel = 1
    else:
        weekendLabel = 0
    return weekendLabel
    
def dayDetector(hour):
    dayLabel = 1
    if (hour<20 and hour>9):
        dayLabel = 1
    else: 
        dayLabel = 0
    return dayLabel
    
simpleVectorOFDays = [0,1,2,3,4,5,6]
# weekendorNotVector 
# A very pythonic solution !!!
# List comprehension !!!

x=[1,2,3]
y = [item**2 for item in x]
weekendorNotVector = [weekendDetector(thisDay) for thisDay in simpleVectorOFDays]
hoursOFdayVector = range(0,24,1)
dayOrNotVector = [dayDetector(ThisHour) for ThisHour in  hoursOFdayVector]
        
DF_FinalDataSet['weekend'] = [weekendDetector(thisDay) for thisDay in DF_FinalDataSet.index.dayofweek ]
DF_FinalDataSet['day_night'] = [dayDetector(thisHour) for thisHour in DF_FinalDataSet.index.hour ]
DF_FinalDataSet.head()


