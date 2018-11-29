# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
# importing the external files
ExternalFilesFolder =  r"C:\Users\behzad\Dropbox\2 Teaching Activities\0 EETBS 2018\forked_repos\python4ScientificComputing_Numpy_Pandas_MATPLotLIB\ExternalFiles"
ConsumptionFileName= "consumption_5545.csv"
TemperatureFileName= "Austin_weather_2014.csv"
IrradianceFileName= "irradiance_2014_gen.csv"

path_consumptionFile = os.path.join(ExternalFilesFolder,ConsumptionFileName)
path_TemperatureFile = os.path.join(ExternalFilesFolder,TemperatureFileName)
path_IrradianceFile = os.path.join(ExternalFilesFolder,IrradianceFileName)

DF_consumption = pd.read_csv(path_consumptionFile,sep=",", index_col=0)
DF_consumption.head()
DF_consumption.tail(10)

PreviousIndex = DF_consumption.index
NewParsedIndex= pd.to_datetime(PreviousIndex)
DF_consumption.index =NewParsedIndex 

DF_consumption.head()
DF_consumption.index.hour
DF_consumption.index.month
DF_consumption.index.dayofweek

DF_consumption_someDaysInJuly=DF_consumption["2014-07-01 00:00:00":"2014-07-03 23:00:00"]

plt.figure()
plt.plot(DF_consumption_someDaysInJuly)
plt.xlabel("Time")
plt.ylabel("AC Power (W)")
plt.show()

# There is asecond way of doing this !!!
plt.figure()
DF_consumption_someDaysInJuly.plot()
plt.xlabel("Time")
plt.ylabel("AC Power (W)")
plt.show()

#This is better !!

# LEt's import the weather data 

DF_weather = pd.read_csv(path_TemperatureFile,sep=";",index_col=0)
DF_weather.head(24)
previousIndex_weather=DF_weather.index
newIndex_weather=pd.to_datetime(previousIndex_weather)
DF_weather.index = newIndex_weather
DF_weather.columns
Series_Temperature = DF_weather["temperature"]

DF_Temperature= DF_weather[["temperature"]]
DF_Temperature.head()


DF_irradianceSource = pd.read_csv(path_IrradianceFile,sep=";",index_col=1)
DF_irradianceSource.head(24)

DF_irradiance=DF_irradianceSource[["gen"]]
DF_irradiance.head(24)

DF_irradiance["gen"]<0
DF_irradiance[DF_irradiance["gen"]<0] = 0
DF_irradiance.head(24)

DF_joined = DF_consumption.join([DF_Temperature,DF_irradiance])
DF_joined.head(24)

DF_joined_cleaned = DF_joined.dropna()
DF_joined_cleaned.head(24)

DF_joined_cleaned_copy = DF_joined.dropna().copy()

DF_joined_cleaned_chosenDates = DF_joined_cleaned_copy["2014-08-01":"2014-08-04"]

# We need to solve the problem with the timezone
DF_joined_cleaned_chosenDates["temperature"]=DF_joined_cleaned_chosenDates["temperature"].shift(-5)
DF_joined_cleaned_chosenDates.dropna()
DF_joined_cleaned_chosenDates.head()
DF_joined_cleaned_chosenDates.describe()

DF_joined_cleaned_chosenDates_min=DF_joined_cleaned_chosenDates.min()
DF_joined_cleaned_chosenDates_max=DF_joined_cleaned_chosenDates.max()
DF_joined_cleaned_chosenDates_normalized= (DF_joined_cleaned_chosenDates-DF_joined_cleaned_chosenDates_min)/(DF_joined_cleaned_chosenDates_max-DF_joined_cleaned_chosenDates_min)
plt.figure()
DF_joined_cleaned_chosenDates_normalized.plot()


# Creating lagged features
# First let me reconstruct my dataframe by applying time-zone to the whole dataset

DF_joined = DF_consumption.join([DF_Temperature,DF_irradiance])
DF_mod = DF_joined.copy()
DF_mod["temperature"]=DF_mod["temperature"].shift(-5)
DF_mod.dropna(inplace=True) # It is the same as writing it like this: DF_mod=DF_mod.dropna()
# In order to consider the time-lags we will need to include 
# all of these columns with lags (shifts)
DF_mod.head()
DF_mod.describe()

DF_mod["Temperature -1h"]= DF_mod["temperature"].shift(1)
DF_mod["Temperature -2h"]= DF_mod["temperature"].shift(2)
DF_mod["Temperature -3h"]= DF_mod["temperature"].shift(2)
DF_mod["Temperature -4h"]= DF_mod["temperature"].shift(2)
DF_mod["Temperature -5h"]= DF_mod["temperature"].shift(2)
DF_mod["Temperature -6h"]= DF_mod["temperature"].shift(2)
DF_mod.head()

# but there is abetter way of doing this !!

 lag_start=1
 lag_end = 6
 lag_interval=1
 

column_name="temperature"
df=DF_mod
for i in range(lag_start,lag_end+1,lag_interval):
    new_column_name = column_name+" -"+str(i)+"hr"
    print new_column_name
    df[new_column_name]=df[column_name].shift(i)   
    df.dropna(inplace=True) #this removes all the row with a Nan

def lag_feature(df,column_name,lag_start,lag_end,lag_interval):
    for i in range(lag_start,lag_end+1,lag_interval):
        new_column_name = column_name+" -"+str(i)+"hr"
        print new_column_name
        df[new_column_name]=df[column_name].shift(i)   
        df.dropna(inplace=True) #this removes all the row with a Nan
    return df



# Let's do the same for the irradiance and consumption
# but I don't like the names of irradiance and consumption columns

# For renaming the column names , you have two ways

#DF_mod.columns =["AC_consumption","temperature","irradiance"]
# The second way of doing this:
DF_mod=DF_mod.rename(columns={"air conditioner_5545":"AC_consumption","gen":"irradiance"})

DF_mod=  lag_feature(DF_mod,"temperature",1,6,1)
DF_mod.head()
# Let's lag the irraidance , I would do it just for 3 to 6 hours
DF_mod=  lag_feature(DF_mod,"irradiance",3,6,1)

# Let's add the previous consumptions in the last 24 hours!
DF_mod=  lag_feature(DF_mod,"AC_consumption",1,24,1)
DF_mod.head()
DF_mod.describe()

# Now let's add the seasonality parameters (time-related parameters)

DF_mod["hour"]=DF_mod.index.hour
DF_mod["hour"].head()
DF_mod["sin_hour"]=np.sin(DF_mod.index.hour*2*np.pi/24)
DF_mod["cos_hour"]=np.cos(DF_mod.index.hour*2*np.pi/24)

DF_mod["day_of_week"]=DF_mod.index.dayofweek
DF_mod[["hour","sin_hour","cos_hour","day_of_week"]].head(24)

DF_mod["month"]=DF_mod.index.month

DF_mod["week_of_year"]=DF_mod.index.week

def WeekendDetector(day):
    if (day==5 or day == 6):
        weekendLabel=1
    else:
        weekendLabel=0
    return weekendLabel
4
DF_mod["weekend"]= DF_mod["day_of_week"].apply(WeekendDetector)
    
def DayDetector(hour):
    if (hour< 19 and hour>=9):
        DayLabel=1
    else:
        DayLabel=0
    return DayLabel

DF_mod["workingTime"] = DF_mod["hour"].apply(DayDetector)

DF_mod[["workingTime","weekend"]].head(24)

DF_mod.head()
DF_mod.columns

DF_mod.corr()
