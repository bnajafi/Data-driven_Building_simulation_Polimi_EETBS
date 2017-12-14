# -*- coding: utf-8 -*-



# # This notebook is a simple demonstration on how to load a time series data from a file, plot it and explore its properties.
# ## For any data-driven modeling, we may have to compare 2 or more dataset and find the similarities or correlation between them, if they have any.
# ## So, here we have 3 time series data set.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns


# #### Here i store some local folder paths in variables, so that i dont have to type out a long file location in my code.


#consumption_path = 'C:/Users/MANOJ/ML_SmartMeterAnalytics/data/consumption/'
#data_path = 'C:/Users/MANOJ/ML_SmartMeterAnalytics/data/'
#weather_path = 'C:/Users/MANOJ/ML_SmartMeterAnalytics/data/weather/'


# #### H5 or HDF5 (Hierarchical Data Format) is a type for format for storing and managing data. Python Pandas can read such a file using the code below

# In[47]:
"""
combined = pd.HDFStore(join(consumption_path,'5545_AC.h5'))
#key = combined.keys()[0].replace('/','')
key = '5545'
dataframe = combined.get(key)
combined.close()
"""
DataFolderPath="C:/Users/behzad/Dropbox/_2_Teaching Activities/_0_EETBS- On-going/Datadriven building Lessons/Files to GIT/Data"
ConsumptionFileName = "consumption_5545.csv"
ConsumptionFilePath= DataFolderPath+"/"+ConsumptionFileName
DF_consumption = pd.read_csv(ConsumptionFilePath,sep=",",index_col=0)
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


# Now let's see what does the weather files have:

# ## Visualizing the AC consumption with Temperature and Solar irradiance
# ### The temperature dataset is avaiable along with humidity, wind speed and more such paramters in a CSV format. For now we just take the temperature alone from the CSV file. Similarly we load the solar irradiance data.
weatherFileName = "Austin_weather_2014.csv"
weatherFilePath = DataFolderPath+"/"+weatherFileName
DF_weather_source = pd.read_csv(weatherFilePath,sep = ";",index_col=0)
previousIndex= DF_weather_source.index
NewparsedIndex = pd.to_datetime(previousIndex)
DF_weather_source.index= NewparsedIndex
DF_temperature = DF_weather_source[["temperature"]]
#series_temperature = DF_weather_source["temperature"]

# ### The irradiance data was obtained from the PV generation units from the buildings. The system may sometimes record negative values, which makes no sense from an irradiance point of view. So the negative values have been set to zero after loading the data

# In[31]:
IrradianceSourceFileName = "irradiance_2014_gen.csv"
IrradianceSourceFilePath = DataFolderPath+"/"+IrradianceSourceFileName
DF_irradiance_source = pd.read_csv(IrradianceSourceFilePath,sep = ";",index_col=1)
previousIndex= DF_irradiance_source.index
NewparsedIndex = pd.to_datetime(previousIndex)
DF_irradiance_source.index= NewparsedIndex

#series_irradiance = DF_irradiance_source["gen"]

DF_irradiance = DF_irradiance_source[["gen"]]
#series_irradiance[series_irradiance <0.0]=0
#DF_irradiance[DF_irradiance['gen'] <0.0] = 07uyu8
#Joined_DF= pd.concat([DF_consumption,series_temperature,series_irradiance])
df_joined = DF_consumption.join([DF_temperature,DF_irradiance])
df_joined.head(10)
# ### The temperature and irradiance datasets have been loaded into dataframe which is now joined with the AC consumption dataframe. by simply calling the join() function.

# ### We may have some rows where there maybe no values(empty), Pandas display them as "NaN". We don't need those rows, since it may interfere with further data exploration. We call the function dropna() to remove any rows, where at least one column values is "NaN"

# In[65]:

df_joined = df_joined.dropna()


# In[66]:

df_chosen_dates = df_joined['2014-06-10':'2014-06-12']

plt.figure()
#plt.plot(df_joined["air conditioner_5545"])
df_chosen_dates.plot()

fig = plt.figure()
ax1=fig.add_subplot()
ax2=ax1.twinx()
ax3=ax1.twinx()
ax1 = fig.add_subplot(111) # axis for consumption
ax2 = ax1.twinx()          # axis for temperature
ax3 = ax1.twinx()          # axis for solar irradiance
rspine = ax3.spines['right']
rspine.set_position(('axes', 1.1))
consum_col='air conditioner_5545'

df_chosen_dates.plot(ax=ax1, y=consum_col, legend=False,color='b')
ax1.set_ylabel('Consumption',color='b')
ax1.tick_params(axis='y', colors='b')

df_chosen_dates.plot(ax=ax2, y='temperature', legend=False, color='g')
ax2.set_ylabel('Temperature deg C',color='g')
ax2.tick_params(axis='y',colors='g')

df_chosen_dates.plot(ax=ax3,y='gen',legend=False,color='r')
ax3.set_ylabel('Irradiance [from PV]',color='r')
ax3.tick_params(axis='y',colors='r')

ax1.set_xlabel('Time')
plt.show()


# In[76]:

df_lagged = df_joined.copy()
df_lagged['temperature_1'] = df_lagged['temperature'].shift(1)
df_lagged['temperature_2'] = df_lagged['temperature'].shift(2)
df_lagged['temperature_3'] = df_lagged['temperature'].shift(3)
df_lagged['temperature_4'] = df_lagged['temperature'].shift(4)
df_lagged['gen_1'] = df_lagged['gen'].shift(1)
df_lagged['gen_2'] = df_lagged['gen'].shift(2)
df_lagged['gen_3'] = df_lagged['gen'].shift(3)
df_lagged['gen_4'] = df_lagged['gen'].shift(4)
df_lagged['gen_5'] = df_lagged['gen'].shift(5)
df_lagged['gen_6'] = df_lagged['gen'].shift(6)
df_lagged['gen_7'] = df_lagged['gen'].shift(7)

df_lagged.dropna(inplace=True)


# In[82]:
    # creatures time based features from pandas dataframe
    # such hour of day, weekday/weekend, day/night and so on
    # sin hour and cos hour as just indirect representation of time of day
df_lagged['sin_hour'] = np.sin((df_lagged.index.hour)*2*np.pi/24)
df_lagged['cos_hour'] = np.cos((df_lagged.index.hour)*2*np.pi/24)#later try 24 vector binary format
df_lagged['hour'] = df_lagged.index.hour # 0 to 23
df_lagged['day_of_week'] = df_lagged.index.dayofweek #Monday = 0, sunday = 6
df_lagged['weekend'] = [ 1 if day in (5, 6) else 0 for day in df.index.dayofweek ] # 1 for weekend and 0 for weekdays
df['month'] = df.index.month
df['week_of_year'] = df.index.week
# day = 1 if(10Hrs -19Hrs) and Night = 0 (otherwise)
df['day_night'] = [1 if day<20 and day>9 else 0 for day in df.index.hour ]
return df

df_correlation = df_lagged.corr()   #Computes pairwise correlation of columns

fig = plt.figure()
#plot = fig.add_axes()
plot = sns.heatmap(df_correlation, annot=True)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()


# ### We filter temperature values less than 30 F and consumption values less than 10 W to remove most of the outliers.
# ### From the previous correlation plot we found that temperature shift of 3 hours and Irradiance shift of 6 hours matches well with the consumption.
# ### Now we can take the df_joined dataframe again, shift the temperature by 3 and irradiance by 6. Then perform a scatter plot to see the effect of temperature on the consumption.

# In[81]:

# remove outliers
df = df_joined.loc[(df_joined['temperature']>30) & (df_joined[consum_col] > 10)]

df_temperature = df.copy()
df_temperature['temperature'] = df_temperature['temperature'].shift(3)
df_temperature.dropna(inplace=True)

fig = plt.figure()

ax1 = fig.add_subplot(121)
plot = sns.regplot(x='temperature', y=consum_col, data=df_temperature,ax=ax1)
plt.title('Effect of temperature on AC consumption')
plot.set_xlim([60,110])
plot.set_ylim([0,3000])
plot.set_xlabel('Temperature Fahrenheit')
plot.set_ylabel('AC consumption')
regline = plot.get_lines()[0];
regline.set_color('red')
regline.set_zorder('5')

ax2 = fig.add_subplot(122)
df_generation = df.copy()
df_generation['gen'] = df_generation['gen'].shift(6)
df_generation.dropna(inplace=True)
gen_plot = sns.regplot(x='gen', y=consum_col, data=df_generation,ax=ax2)
plt.title('Effect of irradiance on AC consumption')
gen_plot.set_xlim([0.1,5])
gen_plot.set_ylim([0,3000])
gen_plot.set_xlabel('PV generation kW (Solar Irradiance)')
gen_plot.set_ylabel('AC consumption')
regline = gen_plot.get_lines()[0];
regline.set_color('red')
regline.set_zorder('5')

