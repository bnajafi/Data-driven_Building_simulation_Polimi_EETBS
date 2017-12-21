# -*- coding: utf-8 -*-



# # This notebook is a simple demonstration on how to load a time series data from a file, plot it and explore its properties.
# ## For any data-driven modeling, we may have to compare 2 or more dataset and find the similarities or correlation between them, if they have any.
# ## So, here we have 3 time series data set.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns


# #### Here i store some local folder paths in variables, so that i dont have to type out a long file location in my code.


DataFolderPath = "C:/Users/behzad/Dropbox/_2_Teaching Activities/_0_EETBS- On-going/git_fork_clone/Data-driven_Building_simulation_Polimi_EETBS/Data"
ConsumptionFileName = "consumption_5545.csv"
ConsumptionFilePath = DataFolderPath+"/"+ConsumptionFileName 

# #### H5 or HDF5 (Hierarchical Data Format) is a type for format for storing and managing data. Python Pandas can read such a file using the code below

# In[47]:
"""
combined = pd.HDFStore(join(consumption_path,'5545_AC.h5'))
#key = combined.keys()[0].replace('/','')
key = '5545'
dataframe = combined.get(key)
combined.close()
"""
#ataFolderPath="C:/Users/MANOJ/Dropbox/1 Data4Building- Review/2017_batch_projects/Projects/Presentation_data_driven"
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
DF_temperature = DF_temperature.shift(-6) # to account for the UTC to local time difference
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
DF_irradiance[DF_irradiance['gen'] <0.0] = 0
#Joined_DF= pd.concat([DF_consumption,series_temperature,series_irradiance])
df_joined = DF_consumption.join([DF_temperature,DF_irradiance])
df_joined.head(10)
# ### The temperature and irradiance datasets have been loaded into dataframe which is now joined with the AC consumption dataframe. by simply calling the join() function.

# ### We may have some rows where there maybe no values(empty), Pandas display them as "NaN". We don't need those rows, since it may interfere with further data exploration. We call the function dropna() to remove any rows, where at least one column values is "NaN"

# In[65]:

df_joined = df_joined.dropna()


# In[66]:

df_chosen_dates = df_joined['2014-06-16':'2014-06-18']

# Now we want to plot them with respect to each other: One solution is to normalize them !!

df_chosen_dates_normalized = (df_chosen_dates- df_chosen_dates.min())/(df_chosen_dates.max()-df_chosen_dates.min())

plt.figure()
df_chosen_dates_normalized.plot()


# Solution II 


plt.figure()
#plt.plot(df_joined["air conditioner_5545"])
df_chosen_dates.plot()

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3= fig.add_subplot(3,1,3)

df_chosen_dates.iloc[:,0].plot(ax=ax1,legend=True,color="b")
df_chosen_dates.iloc[:,1].plot(ax=ax2,legend=True,color="r")
df_chosen_dates.iloc[:,2].plot(ax=ax3,legend=True,color="g")
ax1.set_ylabel(" AC consumption", color="b")
ax2.set_ylabel(" Temperature ", color="r")
ax3.set_ylabel(" Irradiation (PV Gen)", color="g")
ax1.tick_params(axis='y',colors='b')
ax2.tick_params(axis='y',colors='r')
ax3.tick_params(axis='y',colors='g')

plt.figure()
ax1_1 = fig.add_subplot(111) # axis for consumption
ax2_1= ax1.twinx()          # axis for temperature
ax3_1= ax1.twinx()          # axis for solar irradiance
rspine = ax3_1.spines['right']
rspine.set_position(('axes', 1.1))
consum_col='air conditioner_5545'

df_chosen_dates.iloc[:,0].plot(ax=ax1_1, legend=False,color='b')
ax1_1.set_ylabel('Consumption',color='b')
ax1_1.tick_params(axis='y', colors='b')

df_chosen_dates.plot(ax=ax2_1, y='temperature', legend=False, color='g')
ax2_1.set_ylabel('Temperature deg C',color='g')
ax2_1.tick_params(axis='y',colors='g')

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




df_correlation = df_lagged.corr()   #Computes pairwise correlation of columns



# ### We filter temperature values less than 30 F and consumption values less than 10 W to remove most of the outliers.
# ### From the previous correlation plot we found that temperature shift of 3 hours and Irradiance shift of 6 hours matches well with the consumption.
# ### Now we can take the df_joined dataframe again, shift the temperature by 3 and irradiance by 6. Then perform a scatter plot to see the effect of temperature on the consumption.

# In[81]:

# remove outliers
"""
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
"""


# Let's Create the final DataSet

def features_creation(df):
    # creatures time based features from pandas dataframe
    # such hour of day, weekday/weekend, day/night and so on
    # sin hour and cos hour as just indirect representation of time of day
    df['sin_hour'] = np.sin((df.index.hour)*2*np.pi/24)
    df['cos_hour'] = np.cos((df.index.hour)*2*np.pi/24)#later try 24 vector binary format
    df['hour'] = df.index.hour # 0 to 23
    df['day_of_week'] = df.index.dayofweek #Monday = 0, sunday = 6
    df['weekend'] = [ 1 if day in (5, 6) else 0 for day in df.index.dayofweek ] # 1 for weekend and 0 for weekdays
    df['month'] = df.index.month
    df['week_of_year'] = df.index.week
    # day = 1 if(10Hrs -19Hrs) and Night = 0 (otherwise)
    df['day_night'] = [1 if day<20 and day>9 else 0 for day in df.index.hour ]
    return df
    
def lag_column(df,column_names,lag_period=1):
#df              > pandas dataframe
#column_names    > names of column/columns as a list
#lag_period      > number of steps to lag ( +ve or -ve) usually postive 
#to include past values for current row 
    for column_name in column_names:
        column_name = [str(column_name)]
        for i in np.arange(1,lag_period+1,1):
            new_column_name = [col +'_'+str(i) for col in column_name]
            df[new_column_name]=(df[column_name]).shift(i)
    return df

def normalize(df):
    return (df-df.min())/(df.max()-df.min())

df_FinalDataSet  = df_joined

# Let's first see what we can take from the fact that we converted everything into dataT
DF_chosenDuration = df_FinalDataSet["2014-01-01 13:00:00":"2014-01-01 21:00:00"]
DF_chosenDuration.index.hour
DF_chosenDuration.index.dayofweek # so it was a wednesday !
DF_chosenDuration.index.month
DF_chosenDuration.index.week



df_FinalDataSet['hour'] = df_FinalDataSet.index.hour # 0 to 23
#df_FinalDataSet['sin_hour'] = np.sin((df_FinalDataSet.index.hour)*2*np.pi/24)
#df_FinalDataSet['cos_hour'] = np.cos((df_FinalDataSet.index.hour)*2*np.pi/24)#later try 24 vector binary format
df_FinalDataSet['day_of_week'] = df_FinalDataSet.index.dayofweek #Monday = 0, sunday = 6
df_FinalDataSet['month'] = df_FinalDataSet.index.month
df_FinalDataSet['week_of_year'] = df_FinalDataSet.index.week

# here we need to know list comprehension !!!!
result = [x**2 for x in [2,3,5]]
z = [2,3,5,9,12,13,15]
#let write a list comprehension that would convert it to zero if x<5 and  5 if z>5
z_converted = [thisItem**2 for thisItem in z]

result2 = [0 if thisItem<5 else 5 for thisItem in z]

#let write a list comprehension that would convert it to zero if x<5 and  5 if z>5 and z<10 and finally 10 if z>10



# So let's use this list comprehension to build new features !!!!
# Let me first write a function

def DayDetector(hour):
    dayLabel=1
    if (hour<20 and hour > 9):
        dayLabel = 1
    else:
        dayLabel = 0
    return dayLabel

df_FinalDataSet['day_night'] = [DayDetector(thisHour) for thisHour in df_FinalDataSet.index.hour] # day = 1 if(10Hrs -19Hrs) and Night = 0 (otherwise

def WeekendDetector(day):
    weekendLabel = 0
    if (day==5 or day==6):
        weekendLabel = 1
    else:
        weekendLabel=0
    return weekendLabel

df_FinalDataSet['weekend'] = [ WeekendDetector(thisDay) for thisDay in df_FinalDataSet.index.dayofweek ] # 1 for weekend and 0 for weekdays
df_FinalDataSet.head()
# Let me add weather it was weekday or weekend !
"""
df_FinalDataSet['weekday_or_weekend'] = np.zeros(df_FinalDataSet.count()[0])
condition_weekend = ()
df_FinalDataSet['weekday_or_weekend'][df_FinalDataSet['day_of_week'] == 5] = 1
df_FinalDataSet['weekday_or_weekend'][df_FinalDataSet['day_of_week'] == 6] = 1
"""

"""optionally to check for national holidays!

accepted
You don't need to convert anything. Just compare straight up. pandas is smart enough to compare a lot of different types with regards to dates and times. You have to have a slightly more esoteric format if you're having issues with date/time compatibility.

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

dr = pd.date_range(start='2015-07-01', end='2015-07-31')
df = pd.DataFrame()
df['Date'] = dr

cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())

df['Holiday'] = df['Date'].isin(holidays)
print df"""

"""
# Let's first add temporal features
df_FinalDataSet['sin_hour'] = np.sin((df_FinalDataSet.index.hour)*2*np.pi/24)
df_FinalDataSet['cos_hour'] = np.cos((df_FinalDataSet.index.hour)*2*np.pi/24)#later try 24 vector binary format
df_FinalDataSet['hour'] = df_FinalDataSet.index.hour # 0 to 23
df_FinalDataSet['day_of_week'] = df_FinalDataSet.index.dayofweek #Monday = 0, sunday = 6
df_FinalDataSet['weekend'] = [ 1 if day in (5, 6) else 0 for day in df_FinalDataSet.index.dayofweek ] # 1 for weekend and 0 for weekdays
df_FinalDataSet['month'] = df_FinalDataSet.index.month
df_FinalDataSet['week_of_year'] = df_FinalDataSet.index.week
 # day = 1 if(10Hrs -19Hrs) and Night = 0 (otherwise)
df_FinalDataSet['day_night'] = [1 if ThisHour<20 and ThisHour>9 else 0 for ThisHour in df_FinalDataSet.index.hour ]
"""

# Let's change the first colum's name
df_FinalDataSet.rename(columns = {'air conditioner_5545':"AC_cons"},inplace=True)
df_FinalDataSet.head()

# This is a really stupid way of doing it !!
df_FinalDataSet_withLaggedFeatures = df_FinalDataSet
df_FinalDataSet_withLaggedFeatures["temperature-1hr"] = df_FinalDataSet_withLaggedFeatures["temperature"].shift(1)
df_FinalDataSet_withLaggedFeatures.head()
df_FinalDataSet_withLaggedFeatures["temperature-2hr"] = df_FinalDataSet_withLaggedFeatures["temperature"].shift(2)
df_FinalDataSet_withLaggedFeatures["temperature-3hr"] = df_FinalDataSet_withLaggedFeatures["temperature"].shift(3)
df_FinalDataSet_withLaggedFeatures["temperature-4hr"] = df_FinalDataSet_withLaggedFeatures["temperature"].shift(4)
df_FinalDataSet_withLaggedFeatures["temperature-5hr"] = df_FinalDataSet_withLaggedFeatures["temperature"].shift(5)

df_FinalDataSet_withLaggedFeatures["gen-5hr"] = df_FinalDataSet_withLaggedFeatures["gen"].shift(5)
df_FinalDataSet_withLaggedFeatures["gen-6hr"] = df_FinalDataSet_withLaggedFeatures["gen"].shift(6)


# For consumptions we will use an automatic way !!

def lag_column(df,column_name,lag_period=1):
    for i in range(1,lag_period+1,1):
        new_column_name = column_name+" -"+str(i)+"hr"
        df[new_column_name]=(df[column_name]).shift(i)
    return df
    
df_FinalDataSet_withLaggedFeatures = lag_column(df_FinalDataSet_withLaggedFeatures,"AC_cons",24)
df_FinalDataSet_withLaggedFeatures.head(24)

# Now I should remove all the lines with a  NAN
df_FinalDataSet_withLaggedFeatures.dropna(inplace=True)

# Now let's choose the features columns and the target one 

def normalize(df):
    return (df-df.min())/(df.max()-df.min())

DF_target = df_FinalDataSet_withLaggedFeatures["AC_cons"]
DF_features = df_FinalDataSet_withLaggedFeatures.drop("AC_cons",axis=1)

df_FinalDataSet_withLaggedFeatures_norm = normalize(df_FinalDataSet_withLaggedFeatures)
DF_target_norm = df_FinalDataSet_withLaggedFeatures_norm["AC_cons"]
DF_features_norm = df_FinalDataSet_withLaggedFeatures_norm.drop("AC_cons",axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF_features, DF_target, test_size=0.2, random_state=41234)
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(DF_features_norm, DF_target_norm, test_size=0.2, random_state=41234)




from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

linear_reg.fit(X_train, y_train)
predict_linearReg_split= linear_reg.predict(X_test)
predict_DF_linearReg_split=pd.DataFrame(predict_linearReg_split, index = y_test.index,columns=["AC_ConsPred_linearReg_split"])

predict_DF_linearReg_split = predict_DF_linearReg_split.join(y_test)

#predictions = pd.Series(predict.ravel(),index=y_test.index).rename("AC_consump"+"_predicted")
#predictions_frame = pd.DataFrame(predictions).join(y_test)

predict_DF_linearReg_split['2014-08-01':'2014-08-20'].plot()
plt.xlabel('Time')
plt.ylabel('AC Power [w]')
plt.ylim([0,4000])

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
metric_R2_score = r2_score(y_test,predict_linearReg_split)
metric_mean_absolute_error = mean_absolute_error(y_test,predict_linearReg_split)
metric_mean_squared_error = mean_squared_error(y_test,predict_linearReg_split)
coeff_variation = np.sqrt(metric_mean_squared_error)/y_test.mean()

#Let's see how is the situation with cross validation
from sklearn.model_selection import cross_val_predict
predict_linearReg_CV = cross_val_predict(linear_reg,DF_features,DF_target,cv=10)
 
predict_DF_linearReg_CV=pd.DataFrame(predict_linearReg_CV, index = DF_target.index,columns=["AC_ConsPred_linearReg_CV"])

predict_DF_linearReg_CV = predict_DF_linearReg_CV.join(DF_target)

#predictions = pd.Series(predict.ravel(),index=y_test.index).rename("AC_consump"+"_predicted")
#predictions_frame = pd.DataFrame(predictions).join(y_test)

predict_DF_linearReg_CV['2014-08-01':'2014-08-20'].plot()

import seaborn as sns
fig = plt.figure()
ax1 = fig.add_subplot(111)
plot = sns.regplot(x="AC_cons", y="AC_ConsPred_linearReg_CV",
                   data=predict_DF_linearReg_CV,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Actual consumption VS. Predicted consumption')
plot.set_xlim([0,3000])
plot.set_ylim([0,3000])
plot.set_xlabel('Actual AC consumption')
plot.set_ylabel('Predicted AC consumption')
regline = plot.get_lines()[0];
regline.set_color('red')

R2_score_linearReg_CV = r2_score(predict_DF_linearReg_CV["AC_cons"],predict_DF_linearReg_CV["AC_ConsPred_linearReg_CV"])
mean_absolute_error_linearReg_CV = mean_absolute_error(predict_DF_linearReg_CV["AC_cons"],predict_DF_linearReg_CV["AC_ConsPred_linearReg_CV"])
mean_squared_error_linearReg_CV = mean_squared_error(predict_DF_linearReg_CV["AC_cons"],predict_DF_linearReg_CV["AC_ConsPred_linearReg_CV"])
coeff_variation_linearReg_CV = np.sqrt(metric_mean_squared_error)/predict_DF_linearReg_CV["AC_cons"].mean()

# Let's evaluate support vector machines
from sklearn.svm import SVR
SVR_reg = SVR(kernel='rbf',C=10,gamma=1)
#Input for SVR should be normalized tables only
predict_SVR_CV = cross_val_predict(SVR_reg,DF_features_norm,DF_target_norm,cv=10)
 
predict_DF_SVR_CV=pd.DataFrame(predict_SVR_CV, index = DF_target_norm.index,columns=["AC_ConsPred_SVR_CV"])

predict_DF_SVR_CV = predict_DF_SVR_CV.join(DF_target_norm).dropna()

#predictions = pd.Series(predict.ravel(),index=y_test.index).rename("AC_consump"+"_predicted")
#predictions_frame = pd.DataFrame(predictions).join(y_test)

predict_DF_SVR_CV['2014-08-01':'2014-08-20'].plot()


fig = plt.figure()
ax1 = fig.add_subplot(111)
plot = sns.regplot(x="AC_cons", y="AC_ConsPred_SVR_CV",
                   data=predict_DF_SVR_CV,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Actual consumption VS. Predicted consumption')
plot.set_xlim([0,1])
plot.set_ylim([0,1])
plot.set_xlabel('Actual AC consumption')
plot.set_ylabel('Predicted AC consumption')
regline = plot.get_lines()[0];
regline.set_color('red')

R2_score_DF_SVR_CV = r2_score(predict_DF_SVR_CV["AC_cons"],predict_DF_SVR_CV["AC_ConsPred_SVR_CV"])
mean_absolute_error_SVR_CV = mean_absolute_error(predict_DF_SVR_CV["AC_cons"],predict_DF_SVR_CV["AC_ConsPred_SVR_CV"])
mean_squared_error_SVR_CV = mean_squared_error(predict_DF_SVR_CV["AC_cons"],predict_DF_SVR_CV["AC_ConsPred_SVR_CV"])
coeff_variation_SVR_CV = np.sqrt(mean_squared_error_SVR_CV)/predict_DF_SVR_CV["AC_cons"].mean()


from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()
predict_RF_CV = cross_val_predict(reg_RF,DF_features,DF_target,cv=10)
 
predict_DF_RF_CV=pd.DataFrame(predict_RF_CV, index = DF_target.index,columns=["AC_ConsPred_RF_CV"])

predict_DF_RF_CV = predict_DF_RF_CV.join(DF_target).dropna()
predict_DF_RF_CV['2014-08-01':'2014-08-20'].plot()

fig = plt.figure()
ax1 = fig.add_subplot(111)
plot = sns.regplot(x="AC_cons", y="AC_ConsPred_RF_CV",
                   data=predict_DF_RF_CV,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Actual consumption VS. Predicted consumption')
plot.set_xlim([0,3000])
plot.set_ylim([0,3000])
plot.set_xlabel('Actual AC consumption')
plot.set_ylabel('Predicted AC consumption')
regline = plot.get_lines()[0];
regline.set_color('red')

R2_score_DF_RF_CV = r2_score(predict_DF_RF_CV["AC_cons"],predict_DF_RF_CV["AC_ConsPred_RF_CV"])
mean_absolute_error_DF_CV = mean_absolute_error(predict_DF_RF_CV["AC_cons"],predict_DF_RF_CV["AC_ConsPred_RF_CV"])
mean_squared_error_DF_CV = mean_squared_error(predict_DF_RF_CV["AC_cons"],predict_DF_RF_CV["AC_ConsPred_RF_CV"])
coeff_variation_DF_CV = np.sqrt(mean_squared_error_DF_CV)/predict_DF_RF_CV["AC_cons"].mean()

from sknn.mlp import Regressor, Layer
reg_NN = Regressor(layers=[Layer("Rectifier",units=5),  # Hidden Layer1
                                Layer("Rectifier",units=3),  # Hidden Layer2
                                Layer("Linear")],            # Output Layer
                        n_iter = 100, learning_rate=0.02)
reg_NN.fit(X_train_norm.as_matrix(), y_train_norm.as_matrix())
predict_DF_NN = reg_NN.predict(X_test_norm.as_matrix())

predict_DF_NN_CV=pd.DataFrame(predict_DF_NN, index = y_test_norm.index,columns=["AC_ConsPred_NN_CV"])
predict_DF_NN_CV = predict_DF_NN_CV.join(y_test_norm).dropna()
predict_DF_NN_CV['2014-08-01':'2014-08-20'].plot()

R2_score_DF_NN_CV = r2_score(predict_DF_NN_CV["AC_cons"],predict_DF_NN_CV["AC_ConsPred_NN_CV"])
mean_absolute_error_DF_CV = mean_absolute_error(predict_DF_NN_CV["AC_cons"],predict_DF_NN_CV["AC_ConsPred_NN_CV"])
mean_squared_error_DF_CV = mean_squared_error(predict_DF_NN_CV["AC_cons"],predict_DF_NN_CV["AC_ConsPred_NN_CV"])
coeff_variation_DF_CV = np.sqrt(mean_squared_error_DF_CV)/predict_DF_NN_CV["AC_cons"].mean()



