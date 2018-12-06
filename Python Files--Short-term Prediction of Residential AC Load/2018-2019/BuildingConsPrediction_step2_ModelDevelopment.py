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
"""
DF_mod["Temperature -1h"]= DF_mod["temperature"].shift(1)
DF_mod["Temperature -2h"]= DF_mod["temperature"].shift(2)
DF_mod["Temperature -3h"]= DF_mod["temperature"].shift(2)
DF_mod["Temperature -4h"]= DF_mod["temperature"].shift(2)
DF_mod["Temperature -5h"]= DF_mod["temperature"].shift(2)
DF_mod["Temperature -6h"]= DF_mod["temperature"].shift(2)
DF_mod.head()
"""
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

DF_mod.describe()
DF_mod.info()
DF_mod=DF_mod["2014-03-01":"2014-09-30"]

DF_target= DF_mod["AC_consumption"] # I am actually making a series , we could also use double brackets
DF_features= DF_mod.drop("AC_consumption", axis=1) # I am actually m.dropaking a series , we could also use double brackets

# From now on we are use the marvelous sklearn !!!
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(DF_features,DF_target,test_size = 0.2, random_state=41234)

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

# The second step will be fitting a model
linear_reg.fit(X_train, Y_train)

predicted_linearReg_split = linear_reg.predict(X_test)

predicted_DF_linearReg_split=pd.DataFrame(predicted_linearReg_split,index=Y_test.index, columns=["AC_cons_predicted_linearReg_split"])
predicted_DF_linearReg_split=predicted_DF_linearReg_split.join(Y_test)

predicted_DF_linearReg_split_august=predicted_DF_linearReg_split["2014-08-01":"2014-08-31"]
predicted_DF_linearReg_split_august.plot()

# Now we want calculate how accurate our predictions are !!
# again we import everything
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
MAE_linearReg_split= mean_absolute_error(predicted_linearReg_split,Y_test)
MSE_linearReg_split= mean_squared_error(predicted_linearReg_split,Y_test)
R2_linearReg_split = r2_score(predicted_linearReg_split,Y_test)

# The second way of doing this is using k-fold cross valiadtion
from sklearn.model_selection import cross_val_predict
predict_linearReg_CV = cross_val_predict(linear_reg,DF_features,DF_target,cv=10)
predicted_DF_linearReg_CV=pd.DataFrame(predict_linearReg_CV,
                                       index=DF_target.index, 
                                       columns=["AC_cons_predicted_linearReg_CV"])
predicted_DF_linearReg_CV=predicted_DF_linearReg_CV.join(DF_target)
predicted_DF_linearReg_CV_august=predicted_DF_linearReg_CV["2014-08-01":"2014-08-31"]
predicted_DF_linearReg_CV_august.plot()

MAE_linearReg_CV= mean_absolute_error(predict_linearReg_CV,DF_target)
MSE_linearReg_CV= mean_squared_error(predict_linearReg_CV,DF_target)
R2_linearReg_CV = r2_score(predict_linearReg_CV,DF_target)


# Now, let's try another algorithm! Random forests are a very good candidate!
from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()

predict_RF_CV = cross_val_predict(reg_RF,DF_features,DF_target,cv=10)

predicted_DF_RF_CV=pd.DataFrame(predict_RF_CV,
                                       index=DF_target.index, 
                                       columns=["AC_cons_predicted_RF_CV"])
predicted_DF_RF_CV=predicted_DF_RF_CV.join(DF_target)
predicted_DF_RF_CV_august=predicted_DF_RF_CV["2014-08-01":"2014-08-31"]
predicted_DF_RF_CV_august.plot()

MAE_RF_CV= mean_absolute_error(predict_RF_CV,DF_target)
MSE_RF_CV= mean_squared_error(predict_RF_CV,DF_target)
R2_RF_CV = r2_score(predict_RF_CV,DF_target)


#• What if we want to use online learning
DF_onlineConsumptionPrediction = pd.DataFrame(index=DF_mod.index)
period_of_training = pd.Timedelta(10, unit="d")

FirstTimeStamp_measured = DF_mod.index[0]
LastTimeStamp_measured = DF_mod.index[-1]

FirstTimeStamp_toPredict= FirstTimeStamp_measured+period_of_training

training_startTimeStamp=FirstTimeStamp_measured
training_endTimeStamp=FirstTimeStamp_toPredict # it is called end because it will include this time stamp !!

TimeStamp_toPredict=FirstTimeStamp_toPredict
while TimeStamp_toPredict<LastTimeStamp_measured:
 
    #print("time Stamp to be predicted is:")
    #print (TimeStamp_toPredict) 
    DF_features_train = DF_features.truncate(before=training_startTimeStamp,after= training_endTimeStamp)
    DF_target_train = DF_target.truncate(before=training_startTimeStamp,after= training_endTimeStamp)
 
    DF_features_test = DF_features.loc[TimeStamp_toPredict].values.reshape(1,-1)
    DF_target_test = DF_target.loc[TimeStamp_toPredict]
        
    reg_RF.fit(DF_features_train, DF_target_train)
    predicted_consumption= reg_RF.predict(DF_features_test)  
    DF_OnlineConsumptionPrediction.loc[TimeStamp_toPredict,"Real"] =DF_target_test
    DF_OnlineConsumptionPrediction.loc[TimeStamp_toPredict,"Predicted"] =predicted_consumption

    
    TimeStamp_toPredict=TimeStamp_toPredict+pd.Timedelta(1,"h")
    training_endTimeStamp=training_endTimeStamp+pd.Timedelta(1,"h")
    training_startTimeStamp=training_startTimeStamp+pd.Timedelta(1,"h")

DF_features_test


DF_OnlineConsumptionPrediction.dropna(inplace=True)
R2_score_linearReg_CV = r2_score(DF_OnlineConsumptionPrediction[["Real"]],DF_OnlineConsumptionPrediction[["Predicted"]])
DF_OnlineConsumptionPrediction.plot()

DF_OnlineConsumptionPrediction_august=DF_OnlineConsumptionPrediction["2014-08-01":"2014-08-31"]
DF_OnlineConsumptionPrediction_august.plot()