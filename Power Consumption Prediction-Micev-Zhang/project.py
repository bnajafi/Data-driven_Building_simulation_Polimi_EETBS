import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
DataFolderPath="C:/Users/Pc/Desktop/politecnico/Energy and Enviromental Technologies For Building Systems/project data/0 Data"
DataFileName="lbnlb74electricity.csv"
DataFile=DataFolderPath+'/'+ DataFileName
DF_data=pd.read_csv(DataFile,sep=";",index_col=0) #way of extracting data into DataFrame in panda
DF_data.head()
DF_data.index=pd.to_datetime(DF_data.index)
DF_data["hour"]=DF_data.index.hour
DF_data["day of week"]=DF_data.index.dayofweek
DF_data["month"]=DF_data.index.month #month 
DF_data["week of year"]=DF_data.index.week
DF_data.head()
def weekendDetector(day):
    weekendLabel=0
    if (day==5 or day ==6):
        weekendLabel=1
    return weekendLabel
def dayDetector(hour):
    dayLabel=0
    if (hour<21 and hour>8):
        dayLabel=1
    return dayLabel
def SummSpringDetector(month):
    monthDetector=0
    if(month>3 and month<10):
        monthDetector=1
    return monthDetector
Hollidays_USA=["2014-01-01","2014-01-20","2014-02-17","2014-05-26" ,"2014-07-04","2014-10-01","2014-11-13","2014-11-27","2014-12-25","2015-01-01","2015-01-19","2015-02-16","2015-05-25" ]
       
DF_data["weekend or weekday"]=[weekendDetector(day) for day in DF_data["day of week"]]
DF_data["day or night"]=[dayDetector(hour) for hour in DF_data["hour"]]
DF_data["summer or winter"]=[SummSpringDetector(month) for month in DF_data["month"]]
for day in Hollidays_USA:
    DF_data["weekend or weekday"][day]=1

