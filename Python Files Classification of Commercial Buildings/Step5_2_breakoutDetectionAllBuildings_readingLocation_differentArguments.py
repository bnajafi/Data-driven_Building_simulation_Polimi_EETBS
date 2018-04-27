import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import timeit
import os
from datetime import datetime
#from __future__ import division
from pylab import *
import matplotlib.dates as mdates
from matplotlib import ticker
import datetime
import matplotlib
import getpass

#import rpy2
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

# Processes Data Folders
processedDataFolder  = r"C:\Users\behzad\Dropbox\3 Research Projects\2 Data for Building\BuildingDataGenomeProject\the-building-data-genome-project\ProcessesData"
individualBuildingFolderName = "individualBuildings"
individualBuildingFolderPath= os.path.join(processedDataFolder,individualBuildingFolderName)
BreakOutLocationReportMainFolder = "BreakoutReport"
BreakOutLocationReportMainFolderPath =  os.path.join(processedDataFolder,BreakOutLocationReportMainFolder)
ListOfExistingFiles=  os.listdir(individualBuildingFolderPath)
FinalBreakOutReportFolder= r"C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/ProcessesData/BreakoutReport/__FinalReport__"

breakouts = pd.DataFrame()

minimum_set = [10,30,60]
#○minimum_set = [10]
beta_set = [0.002,0.005,0.010] 
#beta_set = [0.002] 
degrees_set = [2,3,5]
for Minimum in minimum_set:
    for Beta in beta_set:
        for Degrees in degrees_set:
            subFolder = "Minimum_"+str(Minimum)+"_Beta_"+str(Beta)+"_Degrees_"+str(Degrees)
            print subFolder
            breakoutLocationReportSubFolder = os.path.join(BreakOutLocationReportMainFolderPath,subFolder)
            breakouts_single = pd.DataFrame()
            
            for building in DF_metaData.index:
                #print building 
                timezone = DF_metaData.T[building].timezone
                start = DF_metaData.T[building].datastart
                end = DF_metaData.T[building].dataend
                building_data = DF_temporalData[building].tz_convert(timezone).truncate(before=start,after=end).resample('D').sum()
                            
                name_locations_building = "Locations_"+building+".csv"
                path_locations_building = os.path.join(breakoutLocationReportSubFolder,name_locations_building)
                #print path_locations_building
                Locations_DF = pd.read_csv(path_locations_building)
                Locations_DF = Locations_DF["Locations_"+building]
                locations =Locations_DF.tolist()
                prevloc = 0
                bo_counter = 0
                for location in locations:
                    building_data[prevloc:location] = bo_counter
                    prevloc = location
                    bo_counter+=1
                building_data[prevloc:] = bo_counter
                breakout = pd.DataFrame({building:building_data})
                breakout = breakout.reset_index(drop=True)
                breakouts_single = pd.merge(breakouts_single, breakout, right_index=True, left_index=True, how='outer')
              #ThisBreakOutReportPath= os.path.join(FinalBreakOutReportFolder,ThisBreakOutReportName)
            ThisBreakOutReportName = "temp_breakouts_"+str(Minimum)+"_"+str(Beta)+"_"+str(Degrees)+".csv"
            ThisBreakOutReportPath = FinalBreakOutReportFolder+"/"+ThisBreakOutReportName
            print "*****************" + ThisBreakOutReportPath+"***************"

            breakouts_single.to_csv(ThisBreakOutReportPath)
            
            NameOfThisColumn = "BG_breakouts_max_"+str(Minimum)+"_"+str(Beta)[-1:]+"_"+str(Degrees)
            print NameOfThisColumn
            breakout_tocombine = pd.DataFrame({NameOfThisColumn:breakouts_single.max()})
                                        
            breakouts = pd.merge(breakouts, breakout_tocombine, right_index=True, left_index=True, how='outer')
            breakouts.index.name = "building_name"
            breakouts.columns.name = "feature_name"
                
FinalBreakOutReportName = "feature_breakouts.csv"
#FinalBreakOutReportFolderName= "__FinalReport__"
#FinalBreakOutReportFolder = os.path.join(processedDataFolder,FinalBreakOutReportFolderName)
FinalBreakOutReportPath = FinalBreakOutReportFolder+"/"+FinalBreakOutReportName
breakouts.to_csv(FinalBreakOutReportPath)
#print breakouts





