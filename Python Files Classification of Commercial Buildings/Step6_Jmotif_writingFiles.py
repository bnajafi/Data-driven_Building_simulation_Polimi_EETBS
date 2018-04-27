# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:04:28 2018

@author: behzad
"""

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
BuildingData_PrimaryUsage_FolderName = "BuildingData_PrimaryUsage"
BuildingData_PrimaryUsageFolderPath= os.path.join(processedDataFolder,BuildingData_PrimaryUsage_FolderName)

JmotifReportFolder = "JmotifReport"
JmotifReportFolderReportFolderPath =  os.path.join(processedDataFolder,JmotifReportFolder)
#ListOfExistingFiles=  os.listdir(individualBuildingFolderPath)
#FinalBreakOutReportFolder= r"C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/ProcessesData/BreakoutReport/__FinalReport__"



DF_temporalData.ix[:,:30].info()

DF_metaData.primaryspaceusage.unique()

def get_tempdata_perclass(temp, meta, primaryuse):
    df = temp[list(meta[(meta['primaryspaceusage'] == primaryuse)].index)]
    df_reset_axis = pd.DataFrame()
    for building in df.columns:
        sample = df[building].dropna().reset_index(drop=True)
        df_reset_axis[building] = sample
    return df_reset_axis.dropna()

office_data_lab = get_tempdata_perclass(DF_temporalData, DF_metaData, "Office")
dorm_data_lab = get_tempdata_perclass(DF_temporalData, DF_metaData, "Dormitory")
lab_data_lab = get_tempdata_perclass(DF_temporalData, DF_metaData, "College Laboratory")
colclass_data_lab = get_tempdata_perclass(DF_temporalData, DF_metaData, "College Classroom")
primsecclass_data_lab = get_tempdata_perclass(DF_temporalData, DF_metaData, "Primary/Secondary Classroom")


office_data_lab.head()


office_data = office_data_lab.reset_index(drop=True).T.reset_index(drop=True)
dorm_data = dorm_data_lab.reset_index(drop=True).T.reset_index(drop=True)
lab_data = lab_data_lab.reset_index(drop=True).T.reset_index(drop=True)
colclass_data = colclass_data_lab.reset_index(drop=True).T.reset_index(drop=True)
primsecclass_data = primsecclass_data_lab.reset_index(drop=True).T.reset_index(drop=True)

office_data.info()

# Now let's write them to CSV file
office_data_Lab_filePath= os.path.join(BuildingData_PrimaryUsageFolderPath,"office_data_lab.csv")
dorm_data_Lab_filePath= os.path.join(BuildingData_PrimaryUsageFolderPath,"dorm_data_lab.csv")
lab_data_Lab_filePath= os.path.join(BuildingData_PrimaryUsageFolderPath,"lab_data_lab.csv")
colclass_data_Lab_filePath= os.path.join(BuildingData_PrimaryUsageFolderPath,"colclass_data_lab.csv")
primsecclass_data_Lab_filePath= os.path.join(BuildingData_PrimaryUsageFolderPath,"primsecclass_data_lab.csv")

office_data_lab.to_csv(office_data_Lab_filePath)
dorm_data_lab.to_csv(dorm_data_Lab_filePath)
lab_data_lab.to_csv(lab_data_Lab_filePath)
colclass_data_lab.to_csv(colclass_data_Lab_filePath)
primsecclass_data_lab.to_csv(primsecclass_data_Lab_filePath)







office_data_filePath= os.path.join(BuildingData_PrimaryUsageFolderPath,"office_data.csv")
dorm_data_filePath= os.path.join(BuildingData_PrimaryUsageFolderPath,"dorm_data.csv")
lab_data_filePath= os.path.join(BuildingData_PrimaryUsageFolderPath,"lab_data.csv")
colclass_data_filePath= os.path.join(BuildingData_PrimaryUsageFolderPath,"colclass_data.csv")
primsecclass_data_filePath= os.path.join(BuildingData_PrimaryUsageFolderPath,"primsecclass_data.csv")

office_data.to_csv(office_data_filePath)
dorm_data.to_csv(dorm_data_filePath)
lab_data.to_csv(lab_data_filePath)
colclass_data.to_csv(colclass_data_filePath)
primsecclass_data.to_csv(primsecclass_data_filePath)


buildinglistlist = [[office_data_lab,"Office","Dorm","Lab","ColClass","PrimSec"], [dorm_data_lab,"Dorm","Office","Lab","ColClass","PrimSec"], 
                    [lab_data_lab,"Lab","Office","Dorm","ColClass","PrimSec"], [colclass_data_lab,"ColClass","Office","Dorm","Lab","PrimSec"],
                    [primsecclass_data_lab,"PrimSec","Dorm","Office","Lab","ColClass"]]

for buildinglist in buildinglistlist:
    print "Getting specificity for list whose first building is "+buildinglist[0].ix[0,:].index[0]
    print buildinglist
    start_time = timeit.default_timer()
    for building in buildinglist[0].columns:
        print building
        # we should now read the data generated by R
        nameTimeIndexBuilding= "time_index_"+building+".csv"
        nameSampleSaxBuilding = "Sax_value_"+building+".csv"
        nameCosineBuilding= "cosines_"+building+".csv"
        path_TimeIndexBuilding = os.path.join(JmotifReportFolderReportFolderPath,nameTimeIndexBuilding)
        path_SampleSaxBuilding = os.path.join(JmotifReportFolderReportFolderPath,nameSampleSaxBuilding)
        path_CosineBuilding = os.path.join(JmotifReportFolderReportFolderPath,nameCosineBuilding)
        TimeIndexBuilding = pd.read_csv(path_TimeIndexBuilding, index_col=0)
        SaxValuesBuilding= pd.read_csv(path_SampleSaxBuilding, index_col=0)
        CosineBuilding= pd.read_csv(path_CosineBuilding, index_col=0)
        name_tfidf= "tfidf.csv"
        path_tfidf = os.path.join(JmotifReportFolderReportFolderPath,name_tfidf)
        tfidf =  pd.read_csv(path_tfidf)
        #TimeIndexBuilding = TimeIndexBuilding.loc[:,1]
        
        time_index = pd.Series(TimeIndexBuilding).astype("int")
        words = pd.DataFrame({building:SaxValuesBuilding})
        emptyframe = pd.DataFrame(index=range(0,8760,1))
        orderedspec = pd.merge(words, tfidf, left_on=building, right_on='words', how='left')
        orderedspec.index = time_index
        orderedspec = pd.merge(orderedspec, emptyframe, right_index=True, left_index=True, how='outer')
        orderedspec = orderedspec.ffill(limit=w)
        
        inclass_specificity_building = orderedspec[buildinglist[1]] - orderedspec[buildinglist[2]] - orderedspec[buildinglist[3]] - orderedspec[buildinglist[4]] - orderedspec[buildinglist[5]]
        inclass_specificity_building = pd.DataFrame({building:inclass_specificity_building})
        inclass_specificity = pd.merge(inclass_specificity, inclass_specificity_building, right_index=True, left_index=True, how='outer')
            
        timezone = meta.T[building].timezone
        start = meta.T[building].datastart
        end = meta.T[building].dataend
        building_data_withindex = temp[building].tz_convert(timezone).truncate(before=start,after=end).tz_localize(None).ix[:8760]  
        
        inclass_specificity_building = inclass_specificity[building]
        inclass_specificity_building.index = building_data_withindex.index
        inclass_specificity_building_daily = pd.DataFrame({building:inclass_specificity_building.resample("D").mean()})
        inclass_specificity_toplot = pd.merge(inclass_specificity_toplot, inclass_specificity_building_daily, right_index=True, left_index=True, how='outer')
        CosineBuilding.index = CosineBuilding.classes
        feature_cosine_similarity[building] = CosineBuilding.cosines        


            