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
ListOfExistingFiles=  os.listdir(individualBuildingFolderPath)

breakouts = pd.DataFrame()
for building in DF_metaData.index:
    print building 
    timezone = DF_metaData.T[building].timezone
    start = DF_metaData.T[building].datastart
    end = DF_metaData.T[building].dataend
    building_data = DF_temporalData[building].tz_convert(timezone).truncate(before=start,after=end).resample('D').sum()
    data = building_data.reset_index(drop=True)
    BuildingFileName= building+".csv"
    if BuildingFileName in ListOfExistingFiles:
        print "This File Already Exists"
    else:
        savedFile_path = os.path.join(individualBuildingFolderPath,BuildingFileName)
        data.to_csv(savedFile_path)





# Now let's read the locations
name_locations_building = "Locations_"+building+".csv"
path_locations_building = os.path.join(processedData_repo,name_locations_building)
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

breakout.info()


DF_building = extract_building_data(DF_temporalData, DF_metaData, building)


def plot_line_example(df_1, df_2,  color):
    sns.set(rc={"figure.figsize": (12,4)})
    sns.set_style('whitegrid')
    fig = plt.figure()
    fig.autofmt_xdate()
    fig.subplots_adjust(hspace=.5)
    gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)
    
    df_1.columns = ["Actual kWh"]
    #df_predicted.columns = ["Predicted kWh"]
    ax1 = fig.add_subplot(gs[1:60,:])
    df_1.plot(ax = ax1, legend=False)
    ax1.xaxis.set_visible(False)
    ax1.set_title("Hourly kWh")
    
    ax2 = fig.add_subplot(gs[68:,:])
    #df_2 = df_2.tz_localize(None)
    x = mdates.drange(df_2.index[0], df_2.index[-1] + datetime.timedelta(days=1), datetime.timedelta(days=1))
    y = np.linspace(0, len(df_2.columns), len(df_2.columns)+1)
    data = np.array(df_2.T)
    datam = np.ma.array(data, mask=np.isnan(data))
    cmap = matplotlib.cm.get_cmap(color)
    
    qmesh = ax2.pcolormesh(x, y, datam, cmap=cmap)
    ax2.set_title("Number of breakouts indicating shifts in long-term steady state")
    
    #leftspacing, 
    cbaxes = fig.add_axes([0.18, 0.08, 0.7, 0.02]) 
    cbar = fig.colorbar(qmesh, ax=ax2, orientation='horizontal', cax=cbaxes)
    tick_locator = ticker.MaxNLocator(nbins=7)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
    ax2.axis('tight')
    ax2.xaxis_date()
    ax2.yaxis.set_visible(False)
    myFmt = mdates.DateFormatter('%b')
    ax2.xaxis.set_major_formatter(myFmt)
    
plot_line_example(DF_building, breakout, "Set1")