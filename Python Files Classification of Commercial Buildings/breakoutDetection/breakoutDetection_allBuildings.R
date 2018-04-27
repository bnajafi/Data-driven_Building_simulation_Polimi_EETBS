#install.packages("devtools")
#devtools::install_github("twitter/BreakoutDetection")
library(BreakoutDetection)
#. file directories

processedDatadirectory = 'C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/ProcessesData'
individualBuildingsFolder = "individualBuildings"
individualBuildingFolderPath =  file.path(processedDatadirectory,individualBuildingsFolder)
breakoutReportFolder = "BreakoutReport"
breakoutReportFolderPath =  file.path(processedDatadirectory,breakoutReportFolder)

# breakout detection arguments:
Minimum <- 30
Beta <- 0.005
Degrees <- 3



# first we  find the name of all buildings which are generated in this folder
listOfIndividualBuildingFiles= dir(path = individualBuildingFolderPath, pattern = "\\.csv$", full.names = FALSE, recursive = TRUE)

# Let's also extract the list of breakout detection reports which have been generated so far
listOfBreakOutDetectionReports= dir(path = breakoutReportFolderPath, pattern = "\\.csv$", full.names = FALSE, recursive = TRUE)


for(FileName in listOfIndividualBuildingFiles){
  BuildingName= sub('\\.csv$', '', FileName) 
  path_file = file.path(individualBuildingFolderPath,FileName)
  
  # let's first check if the report of this file has already been generated or not
  FileFormat = ".csv"
  locations_name= paste("Locations_", BuildingName, sep="")
  outputFileName = paste(locations_name,FileFormat, sep="")
  if(outputFileName %in% listOfBreakOutDetectionReports){
    print("the file already exists")
  }else{
    DF_Building = read.csv(path_file,header=FALSE,row.names=1)
    numericValues_DF_Building <- as.vector(DF_Building[[1]])

    res = breakout(numericValues_DF_Building, min.size=Minimum, method='multi', beta=Beta, degree=Degrees)
    BreakOutDectionMessage = paste("Building:",BuildingName, ", breakout Locations are:")
    print(BreakOutDectionMessage)
    print(res$loc)
    
    # now let's write these locations to a csv file
    locations_DF = data.frame(res$loc)
    names(locations_DF)<-locations_name # this simply changes the name of the column
    
    outputFilePath = file.path(breakoutReportFolderPath,outputFileName, sep="")
    breakoutWritingFileMessage = paste("I wrote them in the file", outputFileName)
    print(breakoutWritingFileMessage)
    print("--------------------------------")
    
    write.csv(locations_DF,file=outputFilePath)
    
  }
  
  
  
  
  
}





