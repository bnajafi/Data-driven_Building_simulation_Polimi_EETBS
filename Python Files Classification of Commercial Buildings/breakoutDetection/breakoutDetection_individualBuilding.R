#install.packages("devtools")
#devtools::install_github("twitter/BreakoutDetection")
library(BreakoutDetection)
processedDatadirectory = 'C:/Users/behzad/Dropbox/3 Research Projects/2 Data for Building/BuildingDataGenomeProject/the-building-data-genome-project/ProcessesData'
individualBuildingsFolder = "individualBuildings"
individualBuildingFolderPath =  file.path(processedDatadirectory,individualBuildingsFolder)

building  = "UnivDorm_Cooper"
FileFormat= ".csv"
name_file= paste(building, FileFormat, sep="")
path_file = file.path(individualBuildingFolderPath,name_file)

DF_Building = read.csv(path_file,header=FALSE,row.names=1)
numericValues_DF_Building <- as.vector(DF_Building[[1]])
class(data)
Minimum <- 30
Beta <- 0.005
Degrees <- 3
res = breakout(avector, min.size=Minimum, method='multi', beta=Beta, degree=Degrees)
res$loc


# now let's write these locations to a csv file

locations_DF = data.frame(res$loc)
FileFormat = ".csv"
locations_name= paste("Locations_", building, sep="")
names(locations_DF)<-locations_name # this simply changes the name of the column

outputFileName = paste(locations_name,FileFormat, sep="")

breakoutReportFolder = "BreakoutReport"
breakoutReportFolderPath =  file.path(processedDatadirectory,breakoutReportFolder)
outputFilePath = file.path(breakoutReportFolderPath,outputFileName, sep="")

write.csv(locations_DF,file=outputFilePath)

