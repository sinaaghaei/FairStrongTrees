# library(data.table)
# library(Publish)
# library(caret)
# library(sigmoid)
# library(rpart)
# library(dplyr)
library(multiplex)
rm(list=ls())

# #Here I read a csv file. You may need to look at the data and choose a proper seprator. Here the data is seperated using "," but sometimes the seperator is ";"
# data_csv <- read.csv("adult.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
# 
# 
# # Here I load a .data file. As I mentioned above we should be careful with the seprator. Here the fileds are sperated with space. So I'm using sep = ' '
# data_data<- read.csv("german.data", header = FALSE, sep = " ",na.strings = "",stringsAsFactors = TRUE)
# 
# 
# # So far we have read .csv and .data files and we have them as dataframes in R. Now we can save them in whatever format we want with the desired seperator.
# 
# #Here I save them as csv
# write.csv(data_csv,'csv_to_csv.csv',row.names = FALSE)
# write.csv(data_data,'data_to_csv.csv',row.names = FALSE)
# 
# 
# # Here I save them as .data
# write.dat(data_csv, 'address_folder')
# write.dat(data_data, 'address_folder')
# 
# 
# 
# data_data2<- read.csv("address_folder/data_data.data", header = FALSE, sep = " ",na.strings = "",stringsAsFactors = TRUE)

file_list <- list.files(path="../../DataSets/")
for(file in file_list){
  file_name = as.character(strsplit(file,'.csv')[1])
  data <- read.csv(file = paste('../../DataSets/',file,sep = ''), header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
  write.csv(data,paste(file_name,'.data',sep = ''),row.names = FALSE,quote = FALSE)
}
