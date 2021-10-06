library(data.table)
library(Publish)
library(caret)
library(sigmoid)
library(rpart)
library(dplyr)

rm(list=ls())
graphics.off()



dataset_groups = list('compas'='race', 'german'='age')
data_types = c('train','train_calibration','test','calibration')
splits = c(1,2,3,4,5)

for(data_group in names(dataset_groups)){
  for(type in data_types){
    for(split in splits){
      setwd('/Users/sina/Documents/GitHub/FairStrongTrees/DataSets/')
      protected_col = as.character(dataset_groups[data_group])
      data_name = paste(data_group,type,split,sep = '_')
      data_enc_name = paste(data_group,type,'enc',split,sep = '_')
      data <- read.csv(paste(data_name,'.csv',sep=''), header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
      data_enc <- read.csv(paste(data_enc_name,'.csv',sep=''), header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
      
      data_enc <- data_enc[,!grepl( protected_col, names(data_enc), fixed = TRUE)]
      data_enc[[protected_col]] = data[[protected_col]]
      
      setwd('/Users/sina/Documents/GitHub/FairStrongTrees/DataSets/Kamiran Version/')
      new_name = paste(data_group,type,'Kamiran',split,sep = '_')
      new_name = paste(new_name,'.csv',sep='')
      write.csv(data_enc,new_name,row.names = FALSE)
    }
  }
}





