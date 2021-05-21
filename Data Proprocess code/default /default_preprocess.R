
library(caret)

library(stringr)
library(outliers)
library(editrules)
library(dplyr)

rm(list=ls())
graphics.off()

#################################################################################################
#Functions
#################################################################################################
dataencoder <- function (data) {
  #encoding data
  must_convert<-sapply(data,is.factor)       # logical vector telling if a variable needs to be displayed as numeric
  M2<-sapply(data[,must_convert],unclass)    # data.frame of all categorical variables now displayed as numeric
  data_num<-cbind(data[,!must_convert],M2)
  data_num <- as.data.frame(data_num)
  
  for(tmp_f in names(data)){
    data_num[[tmp_f]] = as.factor(data_num[[tmp_f]] )
    data_num[[tmp_f]]  = droplevels(data_num[[tmp_f]] )
  }
  
  data_num
}

##########################################################################################################
# read data 
##########################################################################################################
data<- read.csv("default of credit card clients.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)


##########################################################################################################
# tidy preprocess
##########################################################################################################

data$ID <- NULL 
names(data)[names(data)=='default.payment.next.month'] = 'target'



##########################################################################################################
# Feature selection
##########################################################################################################
