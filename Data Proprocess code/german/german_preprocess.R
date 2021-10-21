
library(caret)

library(stringr)
library(outliers)
library(editrules)
library(dplyr)

rm(list=ls())
graphics.off()

'
The analyzer can analyze some data collected by a bank giving a loan. 
The dataset consists of 1000 datapoints of categorical and numerical 
dataas well as a good credit vs bad credit metric which has been assigned by bank employees.
'


Kamiran_version = TRUE
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
setwd('/Users/sina/Documents/GitHub/FairStrongTrees/Data Proprocess code/german/')
data<- read.csv("../german/german.data", header = FALSE, sep = " ",na.strings = "",stringsAsFactors = TRUE)

names(data) <- c("chek_acc","month_duration","credit_history","purpose","Credit_amo","saving_amo","present_employmment",
                 "instalrate","p_status","guatan","present_resident","property","age","installment","Housing",
                 "existing_cards","job","no_people","telephn","foreign_worker","target")

'Right now status = 1 means good and 2 means bad. I want them to be 0 and 1 and 1 
represent the positive outcome. So we change them as follows'
data$target <- abs(data$target - 2)

##########################################################################################################
# tidy preprocess
##########################################################################################################


numeric_features = c('month_duration','Credit_amo','instalrate','present_resident')
for(x in numeric_features){
  data[[x]] = as.numeric(data[[x]])
  data[[x]] = cut(data[[x]],
                  c(-Inf,quantile(data[[x]],0.25),quantile(data[[x]],0.5),quantile(data[[x]],0.75),Inf),
                  labels=c(1,2,3,4))
}

# Categorize Age into 4 groups : <=30, 30-45, 45-60 and >60
if(Kamiran_version){
  data<- data %>% mutate(age = ifelse(age <=25, "<=25",'>25'))
  data$age <- factor(data$age, levels = c('<=25','>25'))
  
}else{
  data<- data %>% mutate(age = ifelse(age <=30, "<=30",
                                      ifelse(age>30 & age <=45, "30-45",
                                             ifelse(age>45 & age <=60,"45-60",
                                                    ">60"))))
  data$age <- factor(data$age, levels = c('<=30','30-45','45-60','>60'))
}

# Check levels result of Age after processing
summary(data$age)



for(f in names(data)){
  data[[f]] = as.factor(data[[f]])
}


##########################################################################################################
# One hot encoded data
##########################################################################################################
data<- dataencoder(data)

data_enc = data

#Now we tuurn all categorical  features into one-hot vectors
dmy <- dummyVars(" ~ .-target", data = data_enc)
data_enc <- data.frame(predict(dmy, newdata = data_enc))

#if a feature has only two levels we should only keep one column
#As our convention, we always keep the first one
cols = c()
tmp <- gsub("\\..*","",names( data_enc ))
for(name in names(data)){
  # a = grepl( name , tmp ,fixed=TRUE)
  a = tmp == name
  if(sum(a)==2){
    cols <- append(cols, min(which(a == TRUE)))
  }else{
    cols <- append(cols, which(a == TRUE))
  }
}

data_enc <- data_enc[,cols]
data_enc$target <- data$target


# Taking care of  the integer columns : If x_ij = 1 then x_i(j+1) should be one as well  for odd i's
if(Kamiran_version){
  features = c('month_duration','Credit_amo','instalrate','present_resident','existing_cards')
}else{
  features = c('month_duration','Credit_amo','instalrate','present_resident','age','existing_cards')
}

for(v in features){
  for(i in seq(2,nlevels(data[[v]]),1)){
    a =  as.numeric(as.character(data_enc[[paste(v,toString(i),sep = ".")]]))
    b =  as.numeric(as.character(data_enc[[paste(v,toString(i-1),sep = ".")]]))
    data_enc[[paste(v,toString(i),sep = ".")]] =  as.numeric(a|b)
    
  }
}

if(Kamiran_version){
  setwd('/Users/sina/Documents/GitHub/FairStrongTrees/DataSets/Kamiran Version/')
}else{
  setwd('/Users/sina/Documents/GitHub/FairStrongTrees/DataSets/')
}


write.csv(data,"german.csv",row.names = FALSE)
write.csv(data_enc,"german_enc.csv",row.names = FALSE)



##########################################################################################################
# Sampling from data
##########################################################################################################
seeds = c(123,156,67,1,43)


for(Run in c(1,2,3,4,5)){
  ## set the seed to make your partition reproducible
  set.seed(seeds[Run])
  ##########################################################################################################
  # Splitting data into training and test
  ##########################################################################################################
  tmp <- data %>%
    mutate(index = row_number()) %>%
    group_by(age, credit_history, target) %>%
    sample_frac(replace = FALSE, size = 0.75)
  
  
  train_ind <- tmp$index
  data_train <- data[train_ind, ]
  data_test <- data[-train_ind, ]
  
  data_train_enc <- data_enc[train_ind, ]
  data_test_enc <- data_enc[-train_ind, ]
  
  tmp <- data_train %>%
    mutate(index = row_number()) %>%
    group_by(age, credit_history, target) %>%
    sample_frac(replace = FALSE, size = 2/3)
  
  train_calibration_ind <- tmp$index
  data_train_calibration <- data_train[train_calibration_ind, ]
  data_calibration<- data_train[-train_calibration_ind, ]
  
  data_train_calibration_enc <- data_train_enc[train_calibration_ind, ]
  data_calibration_enc <- data_train_enc[-train_calibration_ind, ]
  
  # Save files
  write.csv(data_train_enc,paste("german_train_enc_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_test_enc,paste("german_test_enc_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_train,paste("german_train_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_test,paste("german_test_",toString(Run),".csv",sep=''),row.names = FALSE)
  
  write.csv(data_train_calibration,paste("german_train_calibration_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_train_calibration_enc,paste("german_train_calibration_enc_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_calibration,paste("german_calibration_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_calibration_enc,paste("german_calibration_enc_",toString(Run),".csv",sep=''),row.names = FALSE)
}




# # Test
data_train<- read.csv("/Users/sina/Documents/GitHub/FairStrongTrees/DataSets/german_calibration_1.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
data_train_kamiran <- read.csv("/Users/sina/Documents/GitHub/FairStrongTrees/DataSets/Kamiran Version/german_calibration_1.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
# data_test<- read.csv("/Users/sina/Documents/GitHub/FairStrongTrees/DataSets/german_test_1.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
