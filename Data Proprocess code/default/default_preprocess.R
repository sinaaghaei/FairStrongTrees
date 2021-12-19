
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
setwd('/Users/sina/Documents/GitHub/FairStrongTrees/Data Proprocess code/default /')
data<- read.csv("default of credit card clients.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)


##########################################################################################################
# tidy preprocess
##########################################################################################################

data$ID <- NULL 
# This is the target column we are interested for classification
names(data)[names(data)=='default.payment.next.month'] = 'target'

#we can see that the repayment status is indicated in columns PAY_0, PAY_2 ... with no PAY_1 column, 
#so we rename PAY_0 to PAY_1 for ease of understanding.
names(data)[names(data)=='PAY_0'] = 'PAY_1'

#we get rid of pay_amt and bill_amt columns as there is high correlation between these cols and the rest of the cols
data <- dplyr::select(data, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_1, PAY_2, PAY_3,PAY_4,PAY_5,
                      PAY_6, target) 


# Categorize Age into 4 groups : <=30, 30-45, 45-60 and >60
data<- data %>% mutate(age_group = ifelse(AGE <=30, "<=30",
                                          ifelse(AGE>30 & AGE <=45, "30-45",
                                                 ifelse(AGE>45 & AGE <=60,"45-60",
                                                        ">60"))))
data$age_group <- factor(data$age_group, levels = c('<=30','30-45','45-60','>60'))
data$AGE <- NULL


# LIMIT_BAL
x='LIMIT_BAL'
data[[x]] = as.numeric(data[[x]])
data[[x]] = cut(data[[x]],
                c(-Inf,quantile(data[[x]],0.25),quantile(data[[x]],0.5),quantile(data[[x]],0.75),Inf),
                labels=c(1,2,3,4))

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
features = c('LIMIT_BAL','age_group')#,'PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'
for(v in features){
  for(i in seq(2,nlevels(data[[v]]),1)){
    a =  as.numeric(as.character(data_enc[[paste(v,toString(i),sep = ".")]]))
    b =  as.numeric(as.character(data_enc[[paste(v,toString(i-1),sep = ".")]]))
    data_enc[[paste(v,toString(i),sep = ".")]] =  as.numeric(a|b)
    
  }
}


rm(dmy)

setwd('/Users/sina/Documents/GitHub/FairStrongTrees/DataSets/')

write.csv(data,"default.csv",row.names = FALSE)
write.csv(data_enc,"default_enc.csv",row.names = FALSE)


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
  # table(data$sex, data$target)
  tmp <- data %>%
    mutate(index = row_number()) %>%
    group_by(SEX, target) %>%
    sample_frac(replace = FALSE, size = 0.75) %>%
    ungroup()
  
  tmp <- tmp %>%
    sample_n(replace = FALSE, size = 5000)
  
  
  train_ind <- tmp$index
  data_train <- data[train_ind, ]
  data_test <- data[-train_ind, ]
  
  data_train_enc <- data_enc[train_ind, ]
  data_test_enc <- data_enc[-train_ind, ]
  
  
  tmp <- data_train %>%
    mutate(index = row_number()) %>%
    group_by(SEX, target) %>%
    sample_frac(replace = FALSE, size = 2/3)
  
  
  
  train_calibration_ind <- tmp$index
  data_train_calibration <- data_train[train_calibration_ind, ]
  data_calibration<- data_train[-train_calibration_ind, ]
  
  data_train_calibration_enc <- data_train_enc[train_calibration_ind, ]
  data_calibration_enc <- data_train_enc[-train_calibration_ind, ]
  
  
  print('#############################')
  print(table(data_train_calibration$SEX, data_train_calibration$target))
  print(table(data_train$SEX, data_train$target))
  print(table(data_test$SEX, data_test$target))
  print(table(data_calibration$SEX, data_calibration$target))
  # Save files
  write.csv(data_train_enc,paste("default_train_enc_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_test_enc,paste("default_test_enc_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_train,paste("default_train_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_test,paste("default_test_",toString(Run),".csv",sep=''),row.names = FALSE)
  
  write.csv(data_train_calibration,paste("default_train_calibration_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_train_calibration_enc,paste("default_train_calibration_enc_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_calibration,paste("default_calibration_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_calibration_enc,paste("default_calibration_enc_",toString(Run),".csv",sep=''),row.names = FALSE)
}
