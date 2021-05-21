
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
data_1 <- read.csv("adult.data", header = FALSE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
data_2 <- read.csv("adult.test", header = FALSE, sep = ",",na.strings = "",stringsAsFactors = TRUE)

data <- rbind(data_1,data_2)
rm(data_1,data_2)


names(data) <- c('age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship',
                 'race','sex','capital_gain','capital_loss','hours_per_week','native_country','target')

# Let's replace ? with NA and omit them from the dataset
data[data==' ?'] = NA
data <- na.omit(data)
##########################################################################################################
# tidy preprocess
##########################################################################################################

# Check relationship between education and education.num
data %>% distinct(education, education_num) 

# drop education.num variable
data$education_num <- NULL

# Create capital variable which is the difference betwwen capital-gain and capital-loss
data <- data %>% mutate(capital = capital_gain - capital_loss)


# List down Factor Columns in dataframe & Trim string in factor columns
fac_cols <- sapply(data, is.factor)
data <- data.frame(cbind(sapply(data[,fac_cols], trimws, which="both"), data[,!fac_cols]))


# Clean “Workclass” variable by categorizing it into 4 categories: Gov, Self-emp, Private and Other
data <- data %>% mutate(workclass = ifelse(grepl(".gov$", str_trim(workclass)), "Gov", 
                                             ifelse(grepl("^Self.",str_trim(workclass)),"Self-emp",
                                                    ifelse(grepl("^Private$", str_trim(workclass)),"Private", "Other"))))
data$workclass <- as.factor(data$workclass)
levels(data$workclass)




# Clean “Education” variable by categorizing it into groups: Before-Highschool, Associate, Post-graduate, HS-grad, Some-college and Bachelors
data <- data %>% mutate(education = ifelse(grepl(".th$|^Preschool$", (education)), "Before-Highschool",
                                             ifelse(grepl("^Assoc.", (education)),"Associate",
                                                    ifelse(grepl("^Masters$|^Doctorate$|^Pro.",(education)), "Post-Graduate", 
                                                           as.character((education))))))
data$education <- as.factor(data$education)
levels(data$education)



# Clean “Marital Status” variable
data <- data %>% mutate(marital_status = ifelse(grepl("^Married.", marital_status), "Married", as.character(marital_status)))
data$marital_status <- as.factor(data$marital_status)
levels(data$marital_status)


#Clean “Income variable”
data <- data %>% mutate(target = ifelse(grepl("^<=50K.$", target), "<=50K",
                                          ifelse(grepl("^>50K.$", target),">50K", as.character(target))))
data$target <- as.factor(data$target)
levels(data$target)


# Categorize Age into 4 groups : <=30, 30-45, 45-60 and >60
# Convert Age character into numeric because Age has character type as default in dataset.
data$age <- as.integer(data$age)
# Categorize Age into 4 groups
data<- data %>% mutate(age_group = ifelse(age <=30, "<=30",
                                            ifelse(age>30 & age <=45, "30-45",
                                                   ifelse(age>45 & age <=60,"45-60",
                                                          ">60"))))
data$age_group <- factor(data$age_group, levels = c('<=30','30-45','45-60','>60'))
# Check levels result of Age after processing
levels(data$age_group)


#Clean Native Countries variable by categorizing it into two groups : US and Non-US
data<- data %>% mutate(native_country = ifelse(grepl("United.",native_country), "USA", "Non-USA"))
data$native_country <- as.factor(data$native_country)
levels(data$native_country)


# The quantiles of hours.per.weak are not unique; Here we divide people into following categories
data<- data %>% mutate(hours_per_week = ifelse(hours_per_week <=20, "<=20",
                                          ifelse(hours_per_week>20 & hours_per_week <=40, "20-40",
                                                 ifelse(hours_per_week>40 & hours_per_week <=60,"40-60",
                                                        ">60"))))

data$hours_per_week <- factor(data$hours_per_week, levels = c('<=20','20-40','40-60','>60'))


# fnlwgt
x='fnlwgt'
data[[x]] = as.numeric(data[[x]])
data[[x]] = cut(data[[x]],
                    c(-Inf,quantile(data[[x]],0.25),quantile(data[[x]],0.5),quantile(data[[x]],0.75),Inf),
                    labels=c(1,2,3,4))



# #Capital
summary(data$capital)
nrow(subset(data,data$capital>0))/nrow(data)
nrow(subset(data,data$capital<0))/nrow(data)
nrow(subset(data,data$capital==0))/nrow(data)
data<- data %>% mutate(capital = ifelse(capital < 0 , "<0",
                                               ifelse(capital>0 , ">0",
                                                      '=0')))



data$age <- NULL
data$capital_gain <- NULL
data$capital_loss <- NULL



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
features = c('fnlwgt','hours_per_week','capital','age_group')
for(v in features){
  for(i in seq(2,nlevels(data[[v]]),1)){
    a =  as.numeric(as.character(data_enc[[paste(v,toString(i),sep = ".")]]))
    b =  as.numeric(as.character(data_enc[[paste(v,toString(i-1),sep = ".")]]))
    data_enc[[paste(v,toString(i),sep = ".")]] =  as.numeric(a|b)
    
  }
}


write.csv(data,"adult.csv",row.names = FALSE)
write.csv(data_enc,"adult_enc.csv",row.names = FALSE)


# Choose the seeds
seeds = c(123,156,67,1,43)
for(Run in c(1,2,3,4,5)){
  set.seed(seeds[Run])
  ##########################################################################################################
  # Splitting data into training and test 
  ##########################################################################################################
  smp_size = 8000
  
  ## set the seed to make your partition reproducible
  sample_ind <- sample(seq_len(nrow(data)), size = smp_size)
  
  data_sample <- data[sample_ind, ]
  data_sample_enc <- data_enc[sample_ind, ]
  
  
  # Save files
  write.csv(data_sample,paste("adult_sample_",toString(Run),".csv",sep=''),row.names = FALSE)
  write.csv(data_sample_enc,paste("adult_sample_enc_",toString(Run),".csv",sep=''),row.names = FALSE)
  
}



