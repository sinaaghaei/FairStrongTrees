library(data.table)
library(Publish)
library(caret)
library(sigmoid)
library(rpart)
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
# data_raw <- read.csv("compas-analysis-master/compas-scores-raw.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
# data_v <- read.csv("compas-analysis-master/compas-scores-two-years-violent.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
# data_compas <- read.csv("compas-analysis-master/compas-scores.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
data <- read.csv("compas-analysis-master/compas-scores-two-years.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)



data <- dplyr::select(data, race, age_cat, sex,priors_count, c_charge_degree, c_jail_in, c_jail_out, days_b_screening_arrest, 
                    decile_score, score_text, is_recid, two_year_recid) %>% 
  filter(days_b_screening_arrest <= 30) %>%
  filter(days_b_screening_arrest >= -30) %>%
  filter(is_recid != -1) %>%
  filter(c_charge_degree != "O") %>%
  filter(score_text != 'N/A')


data$length_of_stay <- as.numeric(as.Date(data$c_jail_out) - as.Date(data$c_jail_in))

data <- dplyr::select(data, race, age_cat, sex,priors_count, c_charge_degree,length_of_stay, 
                      score_text, two_year_recid)

names(data)[names(data)=="two_year_recid"] = "target"

data$age_cat <- factor(data$age_cat, levels = c('Less than 25','25 - 45','Greater than 45'))
data$score_text <- factor(data$score_text, levels = c('Low','Medium','High'))




# we partition prior convictions into
#four bins: 0, 1–2, 3–4, and 5 or more.
# see: https://arxiv.org/pdf/1701.08230.pdf
data$priors_count = as.numeric(data$priors_count)
data$priors_count = cut(data$priors_count ,
                    c(-Inf,0,2,4,Inf),
                    labels=c(1,2,3,4))

#5 bins: 0, 1, 2–7, 8-15,and 16 or more.
data$length_of_stay = cut(data$length_of_stay ,
                        c(-Inf,0,1,7,15,Inf),
                        labels=c(1,2,3,4,5))

index <- !(data$race %in% c('African-American','Caucasian','Hispanic'))
data$race[index] <- 'Other'

for(f in names(data)){
  data[[f]] = as.factor(data[[f]])
  data[[f]] = droplevels(data[[f]])
}
##########################################################################################################
# encoding data
##########################################################################################################
data <- dataencoder(data)

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
features = c('age_cat','priors_count','length_of_stay','score_text')
for(v in features){
  for(i in seq(2,nlevels(data[[v]]),1)){
    a =  as.numeric(as.character(data_enc[[paste(v,toString(i),sep = ".")]]))
    b =  as.numeric(as.character(data_enc[[paste(v,toString(i-1),sep = ".")]]))
    data_enc[[paste(v,toString(i),sep = ".")]] =  as.numeric(a|b)
    
  }
}

# 
write.csv(data,"compas.csv",row.names = FALSE)
write.csv(data_enc,"compas_enc.csv",row.names = FALSE)
