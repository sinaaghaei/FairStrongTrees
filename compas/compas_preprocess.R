# library(data.table)
# library(Publish)
# library(caret)
# library(sigmoid)
# library(rpart)

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
data <- read.csv("compas-scores-two-years.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)

data <- dplyr::select(data, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, 
                    days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out) %>% 
  filter(days_b_screening_arrest <= 30) %>%
  filter(days_b_screening_arrest >= -30) %>%
  filter(is_recid != -1) %>%
  filter(c_charge_degree != "O") %>%
  filter(score_text != 'N/A')

data <- mutate(data, crime_factor = factor(c_charge_degree)) %>%
  mutate(age_factor = as.factor(age_cat)) %>%
  within(age_factor <- relevel(age_factor, ref = 1)) %>%
  mutate(race_factor = factor(race)) %>%
  within(race_factor <- relevel(race_factor, ref = 3)) %>%
  mutate(gender_factor = factor(sex, labels= c("Female","Male"))) %>%
  within(gender_factor <- relevel(gender_factor, ref = 2)) %>%
  mutate(score_factor = factor(score_text != "Low", labels = c("LowScore","HighScore")))


data <- data[,c("priors_count","gender_factor" ,"age_factor" ,"race_factor",
               "crime_factor" ,"two_year_recid","score_factor")]

names(data)[names(data)=="score_factor"] = "target"

original_data = data
##########################################################################################################
# encoding data
##########################################################################################################
data <- dataencoder(data)

data_enc = data
x = "priors_count"
data_enc[[x]] = as.numeric(data_enc[[x]])
data_enc[[x]] = cut(data_enc[[x]],
                    c(-Inf,5,10,15,20,25,30,35,Inf),
                    labels=c(1,2,3,4,5,6,7,8))
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
features = c("priors_count")
for(v in features){
  for(i in seq(2,8,1)){
    a =  as.numeric(as.character(data_enc[[paste(v,toString(i),sep = ".")]]))
    b =  as.numeric(as.character(data_enc[[paste(v,toString(i-1),sep = ".")]]))
    data_enc[[paste(v,toString(i),sep = ".")]] =  as.numeric(a|b)
    
  }
}


write.csv(data,"compas.csv",row.names = FALSE)
write.csv(data_enc,"compas_enc.csv",row.names = FALSE)
