getwd()
setwd("C:/F/NMIMS/DataScience/Sem-3/Projects/LoanPrediction/data")
library(dplyr)
library(tidyr)
library(stringr)
library(caTools)
library(ISLR)
library(glmnet)
library(caret)
library(class)

loan_data <- read.csv("train_u6lujuX_CVtuZ9i.csv", stringsAsFactors = FALSE, header = TRUE)
head(loan_data)
tail(loan_data)
dim(loan_data)
summary(loan_data)
#Detect NAs
detectNAs<-function(x){
  return(sum(is.na(x)))
}
sapply(loan_data, detectNAs)

#Detect Type
detectType<-function(x){
  return(class(x))
}
sapply(loan_data, detectType)

#Detect Space
detectSpace <- function(x){
  if(class(x)=="character")
    return(sum(str_trim(x)==""))
  else
    return("Not a Character")
}
sapply(loan_data, detectSpace)



loan_data$LoanAmount[is.na(loan_data$LoanAmount)] <- median(loan_data$LoanAmount, na.rm=TRUE)

summarise(group_by(loan_data, Loan_Amount_Term), n())
loan_data$Loan_Amount_Term[is.na(loan_data$Loan_Amount_Term)] <- 360
loan_data$Loan_Amount_Term <- as.integer(loan_data$Loan_Amount_Term)




summarise(group_by(loan_data, Credit_History), n())
loan_data$Credit_History[is.na(loan_data$Credit_History)] <- 2


summarise(group_by(loan_data, Loan_Status), n())
loan_data$Loan_Status[loan_data$Loan_Status == "Y"] <- 1
loan_data$Loan_Status[loan_data$Loan_Status == "N"] <- 0

summarise(group_by(loan_data, Dependents), n())
loan_data$Dependents[str_trim(loan_data$Dependents)==""] <- "0"
loan_data$Dependents[loan_data$Dependents=="3+"] <- "3"
loan_data$Dependents <- as.numeric(loan_data$Dependents)

summarise(group_by(loan_data, Self_Employed), n())
loan_data$Self_Employed[is.na(loan_data$Self_Employed)] <- "No"
loan_data$Self_Employed[str_trim(loan_data$Self_Employed)==""] <- "No"
loan_data$Self_Employed[loan_data$Self_Employed == "Yes"] <- 1
loan_data$Self_Employed[loan_data$Self_Employed == "No"] <- 0

summarise(group_by(loan_data, Married), n())
loan_data$Married[str_trim(loan_data$Married)==""] <- "Yes"
loan_data$Married[loan_data$Married == "Yes"] <- 1
loan_data$Married[loan_data$Married == "No"] <- 0

summarise(group_by(loan_data, Gender), n())
loan_data$Gender[is.na(loan_data$Gender)] <- "Male"
loan_data$Gender[str_trim(loan_data$Gender)==""] <- "Male"
loan_data$Gender[loan_data$Gender == "Male"] <- 1
loan_data$Gender[loan_data$Gender == "Female"] <- 0

summarise(group_by(loan_data, Education), n())
loan_data$Education[loan_data$Education == "Graduate"] <- 1
loan_data$Education[loan_data$Education == "Not Graduate"] <- 0

summarise(group_by(loan_data, Property_Area), n())
loan_data$Property_Area[loan_data$Property_Area == "Rural"] <- 1
loan_data$Property_Area[loan_data$Property_Area == "Semiurban"] <- 2
loan_data$Property_Area[loan_data$Property_Area == "Urban"] <- 3

##Add Debt Ratio column
loan_data <- mutate(loan_data, Debt_Ratio=LoanAmount/(ApplicantIncome+CoapplicantIncome))

loan_data_dummy <- select(loan_data, -Loan_ID, -ApplicantIncome, -CoapplicantIncome, -LoanAmount)
View(loan_data_dummy)

loan_data_dummy$Gender <- as.factor(loan_data_dummy$Gender)
loan_data_dummy$Married <- as.factor(loan_data_dummy$Married)
loan_data_dummy$Education <- as.factor(loan_data_dummy$Education)
loan_data_dummy$Self_Employed <- as.factor(loan_data_dummy$Self_Employed)
loan_data_dummy$Loan_Status <- as.factor(loan_data_dummy$Loan_Status)
loan_data_dummy$Property_Area <- as.factor(loan_data_dummy$Property_Area)
loan_data_dummy$Credit_History <- as.integer(loan_data_dummy$Credit_History)
loan_data_dummy$Dependents <- as.factor(loan_data_dummy$Dependents)



str(loan_data_dummy)

cat("\014")
set.seed(100) # set seed to replicate results
split<-sample.split(loan_data_dummy$Loan_Status, SplitRatio=0.8)
trainSet<-subset(loan_data_dummy, split==TRUE)
testSet<-subset(loan_data_dummy, split==FALSE)


#Defining the training controls for multiple models
fitControl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = F)

#Defining the predictors and outcome
predictors<-c("Gender", "Married", "Dependents", "Education",
              "Self_Employed", "Loan_Amount_Term", "Credit_History", "Property_Area", "Debt_Ratio")
response_variable<-"Loan_Status"


### Fitting Random Forest
model_rf<-train(trainSet[,predictors],trainSet[,response_variable],method='rf',trControl=fitControl,tuneLength=3)

#Predicting using random forest model
pred_rf<-predict(model_rf, testSet[,predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testSet$Loan_Status,pred_rf)


### Fitting knn
model_knn<-train(trainSet[,predictors],trainSet[,response_variable],method='knn',trControl=fitControl,tuneLength=3)

#Predicting using random forest model
pred_knn<-predict(model_knn, testSet[,predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testSet$Loan_Status,pred_knn)


### Fitting Logistic
model_glm<-train(trainSet[,predictors],trainSet[,response_variable],method='glm',trControl=fitControl,tuneLength=3)

#Predicting using random forest model
pred_glm<-predict(model_glm, testSet[,predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testSet$Loan_Status,pred_glm)


#Predicting the probabilities
pred_rf_prob<-predict(model_rf,testSet[,predictors],type='prob')
pred_knn_prob<-predict(model_knn,testSet[,predictors],type='prob')
pred_glm_prob<-predict(model_glm,testSet[,predictors],type='prob')


#Taking weighted average of predictions
pred_weighted_avg<-(pred_rf_prob$`1`*0.3)+(pred_knn_prob$`1`*0.2)+(pred_glm_prob$`1`*0.5)

#Splitting into binary classes at 0.5
pred_weighted_avg<-as.factor(ifelse(pred_weighted_avg>0.5,'Y','N'))
pred_weighted_avg

testSet$Loan_Status <- as.character(testSet$Loan_Status)
testSet$Loan_Status[testSet$Loan_Status == "1"] <- 'Y'
testSet$Loan_Status[testSet$Loan_Status == "0"] <-'N'
table(testSet$Loan_Status,pred_weighted_avg)




######################## Now taking Test set ################################################
loan_testdata <- read.csv("test_Y3wMUE5_7gLdaTN.csv", stringsAsFactors = FALSE, header = TRUE)
head(loan_testdata)
tail(loan_testdata)
dim(loan_testdata)
summary(loan_testdata)
#Detect NAs
detectNAs<-function(x){
  return(sum(is.na(x)))
}
sapply(loan_testdata, detectNAs)

#Detect Type
detectType<-function(x){
  return(class(x))
}
sapply(loan_testdata, detectType)

#Detect Space
detectSpace <- function(x){
  if(class(x)=="character")
    return(sum(str_trim(x)==""))
  else
    return("Not a Character")
}
sapply(loan_testdata, detectSpace)



loan_testdata$LoanAmount[is.na(loan_testdata$LoanAmount)] <- median(loan_testdata$LoanAmount, na.rm=TRUE)

summarise(group_by(loan_testdata, Loan_Amount_Term), n())
loan_testdata$Loan_Amount_Term[is.na(loan_testdata$Loan_Amount_Term)] <- 360
loan_testdata$Loan_Amount_Term <- as.integer(loan_testdata$Loan_Amount_Term)


summarise(group_by(loan_testdata, Credit_History), n())
loan_testdata$Credit_History[is.na(loan_testdata$Credit_History)] <- 2


summarise(group_by(loan_testdata, Dependents), n())
loan_testdata$Dependents[str_trim(loan_testdata$Dependents)==""] <- "0"
loan_testdata$Dependents[loan_testdata$Dependents=="3+"] <- "3"
loan_testdata$Dependents <- as.numeric(loan_testdata$Dependents)

summarise(group_by(loan_testdata, Self_Employed), n())
loan_testdata$Self_Employed[is.na(loan_testdata$Self_Employed)] <- "No"
loan_testdata$Self_Employed[str_trim(loan_testdata$Self_Employed)==""] <- "No"
loan_testdata$Self_Employed[loan_testdata$Self_Employed == "Yes"] <- 1
loan_testdata$Self_Employed[loan_testdata$Self_Employed == "No"] <- 0

summarise(group_by(loan_testdata, Married), n())
loan_testdata$Married[str_trim(loan_testdata$Married)==""] <- "Yes"
loan_testdata$Married[loan_testdata$Married == "Yes"] <- 1
loan_testdata$Married[loan_testdata$Married == "No"] <- 0

summarise(group_by(loan_testdata, Gender), n())
loan_testdata$Gender[is.na(loan_testdata$Gender)] <- "Male"
loan_testdata$Gender[str_trim(loan_testdata$Gender)==""] <- "Male"
loan_testdata$Gender[loan_testdata$Gender == "Male"] <- 1
loan_testdata$Gender[loan_testdata$Gender == "Female"] <- 0

summarise(group_by(loan_testdata, Education), n())
loan_testdata$Education[loan_testdata$Education == "Graduate"] <- 1
loan_testdata$Education[loan_testdata$Education == "Not Graduate"] <- 0

summarise(group_by(loan_testdata, Property_Area), n())
loan_testdata$Property_Area[loan_testdata$Property_Area == "Rural"] <- 1
loan_testdata$Property_Area[loan_testdata$Property_Area == "Semiurban"] <- 2
loan_testdata$Property_Area[loan_testdata$Property_Area == "Urban"] <- 3

##Add Debt Ratio column
loan_testdata <- mutate(loan_testdata, Debt_Ratio=LoanAmount/(ApplicantIncome+CoapplicantIncome))

loan_testdata_dummy <- select(loan_testdata, -Loan_ID, -ApplicantIncome, -CoapplicantIncome, -LoanAmount)
View(loan_testdata_dummy)

loan_testdata_dummy$Gender <- as.factor(loan_testdata_dummy$Gender)
loan_testdata_dummy$Married <- as.factor(loan_testdata_dummy$Married)
loan_testdata_dummy$Education <- as.factor(loan_testdata_dummy$Education)
loan_testdata_dummy$Self_Employed <- as.factor(loan_testdata_dummy$Self_Employed)
loan_testdata_dummy$Property_Area <- as.factor(loan_testdata_dummy$Property_Area)
loan_testdata_dummy$Credit_History <- as.integer(loan_testdata_dummy$Credit_History)
loan_testdata_dummy$Dependents <- as.factor(loan_testdata_dummy$Dependents)

str(loan_testdata_dummy)



#Predicting the probabilities
pred_rf_testprob<-predict(model_rf,loan_testdata_dummy,type='prob')
pred_knn_testprob<-predict(model_knn,loan_testdata_dummy,type='prob')
pred_glm_testprob<-predict(model_glm,loan_testdata_dummy,type='prob')


#Taking weighted average of predictions
pred_testweighted_avg<-(pred_rf_testprob$`1`*0.3)+(pred_knn_testprob$`1`*0.2)+(pred_glm_testprob$`1`*0.5)

#Splitting into binary classes at 0.5
pred_testweighted_avg<-as.factor(ifelse(pred_testweighted_avg>0.5,'Y','N'))
pred_testweighted_avg


loanPred.data <- data.frame(Loan_ID=loan_testdata$Loan_ID, Loan_Status=as.character(pred_testweighted_avg))
write.table(loanPred.data, file = 'Sample_Submission_ZAuTl8O_FK3zQHh.csv', sep=",", row.names=FALSE,col.names=TRUE)




