#### IPHONE 

################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("mlbench")
install.packages("tidyverse")
install.packages("doParallel")
install.packages("plotly")
install.packages ("e1071")
library(e1071)
library(caret)
library(gbm)
library(kknn)
library(corrplot)
library(readr)
library(mlbench)
library(tidyverse)
library(modelr)
library(doParallel) 
library(plotly)


#####################
# Parallel processing
#####################

detectCores()          
cl <- makeCluster(2)   
registerDoParallel(cl) 
getDoParWorkers()      

##############
# Import data 
##############

setwd('C:/Users/santi/OneDrive/Documentos/CURSO DATA ANALYTICS/Cours V/Task 3/Data/smallmatrix_labeled_8d')

iphoneDF <- read.csv("iphone_smallmatrix_labeled_8d.csv", stringsAsFactors = FALSE)


################
# Evaluate data
################

head(iphoneDF)
tail(iphoneDF)
str(iphoneDF)
summary(iphoneDF)

plot_ly(iphoneDF, x= ~iphoneDF$iphonesentiment, type='histogram')

#############
# Preprocess
#############

anyNA(iphoneDF)
anyDuplicated(iphoneDF)
iphoneDF$iphonesentiment <- as.factor(iphoneDF$iphonesentiment)

#####################
# EDA/Visualizations
#####################


fig <- plot_ly(iphoneDF, x= ~iphoneDF$iphonesentiment, type='histogram', color = ~iphonesentiment, colors = "Set2")
fig <- fig %>% layout(title = "iPhone Sentiment",
                      xaxis = list(title = "Sentiment"),
                      yaxis = list(title = "Count"))
fig

iphoneDF %>%
  ggplot( aes(x=iphonesentiment)) +
  geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8)

hist(iphoneDF$iphone)
hist(iphoneDF$ios)
hist(iphoneDF$iphonecampos)
hist(iphoneDF$iphonecamneg)
hist(iphoneDF$iphonecamunc)
hist(iphoneDF$iphonedispos)
hist(iphoneDF$iphonedisneg)
hist(iphoneDF$iphonedisunc)
hist(iphoneDF$iphoneperpos)
hist(iphoneDF$iphoneperneg)
hist(iphoneDF$iphoneperunc)
hist(iphoneDF$iosperpos)
hist(iphoneDF$iosperneg)
hist(iphoneDF$iosperunc)


#######################
# Feature selection
#######################

# ---- Correlation analysis ---- #

corrAll <- cor(iphoneDF)
corrAll
corrplot(corrAll, method = "circle")
corrplot(corrAll, order = "hclust") 

# there are no characteristics highly correlated with the dependent

# ---- Examine Feature Variance ---- #

nzvMetrics <- nearZeroVar(iphoneDF, saveMetrics = TRUE)
nzvMetrics

nzv <- nearZeroVar(iphoneDF, saveMetrics = FALSE) 
nzv

iphoneNZV <- iphoneDF[,-nzv]
str(iphoneNZV)

# ---- Recursive Feature Elimination  ---- #

# Let's sample the data before using RFE
set.seed(123)
iphoneSample <- iphoneDF[sample(1:nrow(iphoneDF), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 2,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults

# Plot results
plot(rfeResults, type=c("g", "o"))

# create new data set with rfe recommended features
iphoneRFE <- iphoneDF[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphoneDF$iphonesentiment

# review outcome
str(iphoneRFE)


##################
# Train/test sets
##################

# -------- iphoneDF -------- #

set.seed(123) 
inTrainingDF <- createDataPartition(iphoneDF$iphonesentiment, p=0.75, list=FALSE)
oobTrainDF <- iphoneDF[inTrainingDF,]   
oobTestDF <- iphoneDF[-inTrainingDF,]   
# verify number of obs 
nrow(oobTrainDF) 
nrow(oobTestDF)  

# -------- iphoneNZV -------- #

set.seed(123) 
inTrainingNZV <- createDataPartition(iphoneNZV$iphonesentiment, p=0.75, list=FALSE)
oobTrainNZV <- iphoneNZV[inTrainingNZV,]   
oobTestNZV <- iphoneNZV[-inTrainingNZV,]   
# verify number of obs 
nrow(oobTrainNZV) 
nrow(oobTestNZV)  

# -------- iphoneRFE -------- #

set.seed(123) 
inTrainingRFE <- createDataPartition(iphoneRFE$iphonesentiment, p=0.75, list=FALSE)
oobTrainRFE <- iphoneRFE[inTrainingRFE,]   
oobTestRFE <- iphoneRFE[-inTrainingRFE,]   
# verify number of obs 
nrow(oobTrainRFE) 
nrow(oobTestRFE)  

################
# Train control
################

# set cross validation
fitControl <- trainControl(method="repeatedcv", number=5, repeats=1) 

###############
# Train models
###############

## ------- SVM ------- ##

### iphoneDF

set.seed(99)
oobSVMfit <- train(iphonesentiment ~., data = oobTrainDF, method = "svmLinear2", trControl = fitControl,  preProcess = c("center","scale"))
oobSVMfit
varImp(oobSVMfit)

#Support Vector Machines with Linear Kernel 

#9732 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#Pre-processing: centered (58), scaled (58) 
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7785, 7785, 7786, 7786, 7786 
#Resampling results across tuning parameters:
  
#  cost  Accuracy   Kappa    
#0.25  0.7052001  0.3959895
#0.50  0.7056106  0.4033636
#1.00  0.7088982  0.4110912

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cost = 1.


### iphoneNZV

set.seed(99)
oobSVMfit1 <- train(iphonesentiment~., data=oobTrainNZV, method="svmLinear2", trControl=fitControl, verbose=FALSE, scale = FALSE)
oobSVMfit1
varImp(oobSVMfit1)

#Support Vector Machines with Linear Kernel 

#9732 samples
#11 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7785, 7785, 7786, 7786, 7786 
#Resampling results across tuning parameters:
  
#  cost  Accuracy   Kappa    
#0.25  0.6859847  0.3508510
#0.50  0.6856764  0.3502780
#1.00  0.6852654  0.3492985

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cost = 0.25.


### iphoneRFE

set.seed(123)
oobSVMfit2 <- train(iphonesentiment~., data=oobTrainRFE, method="svmLinear2", trControl=fitControl, verbose=FALSE,scale = FALSE)
oobSVMfit2
varImp(oobSVMfit2)

#Support Vector Machines with Linear Kernel 

#9732 samples
#10 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7786, 7786, 7785, 7787, 7784 
#Resampling results across tuning parameters:
  
#  cost  Accuracy   Kappa    
#0.25  0.6964680  0.3819263
#0.50  0.6965707  0.3821230
#1.00  0.6967761  0.3818696

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cost = 1.


## ------- RF ------- ##

### iphoneDF

set.seed(123)
oobRFfit <- train(iphonesentiment~., data=oobTrainDF, method="rf", importance=T, trControl=fitControl)

# manual grid
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfit <- train(iphonesentiment~.,data=oobTrainDF ,method="rf",
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfGrid)
oobRFfit
plot(oobRFfit)
varImp(oobRFfit)

#Random Forest 

#9732 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7786, 7786, 7785, 7787, 7784 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#4     0.7292449  0.4491579
#5     0.7486670  0.4968628
#6     0.7537018  0.5095732

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 6.


### iphoneNZV

set.seed(123)
oobRFfit1 <- train(iphonesentiment~., data=oobTrainNZV, method="rf", importance=T, trControl=fitControl)

# manual grid
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfit1 <- train(iphonesentiment~.,data=oobTrainNZV,method="rf",
                   importance=T,
                   trControl=fitControl,
                   tuneGrid=rfGrid)
oobRFfit1
plot(oobRFfit1)
varImp(oobRFfit1)

#Random Forest 

#9732 samples
#11 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7786, 7786, 7785, 7787, 7784 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#4     0.7606876  0.5325235
#5     0.7591463  0.5305944
#6     0.7565769  0.5275517

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 4.


### iphoneRFE

set.seed(123)
oobRFfit2 <- train(iphonesentiment~., data=oobTrainRFE, method="rf", importance=T, trControl=fitControl)

# manual grid
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfit2 <- train(iphonesentiment~.,data=oobTrainRFE,method="rf",
                   importance=T,
                   trControl=fitControl,
                   tuneGrid=rfGrid)
oobRFfit2
plot(oobRFfit2)
varImp(oobRFfit2)

#Random Forest 

#9732 samples
#10 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7786, 7786, 7785, 7787, 7784 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#4     0.7711683  0.5587171
#5     0.7702441  0.5583105
#6     0.7693194  0.5576198

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 4.


## ------- C5.0 ------- ##

### iphoneDF

set.seed(99)
oobC5fit <- train(iphonesentiment~., data=oobTrainDF, method="C5.0", trControl=fitControl, verbose=FALSE)
oobC5fit
varImp(oobC5fit)

#C5.0 

#9732 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7785, 7785, 7786, 7786, 7786 
#Resampling results across tuning parameters:
  
#  model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7736342  0.5605223
#rules  FALSE   10      0.7595572  0.5366126
#rules  FALSE   20      0.7595572  0.5366126
#rules   TRUE    1      0.7733257  0.5597613
#rules   TRUE   10      0.7618171  0.5409730
#rules   TRUE   20      0.7618171  0.5409730
#tree   FALSE    1      0.7712709  0.5574255
#tree   FALSE   10      0.7616129  0.5436569
#tree   FALSE   20      0.7616129  0.5436569
#tree    TRUE    1      0.7708599  0.5561921
#tree    TRUE   10      0.7604815  0.5397567
#tree    TRUE   20      0.7604815  0.5397567

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = rules and winnow = FALSE.

### iphoneNZV 

set.seed(99)
oobC5fit1 <- train(iphonesentiment~., data=oobTrainNZV, method="C5.0", trControl=fitControl, verbose=FALSE)
oobC5fit1
varImp(oobC5fit1)

#C5.0 

#9732 samples
#11 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7785, 7785, 7786, 7786, 7786 
#Resampling results across tuning parameters:
  
#  model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7584270  0.5256078
#rules  FALSE   10      0.7450697  0.5017676
#rules  FALSE   20      0.7450697  0.5017676
#rules   TRUE    1      0.7587353  0.5270051
#rules   TRUE   10      0.7436301  0.4974391
#rules   TRUE   20      0.7436301  0.4974391
#tree   FALSE    1      0.7566803  0.5234355
#tree   FALSE   10      0.7412669  0.4974059
#tree   FALSE   20      0.7412669  0.4974059
#tree    TRUE    1      0.7570914  0.5250053
#tree    TRUE   10      0.7412668  0.4950149
#tree    TRUE   20      0.7412668  0.4950149

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = rules and winnow = TRUE.

### iphoneRFE

set.seed(99)
oobC5fit2 <- train(iphonesentiment~., data=oobTrainRFE, method="C5.0", trControl=fitControl, verbose=FALSE)
oobC5fit2
varImp(oobC5fit2)

#C5.0 

#9732 samples
#10 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7785, 7785, 7786, 7786, 7786 
#Resampling results across tuning parameters:
  
#  model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7673659  0.5505086
#rules  FALSE   10      0.7587346  0.5352310
#rules  FALSE   20      0.7587346  0.5352310
#rules   TRUE    1      0.7673659  0.5505086
#rules   TRUE   10      0.7587346  0.5352310
#rules   TRUE   20      0.7587346  0.5352310
#tree   FALSE    1      0.7697292  0.5564853
#tree   FALSE   10      0.7565776  0.5306052
#tree   FALSE   20      0.7565776  0.5306052
#tree    TRUE    1      0.7697292  0.5564853
#tree    TRUE   10      0.7565776  0.5306052
#tree    TRUE   20      0.7565776  0.5306052

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = tree and winnow = TRUE.

## ------- KKNN ------- ##

### iphoneDF

set.seed(99)
oobKNfit <- train(iphonesentiment~., data=oobTrainDF, method="kknn", trControl=fitControl, verbose=FALSE)
oobKNfit
varImp(oobKNfit)

#k-Nearest Neighbors 

#9732 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7785, 7785, 7786, 7786, 7786 
#Resampling results across tuning parameters:
  
#  kmax  Accuracy   Kappa    
#5     0.3115493  0.1550916
#7     0.3265513  0.1627754
#9     0.3332303  0.1654273

#Tuning parameter 'distance' was held constant at a value of 2
#Tuning parameter 'kernel' was held constant at a value of optimal
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were kmax = 9, distance = 2 and kernel = optimal.

### iphoneNZV 

set.seed(99)
oobKNfit1 <- train(iphonesentiment~., data=oobTrainNZV, method="kknn", trControl=fitControl, verbose=FALSE)
oobKNfit1
varImp(oobKNfit1)

#k-Nearest Neighbors 

#9732 samples
#11 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7785, 7785, 7786, 7786, 7786 
#Resampling results across tuning parameters:
  
#  kmax  Accuracy   Kappa    
#5     0.2956227  0.1306835
#7     0.3125774  0.1412577
#9     0.3118569  0.1380234

#Tuning parameter 'distance' was held constant at a value of 2
#Tuning parameter 'kernel' was held constant at a value of optimal
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were kmax = 7, distance = 2 and kernel = optimal.

### iphoneRFE

set.seed(99)
oobKNfit2 <- train(iphonesentiment~., data=oobTrainRFE, method="kknn", trControl=fitControl, verbose=FALSE)
oobKNfit2
varImp(oobKNfit2)

#k-Nearest Neighbors 

#9732 samples
#10 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7785, 7785, 7786, 7786, 7786 
#Resampling results across tuning parameters:
  
#  kmax  Accuracy   Kappa    
#5     0.2995271  0.1467185
#7     0.3095968  0.1513871
#9     0.3190505  0.1578727

#Tuning parameter 'distance' was held constant at a value of 2
#Tuning parameter 'kernel' was held constant at a value of optimal
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were kmax = 9, distance = 2 and kernel = optimal.


############################
# Predict testSet/validation
############################

## ------- SVM ------- ##

SVMpred <- predict(oobSVMfit, oobTestDF)
postResample(SVMpred, oobTestDF$iphonesentiment)
SVMpred
plot(SVMpred, oobTestDF$iphonesentiment)
cmSVM <- confusionMatrix(SVMpred, oobTestDF$iphonesentiment) 
cmSVM

SVMpred1 <- predict(oobSVMfit1, oobTestNZV)
postResample(SVMpred1, oobTestNZV$iphonesentiment)
SVMpred1
plot(SVMpred1, oobTestNZV$iphonesentiment)
cmSVM1 <- confusionMatrix(SVMpred1, oobTestNZV$iphonesentiment) 
cmSVM1

SVMpred2 <- predict(oobSVMfit2, oobTestRFE)
postResample(SVMpred2, oobTestRFE$iphonesentiment)
SVMpred2
plot(SVMpred2, oobTestRFE$iphonesentiment)
cmSVM2 <- confusionMatrix(SVMpred2, oobTestRFE$iphonesentiment) 
cmSVM2

## ------- RF ------- ##

RFpred <- predict(oobRFfit, oobTestDF)
postResample(RFpred, oobTestDF$iphonesentiment)
RFpred
plot(RFpred, oobTestDF$iphonesentiment)
cmRF <- confusionMatrix(RFpred, oobTestDF$iphonesentiment) 
cmRF

RFpred1 <- predict(oobRFfit1, oobTestNZV)
postResample(RFpred1, oobTestNZV$iphonesentiment)
RFpred1
plot(RFpred1, oobTestNZV$iphonesentiment)
cmRF1 <- confusionMatrix(RFpred1, oobTestNZV$iphonesentiment) 
cmRF1

RFpred2 <- predict(oobRFfit2, oobTestRFE)
postResample(RFpred2, oobTestRFE$iphonesentiment)
RFpred2
plot(RFpred2, oobTestRFE$iphonesentiment)
cmRF2 <- confusionMatrix(RFpred2, oobTestRFE$iphonesentiment) 
cmRF2

## ------- C5.0 ------- ##

C5pred <- predict(oobC5fit, oobTestDF)
postResample(C5pred, oobTestDF$iphonesentiment)
C5pred
plot(C5pred, oobTestDF$iphonesentiment)
cmC5 <- confusionMatrix(C5pred, oobTestDF$iphonesentiment) 
cmC5

C5pred1 <- predict(oobC5fit1, oobTestNZV)
postResample(C5pred1, oobTestNZV$iphonesentiment)
C5pred1
plot(C5pred1, oobTestNZV$iphonesentiment)
cmC51 <- confusionMatrix(C5pred1, oobTestNZV$iphonesentiment) 
cmC51

C5pred2 <- predict(oobC5fit2, oobTestRFE)
postResample(C5pred2, oobTestRFE$iphonesentiment)
C5pred2
plot(C5pred2, oobTestRFE$iphonesentiment)
cmC52 <- confusionMatrix(C5pred2, oobTestRFE$iphonesentiment) 
cmC52

## ------- KKNN ------- ##

KNpred <- predict(oobKNfit, oobTestDF)
postResample(KNpred, oobTestDF$iphonesentiment)
KNpred
plot(KNpred, oobTestDF$iphonesentiment)
cmKN <- confusionMatrix(KNpred, oobTestDF$iphonesentiment) 
cmKN

KNpred1 <- predict(oobKNfit1, oobTestNZV)
postResample(KNpred1, oobTestNZV$iphonesentiment)
KNpred1
plot(KNpred1, oobTestNZV$iphonesentiment)
cmKN1 <- confusionMatrix(KNpred1, oobTestNZV$iphonesentiment) 
cmKN1

KNpred2 <- predict(oobKNfit2, oobTestRFE)
postResample(KNpred2, oobTestRFE$iphonesentiment)
KNpred2
plot(KNpred2, oobTestRFE$iphonesentiment)
cmKN2 <- confusionMatrix(KNpred2, oobTestRFE$iphonesentiment) 
cmKN2

##################
# Model selection
##################

#-- CompleteResponses --# 

Selectmodel <- resamples(list(svmLinear=oobSVMfit ,rf=oobRFfit, C5.0=oobC5fit, kknn=oobKNfit))
Selectmodel1 <- resamples(list(svmLinear=oobSVMfit1 ,rf=oobRFfit1, C5.0=oobC5fit1, kknn=oobKNfit1))
Selectmodel2 <- resamples(list(svmLinear=oobSVMfit2 ,rf=oobRFfit2, C5.0=oobC5fit2, kknn=oobKNfit2))

# output summary metrics for tuned models 

summary(Selectmodel)

#Call:
#  summary.resamples(object = Selectmodel)

#Models: svmLinear, rf, C5.0, kknn 
#Number of resamples: 5 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.6942446 0.7031330 0.7127441 0.7088982 0.7168551 0.7175141    0
#rf        0.7380586 0.7489733 0.7542416 0.7537018 0.7589928 0.7682425    0
#C5.0      0.7657935 0.7728674 0.7733813 0.7736342 0.7734977 0.7826310    0
#kknn      0.3201439 0.3237410 0.3323061 0.3332303 0.3364150 0.3535457    0

#Kappa 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.3779484 0.3968505 0.4212853 0.4110912 0.4296309 0.4297410    0
#rf        0.4763902 0.4984280 0.5089546 0.5095732 0.5213264 0.5427670    0
#C5.0      0.5454137 0.5585497 0.5585858 0.5605223 0.5603994 0.5796631    0
#kknn      0.1531366 0.1560879 0.1649985 0.1654273 0.1664795 0.1864338    0

summary(Selectmodel1)

#Call:
#  summary.resamples(object = Selectmodel1)

#Models: svmLinear, rf, C5.0, kknn 
#Number of resamples: 5 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.6793422 0.6813977 0.6815614 0.6859847 0.6872111 0.7004111    0
#rf        0.7447355 0.7619537 0.7633470 0.7606876 0.7656732 0.7677287    0
#C5.0      0.7447355 0.7564234 0.7611710 0.7587353 0.7620761 0.7692703    0
#kknn      0.3031860 0.3062693 0.3081664 0.3125774 0.3138161 0.3314491    0

#Kappa 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.3342475 0.3381806 0.3417146 0.3508510 0.3524454 0.3876669    0
#rf        0.4978138 0.5353298 0.5385912 0.5325235 0.5422797 0.5486029    0
#C5.0      0.4962738 0.5207245 0.5335136 0.5270051 0.5340867 0.5504268    0
#kknn      0.1322756 0.1332712 0.1385771 0.1412577 0.1412751 0.1608897    0

summary(Selectmodel2)

#Call:
#  summary.resamples(object = Selectmodel2)

#Models: svmLinear, rf, C5.0, kknn 
#Number of resamples: 5 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.6795069 0.6945585 0.6997429 0.6967761 0.7004111 0.7096608    0
#rf        0.7606574 0.7686375 0.7700205 0.7711683 0.7749229 0.7816033    0
#C5.0      0.7611710 0.7677287 0.7718397 0.7697292 0.7734977 0.7744090    0
#kknn      0.3124358 0.3133025 0.3144913 0.3190505 0.3256292 0.3293936    0

#Kappa 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.3433375 0.3761851 0.3899431 0.3818696 0.3906249 0.4092572    0
#rf        0.5356951 0.5521208 0.5564589 0.5587171 0.5679946 0.5813160    0
#C5.0      0.5388986 0.5517004 0.5625102 0.5564853 0.5645205 0.5647967    0
#kknn      0.1488388 0.1515975 0.1545700 0.1578727 0.1625779 0.1717791    0

##--- Save/load top performing model ---##

# save top performing model after validation
saveRDS(oobC5fit, "oobC5fit.rds")  

# load and name model
SelectModelC5<- readRDS("oobC5fit.rds")

#############################
# Predict new data for iphone
#############################

iphoneLargeMatrix <- read.csv("Large_Data_Matrix_iphone.csv", stringsAsFactors = FALSE)

iphoneLargeMatrix <- subset(iphoneLargeMatrix, select = -(id))
iphoneLargeMatrix['iphonesentiment'] <- NA


LMpred <- predict(oobC5fit, iphoneLargeMatrix)
LMpred
summary(LMpred)
plot(LMpred)


#####################################
# Create new dataset with predictions
#####################################

LMprediphone <- add_predictions(iphoneLargeMatrix, oobC5fit, var = "iphonesentiment", type = NULL)
summary(LMprediphone)
write.csv(LMprediphone, file="iphoneLargeMatrixPRED.csv", row.names = TRUE)

predpie <- plot_ly(LMprediphone, labels = ~iphonesentiment, type = 'pie')
predpie <- predpie %>% layout(title = 'Predicted sentiment iPhone',
                          xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                          yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

predpie

# STOP CLUSTER 

stopCluster(cl)

