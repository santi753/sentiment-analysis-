#### GALAXY

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

galaxyDF <- read.csv("galaxy_smallmatrix_labeled_9d.csv", stringsAsFactors = FALSE)


################
# Evaluate data
################

head(galaxyDF)
tail(galaxyDF)
str(galaxyDF)
summary(galaxyDF)

plot_ly(galaxyDF, x= ~galaxyDF$galaxysentiment, type='histogram')

#############
# Preprocess
#############

anyNA(galaxyDF)
anyDuplicated(galaxyDF)
galaxyDF$galaxysentiment <- as.factor(galaxyDF$galaxysentiment)

#####################
# EDA/Visualizations
#####################


fig1 <- plot_ly(galaxyDF, x= ~galaxyDF$galaxysentiment, type='histogram', color = ~galaxysentiment, colors = "Set2")
fig1 <- fig %>% layout(title = "Galaxy Sentiment",
                      xaxis = list(title = "Sentiment"),
                      yaxis = list(title = "Count"))
fig1

iphoneDF %>%
  ggplot( aes(x=galaxysentiment)) +
  geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8)

hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)
hist(galaxyDF)


#######################
# Feature selection
#######################

# ---- Correlation analysis ---- #

corrAll <- cor(galaxyDF)
corrAll
corrplot(corrAll, method = "circle")
corrplot(corrAll, order = "hclust") 

# there are no characteristics highly correlated with the dependent

# ---- Examine Feature Variance ---- #

nzvMetrics <- nearZeroVar(galaxyDF, saveMetrics = TRUE)
nzvMetrics

nzv <- nearZeroVar(galaxyDF, saveMetrics = FALSE) 
nzv

galaxyNZV <- galaxyDF[,-nzv]
str(galaxyNZV)

# ---- Recursive Feature Elimination  ---- #

# Let's sample the data before using RFE
set.seed(123)
galaxySample <- galaxyDF[sample(1:nrow(galaxyDF), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 2,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 galaxysentiment) 
rfeResults <- rfe(galaxySample[,1:58], 
                  galaxySample$galaxysentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults

# Plot results
plot(rfeResults, type=c("g", "o"))

# create new data set with rfe recommended features
galaxyRFE <- galaxyDF[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
galaxyRFE$galaxysentiment <- galaxyDF$galaxysentiment

# review outcome
str(galaxyRFE)


##################
# Train/test sets
##################

# -------- galaxyDF -------- #

set.seed(123) 
inTrainingDFG <- createDataPartition(galaxyDF$galaxysentiment, p=0.75, list=FALSE)
oobTrainDFG <- galaxyDF[inTrainingDFG,]   
oobTestDFG <- galaxyDF[-inTrainingDFG,]   
# verify number of obs 
nrow(oobTrainDFG) 
nrow(oobTestDFG)  

# -------- galaxyNZV -------- #

set.seed(123) 
inTrainingNZVG <- createDataPartition(galaxyNZV$galaxysentiment, p=0.75, list=FALSE)
oobTrainNZVG <- galaxyNZV[inTrainingNZVG,]   
oobTestNZVG <- galaxyNZV[-inTrainingNZVG,]   
# verify number of obs 
nrow(oobTrainNZVG) 
nrow(oobTestNZVG)  

# -------- galaxyRFE -------- #

set.seed(123) 
inTrainingRFEG <- createDataPartition(galaxyRFE$galaxysentiment, p=0.75, list=FALSE)
oobTrainRFEG <- galaxyRFE[inTrainingRFE,]   
oobTestRFEG <- galaxyRFE[-inTrainingRFE,]   
# verify number of obs 
nrow(oobTrainRFEG) 
nrow(oobTestRFEG)  

################
# Train control
################

# set cross validation
fitControl <- trainControl(method="repeatedcv", number=5, repeats=1) 

###############
# Train models
###############

## ------- SVM ------- ##

### galaxyDF

set.seed(99)
oobSVMfitG <- train(galaxysentiment ~., data = oobTrainDFG, method = "svmLinear2", trControl = fitControl,  preProcess = c("center","scale"))
oobSVMfitG
varImp(oobSVMfitG)

#Support Vector Machines with Linear Kernel 

#9686 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#Pre-processing: centered (58), scaled (58) 
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7749, 7748, 7748, 7749, 7750 
#Resampling results across tuning parameters:
  
#  cost  Accuracy   Kappa    
#0.25  0.6968847  0.3547283
#0.50  0.7004985  0.3672686
#1.00  0.7003942  0.3741078

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cost = 0.5.


### galaxyNZV

set.seed(99)
oobSVMfit1G <- train(galaxysentiment~., data=oobTrainNZVG, method="svmLinear2", trControl=fitControl, verbose=FALSE, scale = FALSE)
oobSVMfit1G
varImp(oobSVMfit1G)

#Support Vector Machines with Linear Kernel 

#9686 samples
#11 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7749, 7748, 7748, 7749, 7750 
#Resampling results across tuning parameters:
  
#  cost  Accuracy   Kappa    
#0.25  0.6925461  0.3407028
#0.50  0.6925461  0.3406370
#31.00  0.6924429  0.3405531

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cost = 0.25.


### galaxyRFE

set.seed(123)
oobSVMfit2G <- train(galaxysentiment~., data=oobTrainRFEG, method="svmLinear2", trControl=fitControl, verbose=FALSE,scale = FALSE)
oobSVMfit2G
varImp(oobSVMfit2G)

#Support Vector Machines with Linear Kernel 

#9686 samples
#27 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7747, 7748, 7749, 7750, 7750 
#Resampling results across tuning parameters:
  
#  cost  Accuracy   Kappa    
#0.25  0.7045226  0.3843265
#0.50  0.7052457  0.3861744
#1.00  0.7051424  0.3858370

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cost = 0.5.


## ------- RF ------- ##

### galaxyDF

set.seed(123)
oobRFfitG <- train(galaxysentiment~., data=oobTrainDFG, method="rf", importance=T, trControl=fitControl)

# manual grid
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfitG <- train(galaxysentiment~.,data=oobTrainDFG ,method="rf",
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfGrid)
oobRFfitG
plot(oobRFfitG)
varImp(oobRFfitG)

#Random Forest 

#9686 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7747, 7748, 7749, 7750, 7750 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#4     0.7389025  0.4565358
#5     0.7446856  0.4703542
#6     0.7491242  0.4823771

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 6.


### galaxyNZV

set.seed(123)
oobRFfit1G <- train(galaxysentiment~., data=oobTrainNZVG, method="rf", importance=T, trControl=fitControl)

# manual grid
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfit1G <- train(galaxysentiment~.,data=oobTrainNZVG,method="rf",
                   importance=T,
                   trControl=fitControl,
                   tuneGrid=rfGrid)
oobRFfit1G
plot(oobRFfit1G)
varImp(oobRFfit1G)

#Random Forest 

#9686 samples
#11 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7747, 7748, 7749, 7750, 7750 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#4     0.7574859  0.5111027
#5     0.7556269  0.5088122
#6     0.7538722  0.5062172

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 4.


### galaxyRFE

set.seed(123)
oobRFfit2G <- train(galaxysentiment~., data=oobTrainRFEG, method="rf", importance=T, trControl=fitControl)

# manual grid
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfit2G <- train(galaxysentiment~.,data=oobTrainRFE,method="rf",
                   importance=T,
                   trControl=fitControl,
                   tuneGrid=rfGrid)
oobRFfit2G
plot(oobRFfit2G)
varImp(oobRFfit2G)

#Random Forest 

#9686 samples
#27 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7747, 7748, 7749, 7750, 7750 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#4     0.7603748  0.5125764
#5     0.7678104  0.5338713
#6     0.7693601  0.5387590

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 6.


## ------- C5.0 ------- ##

### galaxyDF

set.seed(99)
oobC5fitG <- train(galaxysentiment~., data=oobTrainDFG, method="C5.0", trControl=fitControl, verbose=FALSE)
oobC5fitG
varImp(oobC5fitG)

#C5.0 

#9686 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7749, 7748, 7748, 7749, 7750 
#Resampling results across tuning parameters:
  
#  model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7652274  0.5305922
#rules  FALSE   10      0.7557315  0.5095965
#rules  FALSE   20      0.7557315  0.5095965
#rules   TRUE    1      0.7674997  0.5356849
#rules   TRUE   10      0.7525307  0.5045868
#rules   TRUE   20      0.7525307  0.5045868
#tree   FALSE    1      0.7654343  0.5317675
#tree   FALSE   10      0.7559365  0.5137897
#tree   FALSE   20      0.7559365  0.5137897
#tree    TRUE    1      0.7668795  0.5350193
#tree    TRUE   10      0.7563496  0.5147155
#tree    TRUE   20      0.7563496  0.5147155

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = rules and winnow = TRUE.


### galaxyNZV 

set.seed(99)
oobC5fit1G <- train(galaxysentiment~., data=oobTrainNZVG, method="C5.0", trControl=fitControl, verbose=FALSE)
oobC5fit1G
varImp(oobC5fit1G)

#C5.0 

#9686 samples
#11 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7749, 7748, 7748, 7749, 7750 
#Resampling results across tuning parameters:
  
#  model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7544922  0.5022521
#rules  FALSE   10      0.7364261  0.4586011
#rules  FALSE   20      0.7364261  0.4586011
#rules   TRUE    1      0.7535629  0.5001654
#rules   TRUE   10      0.7368379  0.4583042
#rules   TRUE   20      0.7368379  0.4583042
#tree   FALSE    1      0.7541822  0.5023184
#tree   FALSE   10      0.7383872  0.4657498
#tree   FALSE   20      0.7383872  0.4657498
#tree    TRUE    1      0.7537691  0.5016601
#tree    TRUE   10      0.7360128  0.4588483
#tree    TRUE   20      0.7360128  0.4588483

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = rules and winnow = FALSE.

### galaxyRFE

set.seed(99)
oobC5fit2G <- train(galaxysentiment~., data=oobTrainRFEG, method="C5.0", trControl=fitControl, verbose=FALSE)
oobC5fit2G
varImp(oobC5fit2G)

#C5.0 

#9686 samples
#27 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7749, 7748, 7748, 7749, 7750 
#Resampling results across tuning parameters:
  
#  model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7665699  0.5340557
#rules  FALSE   10      0.7559364  0.5133809
#rules  FALSE   20      0.7559364  0.5133809
#rules   TRUE    1      0.7669832  0.5348998
#rules   TRUE   10      0.7535610  0.5070210
#rules   TRUE   20      0.7535610  0.5070210
#tree   FALSE    1      0.7663637  0.5337110
#tree   FALSE   10      0.7564531  0.5160581
#tree   FALSE   20      0.7564531  0.5160581
#tree    TRUE    1      0.7667767  0.5346834
#tree    TRUE   10      0.7578981  0.5187866
#tree    TRUE   20      0.7578981  0.5187866

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = rules and winnow = TRUE.

## ------- KKNN ------- ##

### galaxyDF

set.seed(99)
oobKNfitG <- train(galaxysentiment~., data=oobTrainDFG, method="kknn", trControl=fitControl, verbose=FALSE)
oobKNfitG
varImp(oobKNfitG)

#k-Nearest Neighbors 

#9686 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7749, 7748, 7748, 7749, 7750 
#Resampling results across tuning parameters:
  
#  kmax  Accuracy   Kappa    
#5     0.6091560  0.3747447
#7     0.6675870  0.4274417
#9     0.7470579  0.5050756

#Tuning parameter 'distance' was held constant at a value of 2
#Tuning parameter 'kernel' was held constant at a value of optimal
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were kmax = 9, distance = 2 and kernel = optimal.

### galaxyNZV 

set.seed(99)
oobKNfit1G <- train(galaxysentiment~., data=oobTrainNZVG, method="kknn", trControl=fitControl, verbose=FALSE)
oobKNfit1G
varImp(oobKNfit1G)

#k-Nearest Neighbors 

#9686 samples
#11 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7749, 7748, 7748, 7749, 7750 
#Resampling results across tuning parameters:
  
#  kmax  Accuracy   Kappa    
#5     0.6513798  0.4014603
#7     0.7333270  0.4752241
#9     0.7294081  0.4704635

#Tuning parameter 'distance' was held constant at a value of 2
#Tuning parameter 'kernel' was held constant at a value of optimal
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were kmax = 7, distance = 2 and kernel = optimal.

### galaxyRFE

set.seed(99)
oobKNfit2G <- train(galaxysentiment~., data=oobTrainRFEG, method="kknn", trControl=fitControl, verbose=FALSE)
oobKNfit2G
varImp(oobKNfit2G)

#k-Nearest Neighbors 

#9686 samples
#27 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 7749, 7748, 7748, 7749, 7750 
#Resampling results across tuning parameters:
  
#  kmax  Accuracy   Kappa    
#5     0.6101889  0.3770160
#7     0.6705805  0.4316615
#9     0.7508774  0.5119694

#Tuning parameter 'distance' was held constant at a value of 2
#Tuning parameter 'kernel' was held constant at a value of optimal
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were kmax = 9, distance = 2 and kernel = optimal.


############################
# Predict testSet/validation
############################

## ------- SVM ------- ##

SVMpredG <- predict(oobSVMfitG, oobTestDFG)
postResample(SVMpredG, oobTestDFG$galaxysentiment)
SVMpredG
plot(SVMpredG, oobTestDFG$galaxysentiment)
cmSVMG <- confusionMatrix(SVMpredG, oobTestDFG$galaxysentiment) 
cmSVMG

SVMpred1G <- predict(oobSVMfit1G, oobTestNZVG)
postResample(SVMpred1G, oobTestNZVG$galaxysentiment)
SVMpred1G
plot(SVMpred1G, oobTestNZVG$galaxysentiment)
cmSVM1G <- confusionMatrix(SVMpred1G, oobTestNZVG$galaxysentiment) 
cmSVM1G

SVMpred2G <- predict(oobSVMfit2G, oobTestRFEG)
postResample(SVMpred2G, oobTestRFEG$galaxysentiment)
SVMpred2G
plot(SVMpred2G, oobTestRFEG$galaxysentiment)
cmSVM2G <- confusionMatrix(SVMpred2G, oobTestRFEG$galaxysentiment) 
cmSVM2G

## ------- RF ------- ##

RFpredG <- predict(oobRFfitG, oobTestDFG)
postResample(RFpredG, oobTestDFG$galaxysentiment)
RFpredG
plot(RFpredG, oobTestDFG$galaxysentiment)
cmRFG <- confusionMatrix(RFpredG, oobTestDFG$galaxysentiment) 
cmRFG

RFpred1G <- predict(oobRFfit1G, oobTestNZVG)
postResample(RFpred1G, oobTestNZVG$galaxysentiment)
RFpred1G
plot(RFpred1G, oobTestNZVG$galaxysentiment)
cmRF1G <- confusionMatrix(RFpred1G, oobTestNZVG$galaxysentiment) 
cmRF1G

RFpred2G <- predict(oobRFfit2G, oobTestRFEG)
postResample(RFpred2G, oobTestRFEG$galaxysentiment)
RFpred2G
plot(RFpred2G, oobTestRFEG$galaxysentiment)
cmRF2G <- confusionMatrix(RFpred2G, oobTestRFE$galaxysentiment) 
cmRF2G

## ------- C5.0 ------- ##

C5predG <- predict(oobC5fitG, oobTestDFG)
postResample(C5predG, oobTestDFG$galaxysentiment)
C5predG
plot(C5predG, oobTestDFG$galaxysentiment)
cmC5G <- confusionMatrix(C5predG, oobTestDFG$galaxysentiment) 
cmC5G

C5pred1G <- predict(oobC5fit1G, oobTestNZVG)
postResample(C5pred1G, oobTestNZVG$galaxysentiment)
C5pred1G
plot(C5pred1G, oobTestNZVG$galaxysentiment)
cmC51G <- confusionMatrix(C5pred1G, oobTestNZVG$galaxysentiment) 
cmC51G

C5pred2G <- predict(oobC5fit2G, oobTestRFEG)
postResample(C5pred2G, oobTestRFEG$galaxysentiment)
C5pred2G
plot(C5pred2G, oobTestRFEG$galaxysentiment)
cmC52G <- confusionMatrix(C5pred2G, oobTestRFEG$galaxysentiment) 
cmC52G

## ------- KKNN ------- ##

KNpredG <- predict(oobKNfitG, oobTestDFG)
postResample(KNpredG, oobTestDFG$galaxysentiment)
KNpredG
plot(KNpredG, oobTestDFG$galaxysentiment)
cmKNG <- confusionMatrix(KNpredG, oobTestDFG$galaxysentiment) 
cmKNG

KNpred1G <- predict(oobKNfit1G, oobTestNZVG)
postResample(KNpred1G, oobTestNZVG$galaxysentiment)
KNpred1G
plot(KNpred1G, oobTestNZVG$galaxysentiment)
cmKN1G <- confusionMatrix(KNpred1G, oobTestNZVG$galaxysentiment) 
cmKN1G

KNpred2G <- predict(oobKNfit2G, oobTestRFEG)
postResample(KNpred2G, oobTestRFEG$galaxysentiment)
KNpred2G
plot(KNpred2G, oobTestRFEG$galaxysentiment)
cmKN2G <- confusionMatrix(KNpred2G, oobTestRFEG$galaxysentiment) 
cmKN2G

##################
# Model selection
##################

#-- CompleteResponses --# 

SelectmodelG <- resamples(list(svmLinear=oobSVMfitG ,rf=oobRFfitG, C5.0=oobC5fitG, kknn=oobKNfitG))
Selectmodel1G <- resamples(list(svmLinear=oobSVMfit1G ,rf=oobRFfit1G, C5.0=oobC5fit1G, kknn=oobKNfit1G))
Selectmodel2G <- resamples(list(svmLinear=oobSVMfit2G ,rf=oobRFfit2G, C5.0=oobC5fit2G, kknn=oobKNfit2G))

# output summary metrics for tuned models 

#summary(SelectmodelG)

#Call:
#  summary.resamples(object = SelectmodelG)

#Models: svmLinear, rf, C5.0, kknn 
#Number of resamples: 5 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.6831785 0.6960784 0.6969541 0.7004985 0.7076446 0.7186371    0
#rf        0.7390407 0.7417355 0.7528380 0.7491242 0.7547754 0.7572314    0
#C5.0      0.7585139 0.7614868 0.7660124 0.7674997 0.7729618 0.7785235    0
#kknn      0.7383901 0.7443182 0.7449664 0.7470579 0.7502580 0.7573567    0

#Kappa 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.3293290 0.3451012 0.3615241 0.3672686 0.3998367 0.4005520    0
#rf        0.4595024 0.4600004 0.4939710 0.4823771 0.4962830 0.5021287    0
#C5.0      0.5152624 0.5181589 0.5373737 0.5356849 0.5477975 0.5598320    0
#kknn      0.4867102 0.4896090 0.5033559 0.5050756 0.5169489 0.5287541    0

summary(Selectmodel1G)

#Call:
#  summary.resamples(object = Selectmodel1G)

#Models: svmLinear, rf, C5.0, kknn 
#Number of resamples: 5 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.6886939 0.6917914 0.6929825 0.6925461 0.6934985 0.6957645    0
#rf        0.7515496 0.7529654 0.7579979 0.7574859 0.7620031 0.7629132    0
#C5.0      0.7435501 0.7490965 0.7564499 0.7544922 0.7572314 0.7661332    0
#kknn      0.7165720 0.7229102 0.7376033 0.7333270 0.7425181 0.7470315    0

#Kappa 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.3295916 0.3378772 0.3380954 0.3407028 0.3382534 0.3596963    0
#rf        0.4941581 0.5013425 0.5189805 0.5111027 0.5191788 0.5218534    0
#C5.0      0.4786885 0.4841432 0.5085877 0.5022521 0.5135239 0.5263174    0
#kknn      0.4407352 0.4553821 0.4849678 0.4752241 0.4918771 0.5031584    0

summary(Selectmodel2G)

#Call:
#  summary.resamples(object = Selectmodel2G)

#Models: svmLinear, rf, C5.0, kknn 
#Number of resamples: 5 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.6993296 0.7024793 0.7053664 0.7052457 0.7066116 0.7124419    0
#rf        0.7607014 0.7639463 0.7683179 0.7693601 0.7759422 0.7778926    0
#C5.0      0.7585139 0.7620031 0.7639463 0.7669832 0.7734778 0.7769747    0
#kknn      0.7435501 0.7474174 0.7475478 0.7508774 0.7559340 0.7599380    0

#Kappa 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmLinear 0.3738513 0.3790058 0.3852968 0.3861744 0.3889889 0.4037292    0
#rf        0.5201724 0.5243128 0.5412014 0.5387590 0.5513686 0.5567397    0
#C5.0      0.5173198 0.5181589 0.5330429 0.5348998 0.5492699 0.5567076    0
#kknn      0.4955697 0.4983310 0.5083561 0.5119694 0.5253623 0.5322281    0


##--- Save/load top performing model ---##

# save top performing model after validation
saveRDS(oobC5fitG, "oobC5fitG.rds")  

# load and name model
SelectModelC5G<- readRDS("oobC5fitG.rds")

#############################
# Predict new data for galaxy
#############################

galaxyLargeMatrix <- read.csv("Large_Data_Matrix_galaxy.csv", stringsAsFactors = FALSE)

galaxyLargeMatrix <- subset(galaxyLargeMatrix, select = -(id))
galaxyLargeMatrix['galaxysentiment'] <- NA


LMpredG <- predict(oobC5fitG, galaxyLargeMatrix)
LMpredG
summary(LMpredG)
plot(LMpredG)

#####################################
# Create new dataset with predictions
#####################################

LMpredgalaxy <- add_predictions(galaxyLargeMatrix, oobC5fitG, var = "galaxysentiment", type = NULL)
summary(LMpredgalaxy)
write.csv(LMpredgalaxy, file="galaxyLargeMatrixPRED.csv", row.names = TRUE)

predpie1 <- plot_ly(LMpredgalaxy, labels = ~galaxysentiment, type = 'pie')
predpie1 <- predpie1 %>% layout(title = 'Predicted sentiment Galaxy',
                              xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                              yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

predpie1


# STOP CLUSTER 

stopCluster(cl)