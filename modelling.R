library(readxl)
library(caret)
library(ggplot2)
library(randomForest)
library(mlbench)
library(magrittr)
library(e1071)
library(kernlab)
library(tidyr)
library(dplyr)
library(leaps)
library(MASS)

## Data Ingestion and Cleaning


rm(list = ls())
dev.off

Pyramid_Rating_System_Database <- read_excel("C:/Users/Vignesh/Downloads/Pyramid Rating System Database.xlsm", 
                                             sheet = "Team Stats", col_names = T)

df = Pyramid_Rating_System_Database

summary(df)

## remove unncessary rows and columns

df1 = df[,c(1:68)]
df1 = df1[,1:(ncol(df1)-5)]

df1$`WS Play` = NULL
df1$Post = NULL
df1$W = NULL
df1$Lg = NULL
df1$L = NULL
df1$Rk. = NULL
df1$Team = NULL
df1$SB = NULL
df1$CS = NULL
df1$Age...41 = NULL
df1$BK = NULL
ncol(df1)
df1 = as.data.frame(df1)
summary(df1)

df1$`Tm/Frch` = substr(df1$`Tm/Frch`, 0,3)
names(df1)[names(df1) == 'Tm/Frch'] = 'Team'


df1$SV = NULL
df1$Div. = NULL

df1[6,1]= 2019
summary(df1$Year)

df1$WS = ifelse(df$`WS Play` != 'Win', 0,1)
df1$FreeBase = df1$IBB...37 + df1$HBP...34 + df1$BB...26
df1$GaveBase = df1$IBB...51 + df1$HBP...53 + df1$BB...50
df1$TotalBaseFree = df1$FreeBase - df1$GaveBase


df1$IBB...37 = NULL
df1$HBP...34 = NULL
df1$BB...26 = NULL
df1$IBB...51 = NULL
df1$HBP...53 = NULL
df1$BB...50 = NULL
df1$GB = NULL
df1$CG = NULL
df1$HR...21 = NULL
df1$`1B` = NULL
df1$`2B`= NULL
df1$`3B`= NULL
df1$AB = NULL
df1$PA = NULL
df$AB = NULL


df1$WS = ifelse(is.na(df1$WS), 0, df1$WS)
head(df1$WS, 30)
df1 = df1[df1$Year > 1989,]

df1$Year = NULL

##### FEATURE SELECTION
df1$WS
na.omit(df1)
set.seed(2)

df1$WS = as.factor(df1$WS)

levels(df1$WS)
df1$Team = as.factor(df1$Team)
levels(df1$Team)
df1$Team = NULL

df1$nonPWAR = df1$oWAR + df1$dWAR
df1$pWAR = (df1$bWAR...62 + df1$fWAR)/2

summary(df1$oWAR)

## train-test split (80/20)

training.sample = df1$WS %>%
  createDataPartition(p = .80, list = F)
train.data = df1[training.sample,]
test.data = df1[-training.sample,]

summary(train.data$WS)
for (cols in 1:ncol(train.data)){
  print(typeof(train.data[[cols]]))
}

## Feature Importance Analysis:

### LVQ
library(mlbench)
control = trainControl(method = 'repeatedcv', number = 10, repeats = 10)
modeling = train(WS ~., data = train.data, method = 'lvq', 
                 preProcess = c('scale'), trControl = control)
features = varImp(modeling, scale = F)
print(features)

summary(df1)
## RECURSIVE

summary(train.data)

control1 = rfeControl(functions = rfFuncs, method = 'cv', number = 10)
results = rfe(WS ~ ., data = train.data, sizes = c(1:10), rfeControl = control1)
predictors(results)
plot(results, type = c('g','o'))


summary(df1)
df.model = df1[,c(1:2, 6, 8:11, 19, 28,29, 35,38:40)]
colnames(df.model) = make.names(colnames(df.model))
colnames(df.model)[1:3] = c('Pct','Age','SB')
summary(df.model)


########### LOGISTIC Regression

na.omit(df.model)
head(df.model,30)
for (cols in 1:ncol(training)){
  print(typeof(training[[cols]]))
}

set.seed(3)
training.sample1 = df.model$WS %>%
  createDataPartition(p = .80, list = F)
training = df.model[training.sample1,]
testing = df.model[-training.sample1,]
summary(training)
summary(testing)

library(DMwR)
library(pROC)
balanced.data = SMOTE(WS ~ ., training, perc.over = 4000, k= 5, perc.under = 100)
as.data.frame(table(balanced.data$WS))


full.model = glm(WS ~ ., data = balanced.data, family = 'binomial')
step.model = stepAIC(full.model, direction = 'both',
                     trace = F)

formula(step.model)

logModel = glm(WS ~ ., data = balanced.data, family = 'binomial')
probabilities = logModel %>% predict(testing, type = "response")
contrasts(testing$WS)

predicted.classes = ifelse(probabilities > 0.5, 0,1)
mean(predicted.classes == testing$WS)
confusionMatrix(table(predicted.classes, testing$WS))
str(testing$WS)

## 3% accuracy -> not a good predictor

library(LiblineaR)
#svm_Linear = train(Outcome ~ Age + BMI + MeanAvgTemp + HeatIndex + Pregnancies + PM + DispositionFactor, data = train.data, method = 'svmRadial', trControl = trctrl, preProcess = c('center','scale'), tuneLength = 10)
### lets try SVM
trctrl = trainControl(method = 'repeatedcv', number = 10, repeats = 10)
svmRadial =  train(WS ~ Pct + OPS+ OBP + ERA + H9 + pWAR , data = balanced.data, method = 'svmRadial', trControl = trctrl, preProcess = c('center','scale'), tuneLength = 10)
prob1 = predict(svmRadial, testing)
confusionMatrix(table(prob1, testing$WS))



#svmLinear = train(WS ~ Pct + OPS + OBP + ERA + H9 + pWAR + nonPWAR, data = balanced.data, method = 'svmPoly', trControl = trctrl, preProcess = c('center','scale'), tuneLength = 10)
summary(df.model$nonPWAR)

#auc = roc(response = testing$WS, predictor = )

##WORSE
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5,5))
svm_Radial_Grid = train(WS ~ Pct + OPS + OBP + ERA + H9 + pWAR, data = balanced.data, method = 'svmRadialCost',
                        trControl = trctrl,
                        preProcess = c('center','scale'),
                        tuneGrid = grid,
                        tuneLength = 10)

plot(svm_Radial_Grid)
test_prep_grid = predict(svm_Radial_Grid, testing)
confusionMatrix(table(test_prep_grid, testing$WS))
cm = confusionMatrix(test_prep_grid, testing$WS)

cm$byClass

summary(svm_Radial_Grid)

###### LETs TRY DTREES
summary(training$WS)
24/680

model1 = train(WS ~Pct + OPS + OBP + ERA + H9 + pWAR,  data = balanced.data, method = 'rpart', trControl = trctrl)
prob2 = predict(model1, testing)

confusionMatrix(table(prob2, testing$WS))
ggplot(model1)
densityplot(model1, pch = '|')

##### RANDOM FOREST

rf = randomForest(WS ~ Pct + OPS + OBP + ERA + H9 + pWAR, data = balanced.data, ntree = 200, varImp = T)
plot(rf)

predicted.response = predict(rf, testing)
confusionMatrix(table(predicted.response, testing$WS))
cm2 = confusionMatrix(predicted.response, testing$WS)
cm2$byClass

library(randomForestExplainer)

plot_multi_way_importance(rf)

### GBM

gbmGrid = expand.grid(interaction.depth = c(1, 5, 9), 
                      n.trees = (1:30)*50, 
                      shrinkage = 0.1,
                      n.minobsinnode = 20)
gbmFit2 = train(WS ~ Pct + OPS + OBP + ERA + H9 + pWAR, data= balanced.data, method = 'gbm', trControl = trctrl, verbose = F, tuneGrid = gbmGrid)
gbmPredict = predict(gbmFit2, testing)
confusionMatrix(table(gbmPredict, testing$WS))
cm3 = confusionMatrix(gbmPredict, testing$WS)
cm3$byClass

summary(df1$pWAR)

### 2020 DATA TO TEST

summary(df.model)

head(df$`Tm/Frch`,30)

CurrentSeason =read_excel("C:/Users/Vignesh/Documents/2020 DAta.xlsx", 
                                             sheet = "Sheet1", col_names = T)

currentData = CurrentSeason

summary(currentData)
currentData = as.data.frame(currentData)

currentData[,8:13] = NULL
currentData$TEAM = as.factor(currentData$TEAM)

predictREAL = predict(gbmFit2, currentData) ## GBM
summary(predictREAL)
currentData$WinnerGBM = predictREAL
head(currentData,30)

predictREAL1 = predict(rf, currentData)
summary(predictREAL1)
head(predictREAL1) ## OAK WINNER
currentData$WinnerRF = predictREAL1
head(currentData,30)

predictREAL2 = predict(model1, currentData)
summary(predictREAL2)
currentData$WinnerDTree = predictREAL2
head(currentData,30)

predictREAL3 = predict(svmRadial, currentData)
summary(predictREAL3)
currentData$WinnerSVMR = predictREAL3
head(currentData,30)


predictREAL4 = predict(svm_Radial_Grid, currentData)
summary(predictREAL4)
currentData$WinnerSVMwCost = predictREAL4
head(currentData,30)

predictREAL5 = predict(svmLinear, currentData)
summary(predictREAL5)

summary(gbmFit2)
