setwd("D:/PML")  
Mytrainingset <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
Mytestingset  <- read.csv("pml-testing.csv",  na.strings = c("NA", "#DIV/0!", ""))
str(Mytrainingset, list.len=18)
table(Mytrainingset$classe)
prop.table(table(Mytrainingset$user_name, training$classe), 1)
prop.table(table(Mytrainingset$classe))
trainingset <- Mytrainingset[, 7:160]
testingset  <- Mytestingset[, 7:160]
is_data  <- apply(!is.na(trainingset), 2, sum) > 19621  
trainingset <- trainingset[, is_data]
testingset  <- testingset[, is_data]
library(caret)
set.seed(3141592)
train <- createDataPartition(y=trainingset$classe, p=0.60, list=FALSE)
trainf  <- trainingset[train,]
trains  <- trainingset[-train,]
dim(trainf)
dim(trains)
nzv_cols <- nearZeroVar(trainf)
if(length(nzv_cols) > 0) {
  trainf <- trainf[, -nzv_cols]
  trains <- trains[, -nzv_cols]
}
dim(trainf)
dim(trains)
library(randomForest)
set.seed(3141592)
Mymodel <- randomForest(classe~., data=trainf, importance=TRUE, ntree=100)
varImpPlot(Mymodel)
crl = cor(trainf[,c("yaw_belt","roll_belt","num_window","pitch_belt","magnet_dumbbell_z","magnet_dumbbell_y","pitch_forearm","accel_dumbbell_y","roll_arm","roll_forearm")])
diag(crl) <- 0
which(abs(crl)>0.75, arr.ind=TRUE)
cor(trainf$roll_belt, trainf$yaw_belt)
qplot(roll_belt, magnet_dumbbell_y, colour=classe, data=trainf)
library(rpart.plot)
Mymodel <- rpart(classe~., data=trainf, method="class")
prp(Mymodel)
set.seed(3141592)
Mymodel <- train(classe~roll_belt+num_window+pitch_belt+magnet_dumbbell_y+magnet_dumbbell_z+pitch_forearm+accel_dumbbell_y+roll_arm+roll_forearm,data=trainf, method="rf", trControl=trainControl(method="cv",number=2), prox=TRUE, verbose=TRUE, allowParallel=TRUE)
saveRDS(Mymodel, "predictionmodel.Rds")
Model <- readRDS("predictionmodel.Rds")
predictions <- predict(Model, newdata=trains)
confusionMat <- confusionMatrix(predictions, trains$classe)
confusionMat
missClass = function(values, predicted) {
  sum(predicted != values) / length(values)
}
outS_errorRate = missClass(trains$classe, predictions)
outS_errorRate

