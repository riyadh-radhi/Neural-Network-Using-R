rm(list = ls())


library(MASS)
library(dplyr)
library(scales)
library(caret)
library(neuralnet)


df <- read.csv("letterdata.csv")

#1 Filtering based on name/surenmae
#My name is "Riyadh Radhi", so my last name has same letters as my name 

myName_data <- filter(df, letter == "R" | letter == "I" | letter == "Y" | letter == "A" | letter == "D" | letter == "H")


#2. Scaling prediction of myName_data to [0,1] range and then rounding them


myName_data[,-1] <- rescale(as.matrix(myName_data[,-1]), to= c(0,1))
myName_data[,-1] <- round(myName_data[,-1], 2)
head(myName_data)


#3 Generating variables for final nodes for each letter


myName_data$R <- myName_data$letter == "R"
myName_data$I <- myName_data$letter == "I"
myName_data$Y <- myName_data$letter == "Y"
myName_data$A <- myName_data$letter == "A"
myName_data$D <- myName_data$letter == "D"
myName_data$H <- myName_data$letter == "H"


# 4 Doing set.seed and data partitioning 

set.seed(175191)

index <- createDataPartition(myName_data$letter, p = 0.7,list = F)
train <- myName_data[index,]
test <- myName_data[-index,]


#5 Training Neural Network Model

set.seed(175191)

nn <- neuralnet(R+I+Y+A+D+H ~ xbox + ybox + width+height+onpix+xbar+ybar+x2bar+y2bar+xybar+x2ybar+xy2bar+xedge+xedgey+yedge+yedgex,
                data = train, linear.output = F,
                hidden = 3, act.fct = "logistic",
                threshold = 0.2)



# This plot is just to show our three hidden layers with three nodes each

plot(nn, rep="best")


#6 Predict Class Labels on the test set letters
#Please note that I will not use the function compute to do the prediction 
#because it has been adviced to use the new function called predict for neural network prediction 

nn_predict <- predict(nn, test)
head(nn_predict)

#7 Before I will be able to construct the confusion matrix, I will need to transfer my prediction 
#into factor, and also transfer each index number to its corrsponding letter of my name

predicted_class <- apply(nn_predict,1,which.max)-1


predicted_class[which(predicted_class == "0")] <- "R"
predicted_class[which(predicted_class == "1")] <- "I"
predicted_class[which(predicted_class == "2")] <- "Y"
predicted_class[which(predicted_class == "3")] <- "A"
predicted_class[which(predicted_class == "4")] <- "D"
predicted_class[which(predicted_class == "5")] <- "H"

predicted_class <- as.factor(predicted_class)

#8 Constructing the confusion matrix 

nn_confusionMatrix <- confusionMatrix(predicted_class,test$letter) #Remove Warning

# Accuracy 
nn_confusionMatrix$overall

#my table of letter existed vs letter predicted correctly 
nn_confusionMatrix$table


#9 Improving the accuracy of the model 

set.seed(175191)

nn <- neuralnet(R+I+Y+A+D+H ~ xbox + ybox + width+height+onpix+xbar+ybar+x2bar+y2bar+xybar+x2ybar+xy2bar+xedge+xedgey+yedge+yedgex,
                data = train, linear.output = F,
                hidden = 5, act.fct = "logistic",
                threshold = 0.2)
