# Abigail Basener
# 12/20/2022


library(neuralnet)
library(stringr)
library(MASS)
library(ISLR)
library(randomForest)
library(datasets)
library(caTools)
library(party)
library(dplyr)
library(magrittr)
library(tree)
library(randomForest)
library(gbm)
mydata <- read.csv("C:/DataPath/DataWaterloo.csv",header=T)

# ===== Load & Clean Data =====
set.seed (24)
mydata <- mydata[1:7]
mydata[7] <- mydata[7]/mydata[3]
mydata <- unique(mydata)
mydata[c('Type', 'Level')] <- str_split_fixed(mydata$course_code, ' ', 2)
dim(mydata)
mydata[c('UseLiked')] <- as.numeric(unlist(mydata[4]))/as.numeric(unlist(mydata[6]))
mydata$Level = as.numeric(mydata$Level)
mydata <-  na.omit(mydata)
str(mydata)
mydataNN <- mydata[ , c('num_ratings','useful','easy','liked','num_reviews','Level')]
# add Type as numarical value
mydataNNT <- mydataNN
mydataNNT[c('Type')] <- mydataNN$useful
Dept <- unique(mydata$Type)
for (i in 1:length(mydataNNT$Type)){
  mydataNNT$Type[i] <- which(Dept==mydata$Type[i])
}

print(mydata[which.max(mydata$Level),2])
print("-- Course witht the highest useful rating --")
mydata[which.max(mydata$Level),2]

# -- KEY --
# mydata is everything + UsedLike ratio
# mydataNN is all numarical preamitors not UsedLike ratio
# mydataNNT is mydataNN but with Type as a numarical
# Dept is a list of the departments that can decode Type above

# ===== Out put Error rates for each method =====
LDE <- c(0,0) 
RFE <- c(0,0) 
NNE <- c(0,0)
sink("my_data.txt")
print(paste0("--- Without Department Data ---"))
LDE[1] <- Linear(mydataNN)
print(paste0("Linear: ",round(LDE[1],4)))
RFE[1] <- Tree(mydataNN)
print(paste0("Tree: ",round(RFE[1],4)))
NNE[1] <- NN(mydataNN)
print(paste0("Neural Network: ",round(NNE[1],4)))
print(paste0("--- With All Department Data ---"))
print(" ")
LDE[2] <- Linear(mydataNNT)
print(paste0("Linear: ",round(LDE[2],4)))
RFE[2] <- Tree(mydataNNT)
print(paste0("Tree: ",round(RFE[2],4)))
NNE[2] <- NN(mydataNNT)
print(paste0("Neural Network: ",round(NNE[2],4)))
print(paste0("--- By Department* ---"))
print(paste0("There are ",length(Dept)," departments"))
print(" ")
# --- Get errors by departments for each method ---
n <- length(Dept)+3
for (i in 3:n){
  # Get data set for dept
  newData <- subset(mydataNNT, Type == i)
  newData <- newData[ , c('num_ratings','useful','easy','liked','num_reviews','Level')]
  # Get/Store Error rates
  sld <- dim(newData)
  if (sld[1] > 20){
    print(paste0("--- ", Dept[i], " ---"))
    LDE[i] <- Linear(newData)
    print(paste0("Linear: ",round(LDE[i],4)))
    RFE[i] <- Tree(newData)
    print(paste0("Tree: ",round(RFE[i],4)))
    NNE[i] <- NN(newData)
    print(paste0("Neural Network: ",round(NNE[i],4)))
    print(" ")
  }else{
    NNE[i]<-0
    RFE[i]<-0
    LDE[i]<-0
  }
}
print("*Department must have more than 20 classes")
sink()

# === Creat plots of error ===
LDEa <- c(0,0) 
RFEa <- c(0,0) 
NNEa <- c(0,0) 
Depa <- c(0,0) 
require(ggplot2)

count <- 1
for (i in 1:length(LDE)){
  if(LDE[i]!= 0){
    LDEa[count] <- LDE[i]
    RFEa[count] <- RFE[i]
    NNEa[count] <- NNE[i]
    count <- count + 1
  }
}
for(i in 3:length(LDE)){
  Depa[i] <- Dept[i]
}
Depa[1] <- "W/O Dept"
Depa[2] <- "With Dept"

# Plot Chart
plot(LDEa,col = "blue",type = "b",ylim = c(0,max(LDEa)),
     axes=FALSE, ylab = "MSE", xlab = " ",main ="Methods MSE Rates for Departments")
lines(RFEa,col = "green",type = "b")
lines(NNEa,col = "red",type = "b")
grid(nx = length(Depa), ny = NULL, lty = 3, col = "gray", lwd = 1)
legend("topright", c("Linear", "Tree","Neural Network"), fill = c("blue","green","red"))
axis(1, 1:(length(Depa)), Depa, col.axis="black",las = 2)#  main = "Vertical")
axis(2)


#  ===== Functions for the Methods =====

# Random Forest
Tree = function(data){
  set.seed (24)
  # ----- Tree -----
  train <- sample(1:nrow(data), nrow(data)/2)
  mydata.test <- data[-train, "liked"]
  # Bagging RF
  rf.Class <- randomForest(data$liked ~ ., data = data,
                           subset = train , mtry = 2, importance = TRUE, ntree = 30)
  yhat.bag <- predict(rf.Class , newdata = data[-train , ])
  mean((yhat.bag - mydata.test)^2)
  OTE <- mean((yhat.bag - mydata.test)^2)
# Print Tree (comentted out)
  #importance(rf.Class)
  #varImpPlot(rf.Class)
  #plot(tree.mydata)
  #text(tree.mydata , pretty = 0)
  return(OTE)
}

# Linear Regretoin 
Linear = function(data){
  set.seed (24)
  # ----- Linear -----
  train <- sample(1:nrow(data), nrow(data)/2)
  lm.fit1 <- lm(data[train, ]$liked ~ ., data[train , ])
  yhat <- predict(lm.fit1 , newdata = data[-train , ])
  LDE <- mean((yhat - data[-train , ]$liked)^2)
# Print Regretoin  (comentted out)
  #plot(data$useful,data$liked, main = "Usful Vs Liked with our Model")
  #abline(lm.fit1)
  return(LDE)
}

# Neural Network
NN = function(data){
  set.seed (24)
  inp <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
  training_data <- data[inp==1, ]
  test_data <- data[inp==2, ]
  # Make NN
  n <- neuralnet(data$liked ~ .,
                 data = training_data,
                 hidden = 6,
                 err.fct = "sse",
                 linear.output = FALSE,
                 lifesign = 'full',
                 rep = 2,
                 algorithm = "rprop+",
                 stepmax = 100000)
  n$result.matrix
  summary(n)
  # Test NN
  output <- compute(n, rep = 1, test_data[, -1])
  p1 <- output$net.result
  pred1 <- ifelse(p1 > 0.5, 1, 0)
  NNE <- mean((p1-test_data$liked)^2)
# Print Network  (comentted out)
  #plot(n, rep = 1)
  return(NNE)
}
