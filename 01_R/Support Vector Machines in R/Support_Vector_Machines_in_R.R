

# Chapter 1:  Introduction
# Visualizing a sugar content dataset
#load ggplot
library(ggplot2)

df <- read.csv('sugar.csv')

#print variable names
names(df)

#build plot
plot_df <- ggplot(data = df, aes(x = sugar_content, y = c(0))) + 
  geom_point() + 
  geom_text(label = df$sugar_content, size = 2.5, vjust = 2, hjust = 0.5)

#display plot
plot_df


# Find the maximal margin separator
#The maximal margin separator is at the midpoint of the two extreme points in each cluster.
mm_separator <- (8.9 + 10)/2



# Visualize the maximal margin separator
#create data frame
separator <- data.frame(sep = c(mm_separator))

#add ggplot layer 
plot_sep <- plot_df + geom_point(data = separator, x = separator$sep, y = c(0), color = "blue ", size = 4)

#display plot
plot_sep







# Generate a 2d uniformly distributed dataset.
#set seed
set.seed(42)

#set number of data points. 
n <- 600

#Generate data frame with two uniformly distributed predictors lying between 0 and 1.
df <- data.frame(x1 = runif(n), 
                 x2 = runif(n))



# Create a decision boundary
#classify data points depending on location
df$y <- factor(ifelse(df$x2 - 1.4*df$x1 < 0, -1, 1), 
               levels = c(-1, 1))



# Introduce a margin in the dataset
#set margin
delta <-  0.07

# retain only those points that lie outside the margin
df1 <- df[abs(1.4*df$x1 - df$x2) > delta, ]

#build plot
plot_margins <- ggplot(data = df1, aes(x = x1, y = x2, color = y)) + geom_point() + 
  scale_color_manual(values = c("red", "blue")) + 
  geom_abline(slope = 1.4, intercept = 0)+
  geom_abline(slope = 1.4, intercept = delta, linetype = "dashed") +
  geom_abline(slope = 1.4, intercept = -delta, linetype = "dashed")

#display plot 
plot_margins





# Chapter 2: Support Vector Classifiers - Linear Kernels
# Creating training and test datasets
#split train and test data in an 80/20 proportion
df[, "train"] <- ifelse(runif(nrow(df))<0.8, 1, 0)

#assign training rows to data frame trainset
trainset <- df[df$train == 1, ]
#assign test rows to data frame testset
testset <- df[df$train == 0, ]

#find index of "train" column
trainColNum <- grep("train", names(df))

#remove "train" column from train and test dataset
trainset <- trainset[, -trainColNum]
testset <- testset[, -trainColNum]



# Building a linear SVM classifier
library(e1071)

#build svm model, setting required parameters
svm_model<- svm(y ~ ., 
                data = trainset, 
                type = "C-classification", 
                kernel = "linear", 
                scale = FALSE)


# Exploring the model and calculating accuracy
#list components of model
names(svm_model)

#list values of the SV, index and rho
svm_model$SV
svm_model$index
svm_model$rho

#compute training accuracy
pred_train <- predict(svm_model, trainset)
mean(pred_train == trainset$y)

#compute test accuracy
pred_test <- predict(svm_model, testset)
mean(pred_test == testset$y)



# Visualizing support vectors using ggplot
#build scatter plot of training dataset
scatter_plot <- ggplot(data = trainset, aes(x = x1, y = x2, color = y)) + 
  geom_point() + 
  scale_color_manual(values = c("red", "blue"))

#add plot layer marking out the support vectors 
layered_plot <- 
  scatter_plot + geom_point(data = trainset[svm_model$index, ], aes(x = x1, y = x2), color = "purple", size = 4, alpha = 0.5)

#display plot
layered_plot



# Visualizing decision & margin bounds using `ggplot2`
#build weight vector
w <- t(svm_model$coefs) %*% svm_model$SV

#calculate slope and intercept of decision boundary from weight vector and svm model
slope_1 <- -w[1]/w[2]
intercept_1 <- svm_model$rho/w[2]

#add decision boundary
plot_decision <- layered_plot + geom_abline(slope = slope_1, intercept = intercept_1) 
#add margin boundaries
plot_margins <- plot_decision + 
  geom_abline(slope = slope_1, intercept = intercept_1 - 1/w[2], linetype = "dashed")+
  geom_abline(slope = slope_1, intercept = intercept_1 + 1/w[2], linetype = "dashed")
#display plot
plot_margins





# Visualizing decision & margin bounds using `plot()`
#load required library
library(e1071)

#build svm model
svm_model<- 
  svm(y ~ ., data = trainset, type = "C-classification", 
      kernel = "linear", scale = FALSE)

#plot decision boundaries and support vectors
plot(x = svm_model, data = trainset)











# Tuning a linear SVM
#build svm model, cost = 1
svm_model_1 <- svm(y ~ .,
                     data = trainset,
                     type = "C-classification",
                     cost = 1,
                     kernel = "linear",
                     scale = FALSE)

#print model details
svm_model_1


#build svm model, cost = 100
svm_model_100 <- svm(y ~ .,
                     data = trainset,
                     type = "C-classification",
                     cost = 100,
                     kernel = "linear",
                     scale = FALSE)

#print model details
svm_model_100


w_1 <- t(svm_model_1$coefs) %*% svm_model_1$SV
slope_1 <- -w_1[1]/w_1[2]
intercept_1 <- svm_model_1$rho/w[2]


w_100 <- t(svm_model_100$coefs) %*% svm_model_100$SV
slope_100 <- -w_100[1]/w_100[2]
intercept_100 <- svm_model_100$rho/w[2]


scatter_plot <- ggplot(data = trainset, aes(x = x1, y = x2, color = y)) + 
  geom_point() + 
  scale_color_manual(values = c("red", "blue"))

#add plot layer marking out the support vectors 
train_plot <- 
  scatter_plot + geom_point(data = trainset[svm_model$index, ], aes(x = x1, y = x2), color = "purple", size = 4, alpha = 0.5)

train_plot


# Visualizing decision boundaries and margins
#add decision boundary and margins for cost = 1 to training data scatter plot
train_plot_with_margins <- train_plot + 
  geom_abline(slope = slope_1, intercept = intercept_1) +
  geom_abline(slope = slope_1, intercept = intercept_1-1/w_1[2], linetype = "dashed")+
  geom_abline(slope = slope_1, intercept = intercept_1+1/w_1[2], linetype = "dashed")

#display plot
train_plot_with_margins



#add decision boundary and margins for cost = 100 to training data scatter plot
train_plot_with_margins <- train_plot_with_margins + 
  geom_abline(slope = slope_100, intercept = intercept_100, color = "goldenrod") +
  geom_abline(slope = slope_100, intercept = intercept_100-1/w_100[2], linetype = "dashed", color = "goldenrod")+
  geom_abline(slope = slope_100, intercept = intercept_100+1/w_100[2], linetype = "dashed", color = "goldenrod")

#display plot 
train_plot_with_margins





# A multiclass classification problem
library(datasets)
data(iris)

set.seed(42)
df <- iris[,3:5]
colnames(df) <- c("x1", "x2", "y")

df[, "train"] <- ifelse(runif(nrow(iris))<0.8, 1, 0)
trainset <- df[df$train == 1, ]
testset <- df[df$train == 0, ]
trainColNum <- grep("train", names(df))
trainset <- trainset[, -trainColNum]
testset <- testset[, -trainColNum]



#load library and build svm model
library(e1071)
svm_model<- 
  svm(y ~ ., data = trainset, type = "C-classification", 
      kernel = "linear", scale = FALSE)

#compute training accuracy
pred_train <- predict(svm_model, trainset)
mean(pred_train == trainset$y)

#compute test accuracy
pred_test <- predict(svm_model, testset)
mean(pred_test == testset$y)

#plot
plot(svm_model, trainset)





# Iris redux - a more robust accuracy.
accuracy <- vector(mode="numeric", length=100)
#calculate accuracy for n distinct 80/20 train/test partitions
for (i in 1:100){ 
  iris[, "train"] <- ifelse(runif(nrow(iris)) < 0.8, 1, 0)
  trainColNum <- grep("train", names(iris))
  trainset <- iris[iris$train == 1, -trainColNum]
  testset <- iris[iris$train == 0, -trainColNum]
  svm_model <- svm(Species~ ., data = trainset, 
                   type = "C-classification", kernel = "linear")
  pred_test <- predict(svm_model, testset)
  accuracy[i] <- mean(pred_test == testset$Species)
}

#mean and standard deviation of accuracy
mean(accuracy) 
sd(accuracy)