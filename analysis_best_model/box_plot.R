#### Snow Days ####

data <- read.table(pipe("pbpaste"), header=T)
summary(snow.days)

t.test(data$auc~data$activation, mu=0)
levels(data$activation)

## Relu vs. Selu
boxplot(auc~activation, data=data, main="AUC in regard to Activation", 
         xlab="Activation Function", ylab="ROC_AUC score",
         col=(c("gold","darkgreen", "darkblue")))
relu.auc <- data$auc[data$activation == "relu"]
selu.auc <- data$auc[data$activation == "selu"]
t.test(x=relu.auc, y=selu.auc, mu=0)

## Adam vs. SGD
boxplot(auc~optimizer, data=data, main="AUC in regard to Activation", 
        xlab="Activation Function", ylab="ROC_AUC score",
        col=(c("gold","darkgreen")))
adam.auc <- data$auc[data$optimizer == "adam"]
SGD.auc <- data$auc[data$optimizer == "SGD"]
t.test(x=relu.auc, y=selu.auc, mu=0)

length(data$activation[data$activation == "relu"])
length(data$activation[data$activation == "selu"])
length(data$activation[data$activation == "sigmoid"])


plot(data$learning_rate, y=data$auc, type="b")


#### Drawing 3 lines on the same chart ####
# Create the data for the chart.
relu.auc <- data$auc[data$activation=="relu"]
relu.lr <- data$learning_rate[data$activation=="relu"]
selu.auc <- data$auc[data$activation=="selu"]
selu.lr <- data$learning_rate[data$activation=="selu"]
sigmoid.auc <- data$auc[data$activation=="sigmoid"]
sigmoid.lr <- data$learning_rate[data$activation=="sigmoid"]

# Give the chart file a name.
png(file = "lines_charts.jpg")
# Plot the bar chart.
plot(x=relu.lr, y=relu.auc, col = "red", xlab = "Learning Rate", ylab = "ROC_AUC Score", 
     main = "AUC vs. LR")
# Plot the bar chart.
plot(x=selu.lr, y=selu.auc, col = "blue", xlab = "Learning Rate", ylab = "ROC_AUC Score", 
     main = "AUC vs. LR")
# Plot the bar chart.
plot(x=sigmoid.lr, y=sigmoid.auc, col = "gold", xlab = "Learning Rate", ylab = "ROC_AUC Score", 
     main = "AUC vs. LR")

lines(x=selu.lr, y=selu.auc, type = "o", col = "blue")
lines(x=sigmoid.lr, y=sigmoid.auc, type = "o", col = "gold")
# Save the file.
dev.off()