#### Tuning Parameters ####

## Reading data
data <- read.table(pipe("pbpaste"), header=T)
data2 <- read.table(pipe("pbpaste"), header=T)

## Boxplot (Relu vs. Selu vs. Sigmoid)
boxplot(auc~activation, data=data2, main="AUC with regard to Activation", 
        xlab="Activation Function", ylab="ROC_AUC score",
        col=(c("gold","darkgreen", "darkblue")))

## Wilcoxon Test (Relu vs. Selu)
data2_no_sigmoid <- data2[data2$activation != "sigmoid",]
wilcox.test(auc~activation, data=data2_no_sigmoid, paired=F)
## Wilcoxon Test (SGD vs. Adam)
wilcox.test(auc~optimizer, data=data2, paired=F)


## Boxplot (Adam vs. SGD)
boxplot(logauc~optimizer, data=data2, main="AUC with regard to Optimzer", 
        xlab="Optimizer", ylab="ROC_AUC score",
        col=(c("gold","darkgreen")))
boxplot(log(auc)~optimizer, data=data2, main="AUC with regard to Optimzer", 
        xlab="Optimizer", ylab="ROC_AUC score",
        col=(c("gold","darkgreen")))

## Plot the lines chart (AUC vs. Units).
unique_units <- sort(unique(data$num_unit))
unique_units <- sort(unique(data2$units))
mean_auc <- matrix(nrow=4, ncol=3)
sd_auc <- matrix(nrow=4, ncol=3)
for (j in 1:3) {
  mean_auc[1, j] <- mean(data2$auc[data2$units==unique_units[j] & data2$activation=="relu" & data2$optimizer=="SGD"])
  sd_auc[1, j] <- sd(data2$auc[data2$units==unique_units[j] & data2$activation=="relu" & data2$optimizer=="SGD"])
  mean_auc[2, j] <- mean(data2$auc[data2$units==unique_units[j] & data2$activation=="selu" & data2$optimizer=="SGD"])
  sd_auc[2, j] <- sd(data2$auc[data2$units==unique_units[j] & data2$activation=="selu" & data2$optimizer=="SGD"])
  mean_auc[3, j] <- mean(data2$auc[data2$units==unique_units[j] & data2$activation=="selu" & data2$optimizer=="Adam"])
  sd_auc[3, j] <- sd(data2$auc[data2$units==unique_units[j] & data2$activation=="selu" & data2$optimizer=="Adam"])
  mean_auc[4, j] <- mean(data2$auc[data2$units==unique_units[j] & data2$activation=="relu" & data2$optimizer=="Adam"])
  sd_auc[4, j] <- sd(data2$auc[data2$units==unique_units[j] & data2$activation=="relu" & data2$optimizer=="Adam"])
}

plot(x=unique_units, y=mean_auc[1,], col = "red", xlab = "Number of Units", ylab = "ROC_AUC Score", 
     ylim=c(0.3, 1), main = "AUC vs. Number of Units", type="o")
arrows(x0=unique_units, x1=unique_units, y0=mean_auc[1,]-sd_auc[1,],
       y1=mean_auc[1,]+sd_auc[1,], code=3, angle=90, col="red", lwd=2)
lines(x=unique_units, y=mean_auc[2,], type = "o", col = "blue")
arrows(x0=unique_units, x1=unique_units, y0=mean_auc[2,]-sd_auc[2,],
       y1=mean_auc[2,]+sd_auc[2,], code=3, angle=90, col="blue")
lines(x=unique_units, y=mean_auc[3,], type = "o", col = "darkgreen")
arrows(x0=unique_units, x1=unique_units, y0=mean_auc[3,]-sd_auc[3,],
       y1=mean_auc[3,]+sd_auc[3,], code=3, angle=90, col="darkgreen")
lines(x=unique_units, y=mean_auc[4,], type = "o", col = "gold")
arrows(x0=unique_units, x1=unique_units, y0=mean_auc[4,]-sd_auc[4,],
       y1=mean_auc[4,]+sd_auc[4,], code=3, angle=90, col="gold", lw=2)
legend(x="topright", legend=c("relu & SGD", "selu & SGD", "selu & Adam", "relu & Adam"),
       fill = c("red", "blue", "darkgreen", "gold"),
       bg="lightgrey", bty="n")


## Plot the lines chart (AUC vs. Learning rate).
unique_lr <- sort(unique(data2$learning_rate))
mean_auc <- matrix(nrow=4, ncol=5)
sd_auc <- matrix(nrow=4, ncol=5)
for (j in 1:5) {
  mean_auc[1, j] <- mean(data2$auc[data2$learning_rate==unique_lr[j] & data2$activation=="relu" & data2$optimizer=="SGD"])
  sd_auc[1, j] <- sd(data2$auc[data2$learning_rate==unique_lr[j] & data2$activation=="relu" & data2$optimizer=="SGD"])
  
  mean_auc[2, j] <- mean(data2$auc[data2$learning_rate==unique_lr[j] & data2$activation=="selu" & data2$optimizer=="SGD"])
  sd_auc[2, j] <- sd(data2$auc[data2$learning_rate==unique_lr[j] & data2$activation=="selu" & data2$optimizer=="SGD"])
  
  mean_auc[3, j] <- mean(data2$auc[data2$learning_rate==unique_lr[j] & data2$activation=="selu" & data2$optimizer=="Adam"])
  sd_auc[3, j] <- sd(data2$auc[data2$learning_rate==unique_lr[j] & data2$activation=="selu" & data2$optimizer=="Adam"])
  
  mean_auc[4, j] <- mean(data2$auc[data2$learning_rate==unique_lr[j] & data2$activation=="relu" & data2$optimizer=="Adam"])
  sd_auc[4, j] <- sd(data2$auc[data2$learning_rate==unique_lr[j] & data2$activation=="relu" & data2$optimizer=="Adam"])
}

mean_auc[4,1] = 0.6565408
sd_auc[4, 3] = 0.0122
sd_auc[4, 4] = 0.0040
sd_auc[4, 5] = 0.0002

first_color <- rgb(17, 219, 161, alpha=255*0.4, maxColorValue=255)
plot(x=unique_lr, y=mean_auc[1,], col=first_color, xlab = "Learning Rate", ylab = "ROC_AUC Score", 
     ylim=c(0.3, 1), main = "AUC vs. Learning Rate", type="b", lwd=5)
arrows(x0=unique_lr, x1=unique_lr, y0=mean_auc[1,],
       y1=mean_auc[1,]+sd_auc[1,], code=2, angle=90, col=first_color, lwd=5)

second_color <- rgb(219, 104, 17, alpha=255, maxColorValue=255)
lines(x=unique_lr, y=mean_auc[2,], type="b", col=second_color, lwd=2)
arrows(x0=unique_lr, x1=unique_lr, y0=mean_auc[2,],
       y1=mean_auc[2,]+sd_auc[2,], code=2, angle=90, col=second_color, lwd=2)

third_color <- rgb(17, 101, 219, alpha=255*0.4, maxColorValue=255)
lines(x=unique_lr, y=mean_auc[3,], type="b", col=third_color, lwd=5)
arrows(x0=unique_lr, x1=unique_lr, y0=mean_auc[3,],
       y1=mean_auc[3,]+sd_auc[3,], code=2, angle=90, col=third_color, lwd= 5)

fourth_color <- rgb(219, 17, 192, alpha=255, maxColorValue=255)
lines(x=unique_lr, y=mean_auc[4,], type="b", lwd=2, col=fourth_color)
arrows(x0=unique_lr, x1=unique_lr, y0=mean_auc[4,],
       y1=mean_auc[4,]+sd_auc[4,], code=2, angle=90, col=fourth_color, lwd=2)
legend(x="topright", legend=c("relu & SGD", "selu & SGD", "selu & Adam", "relu & Adam"),
       fill = c(first_color, second_color, third_color, fourth_color),
       bg="lightgrey", bty="n")

hist(data2$lr, col='darkgreen', density = F,
     main="Frequency of units with AUC >= 0.8", xlab="Number of Units", nclass = c(512, 1024, 2048))

lines(x=sigmoid.lr, y=sigmoid.auc, type = "o", col = "gold")
dev.off()
