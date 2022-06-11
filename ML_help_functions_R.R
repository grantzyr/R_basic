"
1. Generate random numbers from a distribution
2. Confidence Intervals of Binomial Data
3. Regular Linear Regression Models
4. Simple Logistic Regression
5. get MSPE
6. set Folds
7. simple plots
8. Basic Regressions
9. K-fold CV and boxplots
10. Stepwise
11. Ridge
12. LASSO
13. Partial Least Squares Analysis (PLS)
14. GAM
15. Pruned tree
16. PPR
17. NN
18. Splines - polynomials
# classification
19. scale 
20. KNN
21. Logistic Regression
22. LASSO logistic regressicon
23. Discriminant Analysis LDA 
24. Discriminant Analysis QDA
25. GAM
26. Naive Bayes
27. Classification Tree 
28. Random Forest
29. NN 
30. SVM
"


library(dplyr)
library(rgl)
library(leaps)
library(caret) 
library(MASS)
library(glmnet)
library(nnet)
library(reshape)
####################################################################
# 1. Generate random numbers from a distribution:
# I'll use the normal distribution for example
set.seed(123) #Setting the seed allows fixes the random result (each
          # time the code is run the same result will be obtained)
x <- rnorm(1000,0,1) #Generate the random numbers from N(0,1)

x #view the entire result
head(x) #view the first 6 results
summary(x) #view some basic summary statistics of the values

# Now let's plot the histogram of these values
?hist()
hist(x, freq=F)
curve(dnorm(x,0,1), add = T, col = "red")

#What if we want to know P(X > 2.5)?
pnorm(2.5, 0,1, lower.tail = F)

#To find the quantile, say the 95% critical value (one-tail):
qnorm(0.95,0,1)

#To mark the 0.95 quantile on the histogram
abline(v=qnorm(0.95,0.1), col = "green")

########################################################################
# 2. Confidence Intervals of Binomial Data

#The following example will provide a way to check values of 
# various confidence intervals when dealing with binomial random variables
#install.packages("binom")
library(binom)

#Consider the scenario from the MLE Example
n=50
x = 20
ci <- binom.confint(x,n,conf.level = 0.95,methods="all")
ci

#The usual CI is the Wald CI, which is the asymptotic interval in the output
ci[2, c(1,5,6)] #[row, column]

#Compare this to the confidence interval
# from the formula in the tutorial slides
phat = x/n
LowerLimit = phat - qnorm(0.975)*sqrt(((phat*(1-phat))/n))
UpperLimit = phat + qnorm(0.975)*sqrt(((phat*(1-phat))/n))
c(LowerLimit,UpperLimit)
ci[2, c(5,6)] 

########################################################################
# 3. Regular Linear Regression Models
data(iris)
head(iris)

mod1 <- lm(Petal.Width~Sepal.Length+Sepal.Width+Petal.Length,
       data = iris)
summary(mod1)
plot(mod1)

##########################################################
# 4. Simple Logistic Regression

#Aggregated Data Example:
set.seed(475)
SuccessesPerUnit <- c(seq(0,9,1),seq(0,9,1),seq(0,9,1),seq(0,9,1))
TrialsPerUnit <- rep(10,40)
ExplanatoryVariable <- rnorm(40,4,1)
Data <- data.frame(Successes = SuccessesPerUnit,
               Trials = TrialsPerUnit,
               ExplanatoryVariable = ExplanatoryVariable)
Data
######Method 1: Indicates occurrence of Success
Data$BinResp <- c()
for( i in 1:length(Data$Successes)){
if(Data$Successes[i] > 0){
Data$BinResp[i] = 1
}else{
Data$BinResp[i] = 0
}
}
head(Data)
Mod1 <- glm(BinResp~ExplanatoryVariable, family=binomial,data=Data)
summary(Mod1)

######Method 2: Uses probability of Success
Mod2 <- glm((Successes/Trials)~ExplanatoryVariable,
        family = binomial, weights = Trials, data = Data)
summary(Mod2)

#Non-Aggregated Data Example:
placekick <- read.csv("Placekick.csv", header=T)
head(placekick)
LogMod <- glm(formula=good ~ distance,
           family=binomial(link = logit), data=placekick)
LogMod #Returns the model for logit(pi)
summary(LogMod) #Returns the model for logit(pi)
coef(LogMod) #Access the coefficients
vcov(LogMod) #Access the variance-covariance matrix

#Confidence Intervals for Parameters (Betas)##########################
#WaldCI
confint.default(LogMod)

#Likelihood Ratio CI
confint(LogMod)
confint(LogMod, parm = "distance") #To get LR CI for specific parameter

#Wilson Score CI
binom.confint(sum(placekick$good),length(placekick$good),method = "wilson")

#Hypothesis Test for Parameters (Betas) Note: Ho: Beta=0, Ha: Beta Not = 0

#Wald Test
summary(LogMod)#Values are provided in this output

#Likelihood Ratio Test
library(car)
Anova(LogMod, test = "LR")

#####################################################
# 5. get MSPE
get.MSPE = function(Y, Y.hat) {
residuals = Y - Y.hat
resid.sq = residuals ^ 2
SSPE = sum(resid.sq)
MSPE = SSPE / length(Y)
return(MSPE)
}
######################################################
# 6. get folds
get.folds = function(n, K) {
### Get the appropriate number of fold labels
n.fold = ceiling(n / K) # Number of observations per fold (rounded up)
fold.ids.raw = rep(1:K, times = n.fold) # Generate extra labels
fold.ids = fold.ids.raw[1:n] # Keep only the correct number of labels

### Shuffle the fold labels
folds.rand = fold.ids[sample.int(n)]

return(folds.rand)
}

n = nrow(prostate) 
folds = get.folds(n/K)

############################################################
# 7. Simple plots
# plots between each variable
pairs(prostate)

# plot between two variables 
plot(y = prostate$lpsa, x = prostate$lcavol) 
# to label the plot, use xlab = '', ylab = '', main = ''
# add the regression fit line to the plot
fit <- lm(lpsa~lcavol, data=prostate)
abline(fit) # run this line with the plot code together 

# 3d plot 
library(rgl)
open3d()
plot3d(lpsa ~ lcavol + pgg45, data=prostate, col='blue')

mod2 <- lm(lpsa ~ lcavol + pgg45, data = prostate)
x1 <- seq(from=-2, to=4, by=.05)
xy1 <- data.frame(expand.grid(lcavol=seq(from=-2, to=4, by=.05), 
                          pgg45=seq(from=0, to=100, by=.5)))
pred <- predict(mod2 ,newdata=xy1)
surface = matrix(pred, nrow=length(x1))

open3d()
persp3d(x = seq(from=-2, to=4, by=.05), y = seq(from=0, to=100, by=.5), 
    z = surface, col = "orange", xlab="lcavol", ylab="pgg45", 
    zlab="Predicted lpsa")
points3d(prostate$lpsa ~ prostate$lcavol + prostate$pgg45, col="blue")
#######################################################################
# 8. Basic regressions
# Simple linear regression 
mod.vol <- lm(lpsa ~ lcavol, data = prostate)
summary(mod.vol) # all information 
coef(mod.vol) # only coefficients information 
summary(mod.vol)$coef[2,1] # extract coef, i.e beta1 
summary(mod.vol)$coef[2,2] # extract se of beta1 

# Transformations
#  Powers: X^2, X^3, etc.
#  Logs: Natural: log()    Base 10: log10()
#  Square roots: sqrt() or X^(1/2) (USE PARENTHESES AROUND POWER)!!!)
#  For other roots, use powers
#  Inverse: 1/X or X^-1
# Interaction
mod.cp = lm(lpsa ~ lcavol*pgg45, data=prostate) 

# Create new variables 
AQ$TWcp <- AQ$Temp*AQ$Wind
AQ$TWrat <- AQ$Temp/AQ$Wind

# Create dummy variable 
# use Insurance.csv as example, read in as ins
library(caret)
ins$zone = as.factor(ins$zone)
ins$make = as.factor(ins$make)
ins.dv <- data.frame(predict(dummyVars("~.", data=ins), newdata = ins))
dim(ins.dv) 
#####################################################################
# 9. K-fold CV and boxplots
# set.seed() # your own choice of seed number 
n = nrow(prostate)

# K-fold CV 
K = 10 # or whatever # you want

MSPEs = matrix(NA, nrow = K, ncol = 1) # store the MSPEs, ncol = # of models
colnames(MSPEs) = c("..") # name the columns 

# for() loop: see different models in 'Variable selection' section below 

MSPEs # MSPEs of each of the 10-fold 
colMeans(MSPEs) 

# Confidence interval 
(MSPE.mean = apply(X=MSPEs, MARGIN=2, FUN=mean))
(MSPE.sd = apply(X=MSPEs, MARGIN=2, FUN=sd))
MSPE.CIl = MSPE.mean - qt(p=.975, df=R-1)*MSPE.sd/sqrt(R)
MSPE.CIu = MSPE.mean + qt(p=.975, df=R-1)*MSPE.sd/sqrt(R)
round(cbind(MSPE.CIl, MSPE.CIu),2)

# boxplots 
boxplot(MSPEs, las=2, ylim=c(0,2.5),
    main="MSPE \n Random data splits")
low.s = apply(MSPEs, 1, min) 
boxplot(MSPEs/low.s, las=2,
    main="Relative MSPE \n Random data splits")
boxplot(MSPEs/low.s, las=2, ylim=c(1,1.5),
    main="Focused Relative MSPE \n Random data splits")

###############################################################
# Variable Selection
# 10. Stepwise
library(leaps)

# CV
for(i in 1:K){
#############################
data.train = AQ[folds != i,]# replace AQ with your dataset 
data.valid = AQ[folds == i,]#
n.train = nrow(data.train)  #
Y.train = data.train$Ozone  #
Y.valid = data.valid$Ozone  #
############################# 
# These 5 lines are one-set preparation code for every for() loop

# Stepwise
fit.start = lm(Ozone ~ 1, data = data.train)
fit.end = lm(Ozone ~ ., data = data.train)
step.BIC = step(fit.start, list(upper = fit.end), k = log(nrow(data.train)),
trace = 0)

pred.step.BIC = predict(step.BIC, data.valid)
MSPE.step.BIC = get.MSPE(Y.valid, pred.step.BIC)

MSPEs[i, ] = MSPE.step.BIC
}
##############################################################################
# 11. Ridge 

# make a list of lambda values
lambda.vals = seq(from=0, to=100, by=0.05)
fit.ridge <- lm.ridge(lpsa ~., lambda = lambda.vals, data=data.train)
# Show coefficient path
plot(ridge1)
select(ridge1)

# get best lambda value and its index 
# best is chosen according to smallest GCV value 
ind.min.GCV = which.min(fit.ridge$GCV)
lambda.min = lambda.vals[ind.min.GCV]
all.coefs.ridge = coef(fit.ridge)
coef.min = all.coefs.ridge[ind.min.GCV,]

# to predict lm.ridge, need to make matrix first 
matrix.valid.ridge = model.matrix(Ozone ~ ., data = data.valid)
pred.ridge = matrix.valid.ridge %*% coef.min

MSPE.ridge = get.MSPE(Y.valid, pred.ridge)
MSPEs[i, "Ridge"] = MSPE.ridge


####################################################################
# 12. LASSO

# glmnet() requires x to be in matrix class
matrix.train.raw = model.matrix(Ozone ~ ., data = data.train)
matrix.train = matrix.train.raw[,-1]

all.LASSOs = cv.glmnet(x = matrix.train, y = Y.train)
### Get both 'best' lambda values using $lambda.min and $lambda.1se
lambda.min = all.LASSOs$lambda.min
lambda.1se = all.LASSOs$lambda.1se

coef.LASSO.min = predict(all.LASSOs, s = lambda.min, type = "coef")
coef.LASSO.1se = predict(all.LASSOs, s = lambda.1se, type = "coef")

included.LASSO.min = predict(all.LASSOs, s = lambda.min, 
type = "nonzero")
included.LASSO.1se = predict(all.LASSOs, s = lambda.1se, 
type = "nonzero")

matrix.valid.LASSO.raw = model.matrix(Ozone ~ ., data = data.valid)
matrix.valid.LASSO = matrix.valid.LASSO.raw[,-1]
pred.LASSO.min = predict(all.LASSOs, newx = matrix.valid.LASSO,
s = lambda.min, type = "response")
pred.LASSO.1se = predict(all.LASSOs, newx = matrix.valid.LASSO,
s = lambda.1se, type = "response")

MSPE.LASSO.min = get.MSPE(Y.valid, pred.LASSO.min)
MSPEs[i, "LASSO-Min"] = MSPE.LASSO.min

MSPE.LASSO.1se = get.MSPE(Y.valid, pred.LASSO.1se)
MSPEs[i, "LASSO-1se"] = MSPE.LASSO.1se

########################################################################
# 13. pls
library(pls)

fit.pls = plsr(Ozone ~ ., data = data.train, validation = "CV",
segments = 10)

CV.pls = fit.pls$validation # All the CV information
PRESS.pls = CV.pls$PRESS    # Sum of squared CV residuals
CV.MSPE.pls = PRESS.pls / nrow(data.train)  # MSPE for internal CV
ind.best.pls = which.min(CV.MSPE.pls) # Optimal number of components
print(ind.best.pls)

pred.pls = predict(fit.pls, data.valid, ncomp = ind.best.pls)
MSPE.pls = get.MSPE(Y.valid, pred.pls)
MSPEs[i, "PLS"] = MSPE.pls

###########################################################################
# 14. GAM
fit.gam = gam(Ozone ~ s(Solar.R) + s(Wind) + s(Temp) + s(TWcp) + s(TWrat), 
data = data.train)

pred.gam = predict(fit.gam, data.valid)
MSPE.gam = get.MSPE(Y.valid, pred.gam)
MSPEs[i, "GAM"] = MSPE.gam

###########################################################################
# 15. pruned tree
## full tree
pr.tree2 = rpart(Ozone ~ ., method = "anova", data = data.train, cp = 0)
pred.full = predict(pr.tree2, data.valid)
MSPEs[i, "Full tree"] = get.MSPE(Y.valid, pred.full)

## min and 1se 
# Find location of minimum error
minrow <- which.min(cpt[,4])
# Take geometric mean of cp values at min error and one step up 
cplow.min <- cpt[minrow,1]
cpup.min <- ifelse(minrow==1, yes=1, no=cpt[minrow-1,1])
cp.min <- sqrt(cplow.min*cpup.min)

# Find smallest row where error is below +1SE
se.row <- min(which(cpt[,4] < cpt[minrow,4]+cpt[minrow,5]))
# Take geometric mean of cp values at min error and one step up 
cplow.1se <- cpt[se.row,1]
cpup.1se <- ifelse(se.row==1, yes=1, no=cpt[se.row-1,1])
cp.1se <- sqrt(cplow.1se*cpup.1se)

# Do pruning each way
pr2.prune.min <- prune(pr.tree2, cp=cp.min)
pr2.prune.1se <- prune(pr.tree2, cp=cp.1se)

pred.pr.min = predict(pr2.prune.min, data.valid)
MSPEs[i,"Prune_min"] = get.MSPE(Y.valid, pred.pr.min)

pred.pr.1se = predict(pr2.prune.1se, data.valid)
MSPEs[i,"Prune_1se"] = get.MSPE(Y.valid, pred.pr.1se)

#####################################################################
# 16. PPR
# see assignment for details
max.terms = 5 # number of variables in the dataset, e.g. AQ has 5

### PPR ###
### To fit PPR, we need to do another round of CV. This time, do 5-fold
K.ppr = 5
n.train = nrow(data.train)
folds.ppr = get.folds(n.train, K.ppr)

### Container to store MSPEs for each number of terms on each sub-fold
MSPEs.ppr = array(0, dim = c(K.ppr, max.terms))

for(j in 1:K.ppr){
### Split the training data.
### Be careful! We are constructing an internal validation set by 
### splitting the training set from outer CV.
train.ppr = data.train[folds.ppr != j,]
valid.ppr = data.train[folds.ppr == j,] 
Y.valid.ppr = valid.ppr$Ozone

### We need to fit several different PPR models, one for each number
### of terms. This means another for loop (make sure you use a different
### index variable for each loop).
for(l in 1:max.terms){
  ### Fit model
  fit.ppr = ppr(Ozone ~ ., data = train.ppr, 
    max.terms = max.terms, nterms = l, sm.method = "gcvspline")
  
  ### Get predictions and MSPE
  pred.ppr = predict(fit.ppr, valid.ppr)
  MSPE.ppr = get.MSPE(Y.valid.ppr, pred.ppr) # Our helper function

  ### Store MSPE. Make sure the indices match for MSPEs.ppr
  MSPEs.ppr[j,l] = MSPE.ppr
} 
}
### Get average MSPE for each number of terms
ave.MSPE.ppr = apply(MSPEs.ppr, 1, mean)

### Get optimal number of terms
best.terms = which.min(ave.MSPE.ppr)
print(best.terms)

### Fit PPR on the whole CV training set using the optimal number of terms 
fit.ppr.best = ppr(Ozone ~ ., data = data.train,
max.terms = max.terms, nterms = best.terms, sm.method = "gcvspline")

### Get predictions, MSPE and store results
pred.ppr.best = predict(fit.ppr.best, data.valid)
MSPE.ppr.best = get.MSPE(Y.valid, pred.ppr.best) # Our helper function

CV.MSPEs[i,"tuned PPR"] = MSPE.ppr.best

#######################################################################  
#  17. NN
  library(nnet)
  rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
  }
  ## see assignments for details 
  trcon = trainControl(method="repeatedcv", number=5, repeats=10,
                     returnResamp="all")
parmgrid = expand.grid(size=c(1,3,5,7,9),decay= c(0.001,0.1,0.5, 1,2))

tuned.nnet <- train(x=AQ[,-1], y=AQ[,1], method="nnet", 
                    trace=FALSE, linout=TRUE, 
                    trControl=trcon, preProcess="range", 
                    tuneGrid = parmgrid)
tuned.nnet
names(tuned.nnet)
tuned.nnet$bestTune

# make the plot 
(resamp.caret = tuned.nnet$resample[,-c(2,3)])

library(reshape)
RMSPE.caret = reshape(resamp.caret, direction="wide", v.names="RMSE",
                      idvar=c("size","decay"), timevar="Resample")


# Plot results. 
siz.dec <- paste("NN",RMSPE.caret[,1],"/",RMSPE.caret[,2])
# x11(pointsize=10)
boxplot(x=as.matrix(RMSPE.caret[,-c(1,2)]), use.cols=FALSE, names=siz.dec,
        las=2, main="caret Root-MSPE boxplot for various NNs")

# Plot RELATIVE results. 
lowt = apply(RMSPE.caret[,-c(1,2)], 2, min)

# x11(pointsize=10)
boxplot(x=t(as.matrix(RMSPE.caret[,-c(1,2)]))/lowt, las=2, 
        names=siz.dec)

#Focused 
# x11(pointsize=10)
boxplot(x=t(as.matrix(RMSPE.caret[,-c(1,2)]))/lowt, las=2, 
        names=siz.dec, ylim=c(1,2))

R=10
V=5
relMSPE = t(RMSPE.caret[,-c(1,2)])/lowt
(RRMSPE = apply(X=relMSPE, MARGIN=2, FUN=mean))
(RRMSPE.sd = apply(X=relMSPE, MARGIN=2, FUN=sd))
RRMSPE.CIl = RRMSPE - qt(p=.975, df=R*V-1)*RRMSPE.sd/sqrt(R*V)
RRMSPE.CIu = RRMSPE + qt(p=.975, df=R*V-1)*RRMSPE.sd/sqrt(R*V)
(all.rrcv = cbind(RMSPE.caret[,1:2],round(sqrt(cbind(RRMSPE,RRMSPE.CIl, RRMSPE.CIu)),2)))
all.rrcv[order(RRMSPE),]

###########################################################
# 18. splines - polynomials
library(splines) 
# polynomials 
cubic.poly <- lm(Ozone ~ poly(Temp, degree = 3), data = AQ) # cubic poly
poly.5 <- lm(Ozone ~ poly(Temp, degree = 5), data = AQ) # 5th order polynomial 

# CUBIC SPLINES with different DFs 
cs.5 <- lm(Ozone ~ bs(Temp, df=5),data = AQ) # bs means basis function splines
cs.7 <- lm(Ozone ~ bs(Temp, df=7),data = AQ) # choose your number of df
  # Get predictions
temp.sort <- data.frame(Temp = sort(AQ$Temp)) # SORT
pred.cubic.poly <- predict(cubic.poly, temp.sort)
pred.cs.5 <- predict(cs.5, temp.sort)
pred.cs.7 <- predict(cs.7, temp.sort)
  # plot
plot(x=AQ$Temp, y=AQ$Ozone, col="gray", 
     main="UP TO YOU")
legend(x=55, y=165, # where you want to put the legend
       legend=c("cubic","cs 5 DF", "cs 7 DF"), lty="solid",
       col=colors()[c(24,121,145)], lwd=2) # change the color as you wish
lines(temp.sort$Temp, pred.cubic.poly,col=colors()[24], lwd = 2)
lines(temp.sort$Temp, pred.cs.5, col=colors()[121], lwd=2)
lines(temp.sort$Temp, pred.cs.7, col=colors()[145], lwd=2)

# NATURAL SPLINES with different DFs
ns.5 <- lm(Ozone ~ ns(Temp, df=5),data = AQ) # ns means natural splines 
ns.7 <- lm(Ozone ~ ns(Temp, df=7),data = AQ)
  # Get predictions 
temp.sort <- data.frame(Temp = sort(AQ$Temp))
pred.cubic.poly <- predict(cubic.poly, temp.sort)
pred.ns.5 <- predict(ns.5, temp.sort)
pred.ns.7 <- predict(ns.7, temp.sort)
  # plot
plot(x=AQ$Temp, y=AQ$Ozone, col="gray", 
     main="UP TO YOU")
legend(x=55, y=165, # where you want to put the legend
       legend=c("cubic","ns 5 DF", "ns 7 DF"), lty="solid",
       col=colors()[c(24,121,145)], lwd=1) # change the color as you wish
lines(temp.sort$Temp, pred.cubic.poly,col=colors()[24], lwd = 2)
lines(temp.sort$Temp, pred.ns.5, col=colors()[121], lwd=2)
lines(temp.sort$Temp, pred.ns.7, col=colors()[145], lwd=2)
# SMOOTHING SPLINES 
sm.spl.5 <- smooth.spline(x=AQ$Temp, y=AQ$Ozone, df=5)
sm.spl.7 <- smooth.spline(x=AQ$Temp, y=AQ$Ozone, df=7)
  # plot 
plot(x=AQ$Temp, y=AQ$Ozone, col="gray", 
     main="...")
legend(x=57, y=160, # where you want to put the legend
       legend=c("Smoothing Spline 5 df", "Smoothing Spline 7 df"), 
       lty="solid", col=colors()[c(24,121)], lwd=2) 
lines(sm.spl.5, col=colors()[24], lwd=2)
lines(sm.spl.7, col=colors()[121], lwd=2)

# use CV and GCV to choose the optimal smoothing amount 
# Optimal Spline.  
#   "CV=TRUE" uses N-fold CV.  NOT RECOMMENDED IF DUPLICATE VALUES OF X EXIST
#   "CV=FALSE" uses generalized CV (GCV)
# remove duplicate values of x
AQ.rd = AQ %>%
  distinct(Temp, .keep_all = T) 

sm.spl.opt.rd <- smooth.spline(x=AQ.rd$Temp, y=AQ.rd$Ozone, cv=TRUE) # remove duplicate values
sm.spl.opt.rd
sm.spl.opt2 <- smooth.spline(x=AQ$Temp, y=AQ$Ozone, cv=FALSE) 
sm.spl.opt2

plot(x=AQ.rd$Temp, y=AQ.rd$Ozone, col="gray", 
     main="Comparison of 'Optimum' Smoothing splines")
legend(x=57, y=120, legend=c("N-Fold CV_rd", "Generalized CV"), 
       lty="solid", col=colors()[c(79,28)], lwd=2)

lines(sm.spl.opt.rd, col=colors()[79], lwd=2)
lines(sm.spl.opt2, col=colors()[28], lwd=2)

# LOESS 
fit.loess.5 = loess(Ozone ~ Temp, data = AQ, enp.target = 5)
fit.loess.7 = loess(Ozone ~ Temp, data = AQ, enp.target = 7)
min.Temp = min(AQ$Temp)
max.Temp = max(AQ$Temp)
vals.Temp.raw = seq(from = min.Temp, to = max.Temp, length.out = 100) # TA uses 100
vals.Temp = data.frame(Temp = vals.Temp.raw)
  # prediction 
pred.loess.5 = predict(fit.loess.5, vals.Temp)
pred.loess.7 = predict(fit.loess.7, vals.Temp)
  # plot
plot(x=AQ$Temp, y=AQ$Ozone, col="gray", 
     main="...")
legend(x=57, y=160, # where you want to put the legend
       legend=c("LOESS 5 df", "LOESS 7 df"), 
       lty="solid", col=colors()[c(24,121)], lwd=2) # change the color as you wish

lines(x = vals.Temp$Temp,y = pred.loess.5, col=colors()[24], lwd=2)
lines(x = vals.Temp$Temp,y = pred.loess.7, col=colors()[121], lwd=2)

################################################################
# 19. scale
set.seed(67982193)
perm <- sample(x=nrow(data))
set1 <- data[perm <= 200,] # may change 200
set2 <- data[perm>200,]

## Need to scale all variables to have same SD
scale.1 <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- mean(x2[,col])
    b <- sd(x2[,col])
    x1[,col] <- (x1[,col]-a)/b
  }
  x1
}
# Creating training and test X matrices, then scaling them.
x.1.unscaled <- as.matrix(set1[,-1])
x.1 <- scale.1(x.1.unscaled,x.1.unscaled)
x.2.unscaled <- as.matrix(set2[,-1])
x.2 <- scale.1(x.2.unscaled,x.1.unscaled)
#####################################################################
# 20. KNN
library(FNN)

# TUNE using cv.knn(): leave-one-out (n-fold) CV
kmax <- 40 # choose your own
k <- matrix(c(1:kmax), nrow=kmax)
runknn <- function(x){
  knncv.fit <- knn.cv(train=x.1, cl=set1[,1], k=x)
  # Fitted values are for deleted data from CV
  mean(ifelse(knncv.fit == set1[,1], yes=0, no=1))
}

mis <- apply(X=k, MARGIN=1, FUN=runknn)
mis.se <- sqrt(mis*(1-mis)/nrow(set2)) #SE of misclass rates

#Now plot results
# Plot like the CV plots, with 1SE bars and a horizontal line 
#   at 1SE above minimum.
plot(x=k, y=mis, type="b", ylim=c(.25,.50)) #value of ylim may be different

for(ii in c(1:kmax)){
  lines(x=c(k[ii],k[ii]), y=c(mis[ii]-mis.se[ii], mis[ii]+mis.se[ii]), col=colors()[220])
}
abline(h=min(mis + mis.se), lty="dotted")

# k for Minimum CV error
mink = which.min(mis)
#Trying the value of k with the lowest validation error on test data set.
knnfitmin.2 <- knn(train=x.1, test=x.2, cl=set1[,1], k=mink)

table(knnfitmin.2, set2[,1],  dnn=c("Predicted","Observed"))
(misclass.2.knnmin <- mean(ifelse(knnfitmin.2 == set2[,1], yes=0, no=1)))

# Less variable models have larger k, so find largest k within 
#   1 SE of minimum validation error 
serule = max(which(mis<mis[mink]+mis.se[mink]))
knnfitse.2 <- knn(train=x.1, test=x.2, cl=set1[,1], k=serule)

table(knnfitse.2, set2[,1],  dnn=c("Predicted","Observed"))
(misclass.2.knnse <- mean(ifelse(knnfitse.2 == set2[,1], yes=0, no=1)))
###########################################################################
# 21. Logistic regression
# multinom uses a formula, so need to keep data in data.frame
# Creating training and test X matrices, then scaling them.
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

set1.rescale <- data.frame(cbind(rescale(set1[,-1], set1[,-1]), Y=set1$Y))
set2.rescale <- data.frame(cbind(rescale(set2[,-1], set1[,-1]), Y=set2$Y))
summary(set1.rescale) # now Y is the 17th variable

library(nnet)

mod.fit <- multinom(data=set1.rescale, formula=Y ~ ., 
                    trace=TRUE)
summary(mod.fit)

library(car)
Anova(mod.fit) 

# Misclassification Errors
pred.class.1 <- predict(mod.fit, newdata=set1.rescale, 
                        type="class")
pred.class.2 <- predict(mod.fit, newdata=set2.rescale, 
                        type="class")
(mul.misclass.train <- mean(ifelse(pred.class.1 == set1$Y, 
                                   yes=0, no=1)))
(mul.misclass.test <- mean(ifelse(pred.class.2 == set2$Y, 
                                  yes=0, no=1)))

# Estimated probabilities for test set
pred.probs.2 <- predict(mod.fit, newdata=set2.rescale, 
                        type="probs")
round(head(pred.probs.2),3)

# Test set confusion matrix
table(set2$type, pred.class.2, dnn=c("Obs","Pred"))
#####################################################################
# 22. LASSO logistic regression model
library(glmnet)

logit.fit <- glmnet(x=as.matrix(set1.rescale[,-17]), # 17 = Y
                    y=set1.rescale[,17], family="multinomial")
# Note that parameters are not the same as in multinom()
coef(logit.fit, s=0)

# Predicted probabilities
logit.prob.2 <- predict(logit.fit, s=0, type="response",
                        newx=as.matrix(set2.rescale[,1:16])) # X1 - X16
round(head(logit.prob.2[,,1]), 3)

# Calculate in-sample and out-of-sample misclassification error
las0.pred.train <- predict(object=logit.fit, s=0, type="class",
                           newx=as.matrix(set1.rescale[,1:16]))
las0.pred.test <- predict(logit.fit, s=0, type="class",
                          newx=as.matrix(set2.rescale[,1:16]))
(las0misclass.train <- 
    mean(ifelse(las0.pred.train == set1.rescale$Y, 
                yes=0, no=1)))
(las0misclass.test <- 
    mean(ifelse(las0.pred.test == set2.rescale$Y,
                yes=0, no=1)))


# "Optimal" LASSO Fit
logit.cv <- cv.glmnet(x=as.matrix(set1.rescale[,1:16]), 
                      y=set1.rescale[,17], family="multinomial")
logit.cv
plot(logit.cv)

## Find nonzero lasso coefficients
c <- coef(logit.fit,s=logit.cv$lambda.min) 
cmat <- cbind(as.matrix(c[[1]]), as.matrix(c[[2]]), 
              as.matrix(c[[3]]))
round(cmat,2)
cmat!=0

lascv.pred.train <- predict(object=logit.cv, type="class", 
                            s=logit.cv$lambda.min, 
                            newx=as.matrix(set1.rescale[,1:16]))
lascv.pred.test <- predict(logit.cv, type="class", 
                           s=logit.cv$lambda.min, 
                           newx=as.matrix(set2.rescale[,1:16]))
(lascvmisclass.train <- 
    mean(ifelse(lascv.pred.train == set1$Y, yes=0, no=1)))
(lascvmisclass.test <- 
    mean(ifelse(lascv.pred.test == set2$Y, yes=0, no=1)))
#####################################################################
# 23. Discriminant Analysis - LDA
library(MASS)

set1s <- apply(set1[,-1], 2, scale) # 1 = Y
set1s <- data.frame(set1s,Y=set1$Y)
lda.fit.s <- lda(data=set1s, Y~.)
lda.fit.s

# Fit gives identical results as without scaling, but 
#  can't interpret means
lda.fit <- lda(x=set1[,-1], grouping=set1$Y)
lda.fit

# Plot results.  Create standard colours for classes. 
class.col <-  ifelse(set1$Y=="A",y=53,n=
                 ifelse(set1$Y=="B",y=68,n=
                          ifelse(set1$Y=="C", y=203, n= # change number of color
                                   ifelse(set1$Y=="D", y=298,n=175)))) # Y = A,B,C,D,E

plot(lda.fit, col=colors()[class.col])

# Calculate in-sample and out-of-sample misclassification error
lda.pred.train <- predict(lda.fit, newdata=set1[,-1])$class
lda.pred.test <- predict(lda.fit, newdata=set2[,-1])$class
(lmisclass.train <- mean(ifelse(lda.pred.train == set1$Y, yes=0, no=1)))
(lmisclass.test <- mean(ifelse(lda.pred.test == set2$Y, yes=0, no=1)))

# Test set confusion matrix
table(set2$Y, lda.pred.test, dnn=c("Obs","Pred"))
#####################################################################################
# 24. Discriminant Analysis - QDA
qda.fit <- qda(data=set1, Y~.)
qda.fit

qda.pred.train <- predict(qda.fit, newdata=set1)$class
qda.pred.test <- predict(qda.fit, newdata=set2)$class
(qmisclass.train <- mean(ifelse(qda.pred.train == set1$Y, yes=0, no=1)))
(qmisclass.test <- mean(ifelse(qda.pred.test == set2$Y, yes=0, no=1)))

# Test set confusion matrix
table(set2$Y, qda.pred.test, dnn=c("Obs","Pred"))

qda.corr = ifelse(qda.pred.test == set2$Y, yes="Y", no="N")
lda.corr = ifelse(lda.pred.test == set2$Y, yes="Y", no="N")

mcnemar.test(lda.corr, qda.corr)

#################################################################################
# 25. GAM
library(mgcv)

levels(set1$Y)
set1$Y0 <- as.numeric(set1$Y) - 1
# A will be our baseline class

# Fit full model, all variables in each logit
## Add s() to nonlinear X
gam.m <- gam(data=set10, list(Y0
  # B
  ~ s(X1) + s(X2) + X3 + s(X4) + X5 
  + s(X6) + s(X7) + s(X8) + s(X9) + X10
  + X11 + X12 + s(X13) + s(X14) + s(X15) + X16, 
  # C
  ~ s(X1) + s(X2) + X3 + s(X4) + X5 
  + s(X6) + s(X7) + s(X8) + s(X9) + X10
  + X11 + X12 + s(X13) + s(X14) + s(X15) + X16,
  # D
  ~ s(X1) + s(X2) + X3 + s(X4) + X5 
  + s(X6) + s(X7) + s(X8) + s(X9) + X10
  + X11 + X12 + s(X13) + s(X14) + s(X15) + X16,
  # E
  ~ s(X1) + s(X2) + X3 + s(X4) + X5 
  + s(X6) + s(X7) + s(X8) + s(X9) + X10
  + X11 + X12 + s(X13) + s(X14) + s(X15) + X16),
            family=multinom(K=4)) # K is number of logit

summary(gam.m)

pred.prob.m <- predict(gam.m, newdata=set1, type="response")
pred.class.m <- apply(pred.prob.m,1,function(x) which(max(x)==x)[1])-1

head(cbind(round(pred.prob.m, digits=3), pred.class.m))

pred.prob.2m <- predict(gam.m, newdata=set2, type="response")
pred.class.2m <- apply(pred.prob.2m,1,function(x) which(max(x)==x)[1])-1

(misclassm.train <- mean(pred.class.m != as.numeric(set1$Y)-1))
(misclassm.test <- mean(pred.class.2m != as.numeric(set2$Y)-1))

# Confusion Matrix
table(set2$Y, pred.class.2m,  dnn=c("Observed","Predicted"))
#######################################################################################
# 26. Naive Bayes
library(klaR)

# L18-GAM creats a Y0 in set1 so we re-create set1 
set.seed(67982193)
perm <- sample(x=nrow(data))
set1 <- data[perm <= 200,] # may change 200
set2 <- data[perm>200,]

# Run PCA before Naive Bayes to decorrelate data
pc <-  prcomp(x=set1[,-1], scale.=TRUE) # 1 = Y

# Create the same transformations in all three data sets 
#   and attach the response variable at the end
#   predict() does this 
xi.1 <- data.frame(pc$x,Y = as.factor(set1$Y))
xi.2 <- data.frame(predict(pc, newdata=set2), Y = as.factor(set2$Y))
summary(xi.1)

#  First with normal distributions
NBn.pc <- NaiveBayes(x=xi.1[,-17], grouping=xi.1[,17], usekernel=FALSE) # Now 17 = Y

#  Comment this plot out if you don't want to see the 
#    estimated distributions of Xj within classes
#par(mfrow=c(2,3))
plot(NBn.pc, lwd=2, main="NB Normal with PC")

NBnpc.pred.train <- predict(NBn.pc, newdata=xi.1[,-17], type="class") 
table(NBnpc.pred.train$class, xi.1[,17], dnn=c("Predicted","Observed"))

NBnpc.pred.test <- predict(NBn.pc, newdata=xi.2[,-17], type="class")
table(NBnpc.pred.test$class, xi.2[,17], dnn=c("Predicted","Observed"))


# Error rates
(NBnPCmisclass.train <- mean(ifelse(NBnpc.pred.train$class == xi.1$Y, yes=0, no=1)))
(NBnPCmisclass.test <- mean(ifelse(NBnpc.pred.test$class == xi.2$Y, yes=0, no=1)))

# Repeat, using kernel density estimates
NBk.pc <- NaiveBayes(x=xi.1[,-17], grouping=xi.1[,17], usekernel=TRUE)

#  Comment this plot out if you don't want to see the 
#    estimated distributions of Xj within classes
#par(mfrow=c(2,3))
plot(NBk.pc, lwd=2, main="NB Kernel with PC")

NBkpc.pred.train <- predict(NBk.pc, newdata=xi.1[,-17], type="class")
table(NBkpc.pred.train$class, xi.1[,17], dnn=c("Predicted","Observed"))

NBkpc.pred.test <- predict(NBk.pc, newdata=xi.2[,-17], type="class")
table(NBkpc.pred.test$class, xi.2[,17], dnn=c("Predicted","Observed"))

# Error rates
(NBkPCmisclass.train <- mean(ifelse(NBkpc.pred.train$class == xi.1$Y, yes=0, no=1)))
(NBkPCmisclass.test <- mean(ifelse(NBkpc.pred.test$class == xi.2$Y, yes=0, no=1)))

#####################################################################################
# 27. Classification Tree
library(rpart)
library(rpart.plot)

wh.tree <- rpart(data=set1, Y ~ ., method="class", cp=0)
printcp(wh.tree)
round(wh.tree$cptable[,c(2:5,1)],4)

# summary(wh.tree) #Lots of output

# See pdf of this---Note that it IS making splits that improve 
#   probabilities but do not change classes
prp(wh.tree, type=1, extra=1, main="Original full tree")

# Plot of the cross-validation for the complexity parameter.
##  NOTE: Can be very variable, depending on CV partitioning
plotcp(wh.tree)


# Find location of minimum error
cpt = wh.tree$cptable
minrow <- which.min(cpt[,4])
# Take geometric mean of cp values at min error and one step up 
cplow.min <- cpt[minrow,1]
cpup.min <- ifelse(minrow==1, yes=1, no=cpt[minrow-1,1])
cp.min <- sqrt(cplow.min*cpup.min)

# Find smallest row where error is below +1SE
se.row <- min(which(cpt[,4] < cpt[minrow,4]+cpt[minrow,5]))
# Take geometric mean of cp values at min error and one step up 
cplow.1se <- cpt[se.row,1]
cpup.1se <- ifelse(se.row==1, yes=1, no=cpt[se.row-1,1])
cp.1se <- sqrt(cplow.1se*cpup.1se)

# Creating a pruned tree using a selected value of the CP by CV.
wh.prune.cv.1se <- prune(wh.tree, cp=cp.1se)
# Creating a pruned tree using a selected value of the CP by CV.
wh.prune.cv.min <- prune(wh.tree, cp=cp.min)

# Plot the pruned trees
par(mfrow=c(1,2))
prp(wh.prune.cv.1se, type=1, extra=1, main="Pruned CV-1SE tree")
prp(wh.prune.cv.min, type=1, extra=1, main="Pruned CV-min tree")


# Predict results of classification. "Vector" means store class as a number
pred.test.cv.1se <- predict(wh.prune.cv.1se, newdata=set2, type="class")
pred.test.cv.min <- predict(wh.prune.cv.min, newdata=set2, type="class")
pred.test.full <- predict(wh.tree, newdata=set2, type="class")


(misclass.test.cv.1se <- mean(ifelse(pred.test.cv.1se == set2$Y, yes=0, no=1)))
(misclass.test.cv.min <- mean(ifelse(pred.test.cv.min == set2$Y, yes=0, no=1)))
(misclass.test.full <- mean(ifelse(pred.test.full == set2$Y, yes=0, no=1)))

# Confusion Matrices
table(set2$type, pred.test.full,  dnn=c("Observed","Predicted"))

###################################################################################
# 28. Random Forest
library(randomForest)

# Tuning variables and node sizes

set.seed(879417)
reps=5 # may change
varz = 1:16 # number of your variable (# of selected vars)
nodez = c(1,3,5,7,10) # may change

NS = length(nodez)
M = length(varz)
rf.oob = matrix(NA, nrow=M*NS, ncol=reps)

for(r in 1:reps){
  counter=1
  for(m in varz){
    for(ns in nodez){
      wh.rfm <- randomForest(data=set1, Y~., 
                              mtry=m, nodesize=ns)
      rf.oob[counter,r] = mean(predict(wh.rfm, type="response") != set1$Y)
      counter=counter+1
    }
  }
}

parms = expand.grid(nodez,varz)
row.names(rf.oob) = paste(parms[,2], parms[,1], sep="|")

mean.oob = apply(rf.oob, 1, mean)
mean.oob[order(mean.oob)]

min.oob = apply(rf.oob, 2, min)

boxplot(rf.oob, use.cols=FALSE, las=2)

boxplot(t(rf.oob)/min.oob, use.cols=TRUE, las=2, 
        main="RF Tuning Variables and Node Sizes")

# Suggested parameters are mtry=?, nodesize=?
wh.rf.tun <- randomForest(data=set1, Y~., mtry= ?, nodesize=?,
                      importance=TRUE, keep.forest=TRUE)

# Predict results of classification. 
pred.rf.train.tun <- predict(wh.rf.tun, newdata=set1, type="response")
pred.rf.test.tun <- predict(wh.rf.tun, newdata=set2, type="response")
#"vote" gives proportions of trees voting for each class
pred.rf.vtrain.tun <- predict(wh.rf.tun, newdata=set1, type="vote")
pred.rf.vtest.tun <- predict(wh.rf.tun, newdata=set2, type="vote")
head(cbind(pred.rf.test.tun,pred.rf.vtest.tun))

(misclass.train.rf.tun <- mean(ifelse(pred.rf.train.tun == set1$Y, yes=0, no=1)))
(misclass.test.rf.tun <- mean(ifelse(pred.rf.test.tun == set2$Y, yes=0, no=1)))
#########################################################################################
# 29. NN
library(nnet)

# tuning with caret::train
library(caret)

#Using 10-fold CV so that training sets are not too small
#  ( Starting with 200 in training set)
trcon = trainControl(method="repeatedcv", number=10, repeats=2,
                     returnResamp="all")
parmgrid = expand.grid(size=c(1,3,6,10),decay= c(0,0.001,0.01,0.1)) # may change

tuned.nnet <- train(x=x.1, y=set1$Y, method="nnet", preProcess="range", trace=FALSE, 
                    tuneGrid=parmgrid, trControl = trcon)

tuned.nnet$results[order(-tuned.nnet$results[,3]),]
tuned.nnet$bestTune

# Let's rearrange the data so that we can plot the bootstrap resamples in 
#   our usual way, including relative to best
resamples = reshape(data=tuned.nnet$resample[,-2], idvar=c("size", "decay"), 
                    timevar="Resample", direction="wide")
(best = apply(X=resamples[,-c(1,2)], MARGIN=2, FUN=max))
siz.dec <- paste(resamples[,1],"-",resamples[,2])

boxplot.matrix(x=t(t(1-resamples[,-c(1:2)])), use.cols=FALSE, names=siz.dec,
               main="Misclassification rates for different Size-Decay", las=2)

boxplot.matrix(x=t(t(1-resamples[,-c(1:2)])/(1-best)), use.cols=FALSE, names=siz.dec,
               main="Relative Misclass rates for different Size-Decay", las=2)

par(mfrow=c(1,2))
boxplot(t(t(1-resamples[,-c(1:2)])/(1-best)) ~ resamples[,1], xlab="Size", ylab="Relative Error")
boxplot(t(t(1-resamples[,-c(1:2)])/(1-best)) ~ resamples[,2], xlab="Decay", ylab="Relative Error")
#################################################################
# Write a CV tuner that uses multiple restarts
##################################################################
# WARNING: Can run LONG!  PARALLELIZE!

#For simplicity, rename data as "train.x" and "train.y"
train.x = set1[,-1]
train.y.class = set1[,1] 
train.y = class.ind(train.y.class)

#  Let's do R=2 reps of V=10-fold CV.
set.seed(74375641)
V=10
R=2 
n2 = nrow(train.x)
# Create the folds and save in a matrix
folds = matrix(NA, nrow=n2, ncol=R)
for(r in 1:R){
  folds[,r]=floor((sample.int(n2)-1)*V/n2) + 1
}

# Grid for tuning parameters and number of restarts of nnet
siz <- c(1,3,6,10)
dec <- c(0.0001,0.001,0.01,0.1)
nrounds=10

# Prepare matrix for storing results: 
#   row = 1 combination of tuning parameters
#   column = 1 split
#   Add grid values to first two columns

Mis.cv = matrix(NA, nrow=length(siz)*length(dec), ncol=V*R+2)
Mis.cv[,1:2] = as.matrix(expand.grid(siz,dec))

# Start loop over all reps and folds.  
for (r in 1:R){ 
  for(v in 1:V){
    
    y.1 <- as.matrix(train.y[folds[,r]!=v,])
    x.1.unscaled <- as.matrix(train.x[folds[,r]!=v,]) 
    x.1 <- rescale(x.1.unscaled, x.1.unscaled) 
    
    #Test
    y.2 <- as.matrix(train.y[folds[,r]==v],)
    x.2.unscaled <- as.matrix(train.x[folds[,r]==v,]) # Original data set 2
    x.2 = rescale(x.2.unscaled, x.1.unscaled)
    
    # Start counter to add each model's misclassification to row of matrix
    qq=1
    # Start Analysis Loop for all combos of size and decay on chosen data set
    for(d in dec){
      for(s in siz){
        
        ## Restart nnet nrounds times to get best fit for each set of parameters 
        Mi.final <- 1
        #  check <- MSE.final
        for(i in 1:nrounds){
          nn <- nnet(y=y.1, x=x.1, size=s, decay=d, maxit=2000, softmax=TRUE, trace=FALSE)
          Pi <- predict(nn, newdata=x.1, type="class")
          Mi <- mean(Pi != as.factor(set1[folds[,r]!=v,1]))
          
          if(Mi < Mi.final){ 
            Mi.final <- Mi
            nn.final <- nn
          }
        }
        pred.nn = predict(nn.final, newdata=x.2, type="class")
        Mis.cv[qq,(r-1)*V+v+2] = mean(pred.nn != as.factor(train.y.class[folds[,r]==v]))
        qq = qq+1
      }
    }
  }
}
Mis.cv

(Micv = apply(X=Mis.cv[,-c(1,2)], MARGIN=1, FUN=mean))
(Micv.sd = apply(X=Mis.cv[,-c(1,2)], MARGIN=1, FUN=sd))
Micv.CIl = Micv - qt(p=.975, df=R*V-1)*Micv.sd/sqrt(R*V)
Micv.CIu = Micv + qt(p=.975, df=R*V-1)*Micv.sd/sqrt(R*V)
(all.cv = cbind(Mis.cv[,1:2],round(cbind(Micv,Micv.CIl, Micv.CIu),2)))
all.cv[order(Micv),]


# Plot results. 
siz.dec <- paste("NN",Mis.cv[,1],"-",Mis.cv[,2])
boxplot(x=Mis.cv[,-c(1,2)], use.cols=FALSE, names=siz.dec,
        las=2, main="MisC Rate boxplot for various NNs")

# Plot RELATIVE results. 
lowt = apply(Mis.cv[,-c(1,2)], 2, min)

# margin defaults are 5,4,4,2, bottom, left, top right
#  Need more space on bottom, so increase to 7.
par(mar=c(7,4,4,2))
boxplot(x=t(Mis.cv[,-c(1,2)])/lowt, las=2 ,names=siz.dec,
        main="Relative MisC Rate boxplot for various NNs")

relMi = t(Mis.cv[,-c(1,2)])/lowt
(RRMi = apply(X=relMi, MARGIN=2, FUN=mean))
(RRMi.sd = apply(X=relMi, MARGIN=2, FUN=sd))
RRMi.CIl = RRMi - qt(p=.975, df=R*V-1)*RRMi.sd/sqrt(R*V)
RRMi.CIu = RRMi + qt(p=.975, df=R*V-1)*RRMi.sd/sqrt(R*V)
(all.rrcv = cbind(Mis.cv[,1:2],round(cbind(RRMi,RRMi.CIl, RRMi.CIu),2)))
all.rrcv[order(RRMi),]

###########################################################
#  Fit the suggested model to full training data, 
#    with ? nodes and ? shrinkage
###########################################################

x.1.unscaled <- as.matrix(set1[,-1])
x.2.unscaled <- as.matrix(set2[,-1])
x.1 <- rescale(x.1.unscaled, x.1.unscaled)
x.2 <- rescale(x.2.unscaled, x.1.unscaled)

y.1 <- class.ind(set1[,1])
y.2 <- class.ind(set2[,1])

Mi.final = 1
for(i in 1:10){
  nn <- nnet(y=y.1, x=x.1, size=?, decay=?, maxit=2000, softmax=TRUE, trace=FALSE)
  Pi <- predict(nn, newdata=x.1, type="class")
  Mi <- mean(Pi != as.factor(set1[,1]))
  
  if(Mi < Mi.final){ 
    Mi.final <- Mi
    nn.final <- nn
  }
}


# Test set error
p2.nn.3.01 <-predict(nn.final, newdata=x.2, type="class")
(misclass2.3.01 <- mean(ifelse(p2.nn.3.01 == set2$Y, yes=0, no=1)))
table(p2.nn.3.01, as.factor(set2$Y),  dnn=c("Predicted","Observed"))

######################################################################################
# 30. SVM
# Try tuning with caret::train
# Note that caret uses a different implementation of SVM than e1071
#  Different tuning parameters!
#
# method="svmPoly" has tuninh parameters:
#     degree (Polynomial Degree, d)
#     scale (Scale, s)
#     C (Cost, C)
#
# method="svmRadial"
#     sigma (gamma)
#     C (Cost, C)
#############################################

library(caret)

#Using 10-fold CV so that training sets are not too small
#  ( Starting with 200 in training set)
trcon = trainControl(method="repeatedcv", number=10, repeats=2,
                     returnResamp="all")
parmgrid = expand.grid(C=10^c(0:5), sigma=10^(-c(5:0)))

tuned.nnet <- train(x=set1[,-1], y=set1$Y, method="svmRadial", 
                    preProcess=c("center","scale"), trace=FALSE, 
                    tuneGrid=parmgrid, trControl = trcon)

tuned.nnet$results[order(-tuned.nnet$results[,3]),]
tuned.nnet$bestTune

# Let's rearrange the data so that we can plot the bootstrap resamples in 
#   our usual way, including relative to best
resamples = reshape(data=tuned.nnet$resample[,-2], idvar=c("C", "sigma"), 
                    timevar="Resample", direction="wide")
head(resamples)
(best = apply(X=resamples[,-c(1,2)], MARGIN=2, FUN=max))
C.sigma <- paste(log10(resamples[,1]),"-",log10(resamples[,2]))

boxplot.matrix(x=t(t(1-resamples[,-c(1:2)])), use.cols=FALSE, names=C.sigma,
               main="Misclassification rates for different Cost-Gamma", las=2)

boxplot.matrix(x=t(t(1-resamples[,-c(1:2)])/(1-best)), use.cols=FALSE, names=C.sigma,
               main="Relative Misclass rates for different Cost-Gamma", las=2)

par(mfrow=c(1,2))
boxplot(t(t(1-resamples[,-c(1:2)])/(1-best)) ~ resamples[,1], xlab="C", ylab="Relative Error")
boxplot(t(t(1-resamples[,-c(1:2)])/(1-best)) ~ resamples[,2], xlab="Sigma", ylab="Relative Error")

# Refit the best tuned model

svm.wh.tun <- svm(data=set1, Y ~ ., kernel="radial", 
               gamma=10^(-3), cost=10^4)
summary(svm.wh.tun)
head(svm.wh.tun$decision.values)
head(svm.wh.tun$fitted)

pred1.wh.tun <- predict(svm.wh.tun, newdata=set1)
table(pred1.wh.tun, set1$Y,  dnn=c("Predicted","Observed"))
(misclass1.wh.tun <- mean(ifelse(pred1.wh.tun == set1$Y, yes=0, no=1)))

pred2.wh.tun <- predict(svm.wh.tun, newdata=set2)
table(pred2.wh.tun, set2$Y,  dnn=c("Predicted","Observed"))
(misclass2.wh.tun <- mean(ifelse(pred2.wh.tun == set2$Y, yes=0, no=1)))