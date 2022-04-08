# -------------------------------------------------------------------------
#  Hierarchical Hidden Markov Process with State Prediction: Simulated data analysis
#  Author: Kevin L. McKee (klmckee@ucdavis.edu)
#  Description: This script uses the Stan model to analyze the simulated data as though it were real experimental data.
#  The steps are to (1) load the data, (2) fit each model to the data, (3) run model comparisons using the test data.
#  Data were generated with data_simulation_model3.R.
# -------------------------------------------------------------------------

#Load packages, functions, and the model
require(loo)
require(rstan)
source("functions.R")

#May take a minute to compile. Note: Stan likely to crash after a model has been compiled to the same object ('model3' here) multiple times.
model3<-stan_model("models/model3_test.stan")


# Read data ---------------------------------------------------------------
# Data used for parameter estimation:
fileDir<-"sim data/"
filenames<-list.files(fileDir, full.names = T)
testfile<-read.csv(filenames[1])
dataDims<-c(length(filenames), dim(testfile))
dat<-array(NA, dim=dataDims)
for(i in 1:dataDims[1]){
  dat[i,,]<-as.matrix(read.csv(filenames[i]))
}

# Test data
fileDir<-"test data/"
filenames<-list.files(fileDir, full.names = T)
testfile<-read.csv(filenames[1])
dataDims<-c(length(filenames), dim(testfile))
dat.test<-array(NA, dim=dataDims)
for(i in 1:dataDims[1]){
  dat.test[i,,]<-as.matrix(read.csv(filenames[i]))
}


#Set up the data object for the Stan model. The names set in quotes are defined in the model's code.
stan_dat<-list(
  "N"=dataDims[1],            #Number of people
  "K"=2,                      #Number of hidden states / sub models
  "J"=dataDims[3]-1,          #Number of covariates
  "T"=dataDims[2],            #Number of observations per person
  "X"=dat[,,2:3],             #Covariate data (estimation)
  "X_test"=dat.test[,,2:3],   #Covariate data (testing)
  "Y"=dat[,,1],               #Response data (estimation) 
  "Y_test"=dat.test[,,1],     #Response data (testing)
  "indMat"=genIndMat(2),       #Used internally to organize transition parameters
  "ratePriorMu"=c(-2, 2),
  "ratePriorSD"=c( 2, 2)
)


# -------------------------------------------------------------------------
# Model fitting -----------------------------------------------------------
# -------------------------------------------------------------------------

# 2 hidden states ---------------------------------------------------------
nState<-2

stan_dat$K<-nState
stan_dat$indMat<-genIndMat(nState) #Need to re-generate this for each model
stan_dat$ratePriorMu<-c(-2, 2)
stan_dat$ratePriorSD<-c(2,  2)

# The model can be fit with sampling, optimization, or variational inference. Sampling is best for hierarchical and multi-modal models.
# This kind of model is both.

fit1.samp<-sampling(object=model3, data=stan_dat, 
                   chains=5, cores=5, warmup=200, iter=400, seed=123,
                   control=list(max_treedepth=4, adapt_delta=.7),
                   include=F, pars=c("acc","gamma", "g1", "A", "g1m","A", "A0"))
# saveRDS(fit1.samp, "fit1.RDS")

# fit1.samp<-readRDS("fit1.RDS")
draws1<-extract(fit1.samp) #Extract list containing samples

# For mixture or hidden markov models, there is always a risk of hidden states 'collapsing' together to only get part of the solution.
# It is difficult in general to solve this in an automated way.
# Instead, examine the results for a particular parameter that you expect to be different between states, such as the 
# base response rate:
plot(draws1$b0_mu[,1], col="blue", ylim=c(-4,4))
points(draws1$b0_mu[,2], col="red")

#Note that most chains separate into two modes, but a couple chains collapse into one.
#Use subscripting to pick only the separated modes:
selSub1<-c(1:600)
plot(draws1$b0_mu[selSub1,1], col="blue", ylim=c(-4,4))
points(draws1$b0_mu[selSub1,2], col="red")

#Examine parameter estimates
colMeans(draws1$b0_mu[selSub1,])
colMeans(draws1$b1_mu[selSub1,,])
cbind(0,colMeans(draws1$g1_mu[selSub1,,]))

#Extract the hidden state probabilities and examine them for each person:
z<-colMeans(draws1$alpha_test[selSub1,,,])
par(mfrow=c(6,2), mai=c(.5,.6,.1,.1), mgp=c(2,1,0))
for(i in 1:stan_dat$N){
  plot(z [i,,1], type="l", col="red", lwd=2, ylim=c(0,1), ylab="Probability", xlab="Time")
  lines(z[i,,2], col="blue", lwd=2)
  abline(v=which(stan_dat$X_test[i,,2]==1), lty=2, lwd=2)
} 

#Collect log likelihoods for later model comparison
LL1<-draws1$log_lik[selSub1,]
LL1_test<-draws1$log_lik_test[selSub1,]





# 3 hidden states ---------------------------------------------------------
#All previous steps repeated for a new model with 3 states instead of 2.
nState<-3

stan_dat$K<-nState
stan_dat$indMat<-genIndMat(nState)
stan_dat$ratePriorMu<-c(-1, 0, 1)
stan_dat$ratePriorSD<-c( 2, 2, 2)


fit2.samp<-sampling(object=model3, data=stan_dat,
                    chains=5, cores=5, warmup=200, iter=400,  seed=123,
                    control=list(max_treedepth=4, adapt_delta=.7),
                    include=F, pars=c("acc","gamma", "g1", "A", "g1m","A", "A0"))
# saveRDS(fit2.samp, "fit2.RDS")

# fit2.samp<-readRDS("fit2.RDS") 
draws2<-extract(fit2.samp)

#Examine samples to pick out coherent chains:
plot(draws2$b0_mu[,1], col="blue", ylim=c(-5,5))
points(draws2$b0_mu[,2], col="red")
points(draws2$b0_mu[,3], col="purple")
selSub2<-c(801:1000)
plot(draws2$b0_mu[selSub2,1], col="blue", ylim=c(-4,4))
points(draws2$b0_mu[selSub2,2], col="red")
points(draws2$b0_mu[selSub2,3], col="purple")

colMeans(draws2$b0_mu[selSub2,])
colMeans(draws2$b1_mu[selSub2,,])
colMeans(draws2$g1_mu[selSub2,,])

#Plotting
z<-colMeans(draws2$alpha_test[selSub2,,,])
par(mfrow=c(6,2), mai=c(.4,.5,.1,.1), mgp=c(2,1,0))
for(i in 1:stan_dat$N){
  plot(z [i,,1], type="l", col="red", lwd=1, ylim=c(0,1), ylab="Probability", xlab="Time")
  lines(z[i,,2], col="blue", lwd=1)
  lines(z[i,,3], col=rgb(0,.6,0), lwd=1)
  abline(v=which(stan_dat$X_test[i,,2]==1), lty=2, lwd=2)
} 

LL2<-draws2$log_lik[selSub2,]
LL2_test<-draws2$log_lik_test[selSub2,]





# 4 hidden states ---------------------------------------------------------
nState<-4

stan_dat$K<-nState
stan_dat$indMat<-genIndMat(nState)
stan_dat$indMat<-genIndMat(nState)
stan_dat$ratePriorMu<-c(-3, -1, 1, 3)
stan_dat$ratePriorSD<-c( 2,  2, 2, 2)


fit3.samp<-sampling(object=model3, data=stan_dat,  seed=123,
                    chains=5, cores=5, warmup=200, iter=400,
                    control=list(max_treedepth=4, adapt_delta=.7),
                    include=F, pars=c("acc","gamma", "g1", "A", "g1m","A", "A0"))
# saveRDS(fit3.samp, "fit3.RDS")

# fit3.samp<-readRDS("fit3.RDS")
draws3<-extract(fit3.samp)

plot(draws3$b0_mu[,1], col="blue", ylim=c(-5,5))
points(draws3$b0_mu[,2], col="red")
points(draws3$b0_mu[,3], col="purple")
points(draws3$b0_mu[,4], col="darkgreen")

selSub3<- c(801:1000)
plot(draws3$b0_mu[selSub3,1], col="blue", ylim=c(-5,5))
points(draws3$b0_mu[selSub3,2], col="red")
points(draws3$b0_mu[selSub3,3], col="purple")
points(draws3$b0_mu[selSub3,4], col="darkgreen")

colMeans(draws3$b0_mu[selSub3,])
colMeans(draws3$b1_mu[selSub3,,])
colMeans(draws3$g1_mu[selSub3,,])

#Plotting
z<-colMeans(draws3$alpha_test[selSub3,,,])
par(mfrow=c(6,2), mai=c(.1,.1,.1,.1))
for(i in 1:stan_dat$N){
  plot(z [i,,1], type="l", col="blue", lwd=1, ylim=c(0,1), ylab="Probability", xlab="Time")
  lines(z[i,,2], col="red", lwd=1)
  lines(z[i,,3], col="purple", lwd=1)
  lines(z[i,,4], col=rgb(0,.6,0), lwd=1)
  abline(v=which(stan_dat$X_test[i,,2]==1), lty=2, lwd=2)
} 

LL3<-draws3$log_lik[selSub3,]
LL3_test<-draws3$log_lik_test[selSub3,]




# Compare -----------------------------------------------------------------
#Model comparisons will use the log-likelihood.
#The likelihood of the training data alone is not sufficient, as the training data will predictably better fit by models with more parameters.
#Some approaches exist to estimate out-of-sample (i.e., test data) performance from information derived from the training data.
 
#Leave-one-out cross validation... not always effective
fit1.loo<-loo(LL1)
fit2.loo<-loo(LL2)
fit3.loo<-loo(LL3)
print(loo_compare(list("2 States"=fit1.loo,
                       "3 States"=fit2.loo, 
                       "4 States"=fit3.loo )), simplify=F)
#4-state model is neither statistically better or worse than 3-state. 2-state is definitely worse.


#AIC: Conventional to calculate per model and then choose the one with the lowest AIC value:
AIC1 <-  2*(dim(draws1$b0_mu)[-1]+sum(dim(draws1$b1_mu)[-1])+sum(dim(draws1$g0_mu)[-1])+sum(dim(draws1$g1_mu)[-1]) - mean(rowSums(LL1)))
AIC2 <-  2*(dim(draws2$b0_mu)[-1]+sum(dim(draws2$b1_mu)[-1])+sum(dim(draws2$g0_mu)[-1])+sum(dim(draws2$g1_mu)[-1]) - mean(rowSums(LL2)))
AIC3 <-  2*(dim(draws3$b0_mu)[-1]+sum(dim(draws3$b1_mu)[-1])+sum(dim(draws3$g0_mu)[-1])+sum(dim(draws3$g1_mu)[-1]) - mean(rowSums(LL3)))
AIC1;AIC2;AIC3
#AIC also prefers 3-state.


#The best method, though computationally expensive and requiring extra data, is to have a real hold-out test sample.
#Then, the model can be used to predict the test data. Test likelihoods will not improve as much in general as training likelihoods 
#as a result of model complexity alone.

#Compare test likelihoods:
par(mfrow=c(1,1),mai=c(1,1,.1,.1), mgp=c(2,1,0))
plot(density(rowSums(LL2_test)), col="blue",  xlab="Likelihood", ylab="Density", main="", xlim=c(-1150, -1050))
lines(density(rowSums(LL1_test)), col="red")
lines(density(rowSums(LL3_test)), col="purple")
legend("topleft", col=c("red","blue","purple"), lty=c(1,1,1), c("2 States", "3 States", "4 States"), lwd=c(2,2,2))
#Prefers the 3rd model as well, shows the exact overlap with other models.

#It is useful to compare test likelihoods per person:
par(mfrow=c(6,2),mai=c(.4,.5,.1,.1), mgp=c(2,1,0))
for(i in 1:stan_dat$N){
  plot(density(LL2_test[,i]), col="blue", xlab="Likelihood", ylab="Density", main="")
  lines(density(LL1_test[,i]), col="red")
  lines(density(LL3_test[,i]), col="purple")
}


#We can apply the logic of p-values and compute the probability of each mean test likelihood w.r.t each simpler model
#3-state vs 2-state
(1-pnorm(mean(rowSums(LL2_test)), mean(rowSums(LL1_test)), sd(rowSums(LL1_test))))

#4-state vs 3-state
(1-pnorm(mean(rowSums(LL3_test)), mean(rowSums(LL2_test)), sd(rowSums(LL2_test))))

#4-state vs 2-state
(1-pnorm(mean(rowSums(LL3_test)), mean(rowSums(LL1_test)), sd(rowSums(LL1_test))))




# Plot individual parameter distributions against group distribution ---------------------------------------



#Cluster 1 mean and X1 regression coef 
par(mfrow=c(6,2),mai=c(.4,.5,.1,.1), mgp=c(2,1,0))
for(i in 1:stan_dat$N){
  plot(draws2$b0_mu[selSub2,1], draws2$b1_mu[selSub2,1,1],  pch=".", cex=2, xlab="b0", ylab="b1")
  points(mean(draws2$b0_mu[selSub2,1])   + mean(draws2$b0_sd[selSub2,1])*draws2$b0_z[selSub2,i,1], 
         mean(draws2$b1_mu[selSub2,1,1]) + mean(draws2$b1_sd[selSub2,1,1])*draws2$b1_z[selSub2,i,1,1], col="red", pch=".", cex=3)
}

par(mfrow=c(1,1),mai=c(1,1,.1,.1), mgp=c(2,1,0))
plot(draws2$b0_mu[selSub2,1], draws2$b1_mu[selSub2,1,1], xlab="b0", ylab="b1", pch=".", cex=3, )
for(i in 1:stan_dat$N){
  points(mean(draws2$b0_mu[selSub2,1]   + draws2$b0_sd[selSub2,1]*draws2$b0_z[selSub2,i,1]), 
         mean(draws2$b1_mu[selSub2,1,1] + draws2$b1_sd[selSub2,1,1]*draws2$b1_z[selSub2,i,1,1]), col="red", lwd=2)
}



#Transition parameter and its regression on X2
par(mfrow=c(6,2),mai=c(.4,.5,.1,.1), mgp=c(2,1,0))
for(i in 1:stan_dat$N){
  plot(draws2$g0_mu[selSub2,1], draws2$g1_mu[selSub2,1,2], pch=".", cex=2, xlab="g0", ylab="g1")
  points(mean(draws2$g0_mu[selSub2,1])+mean(draws2$g0_sd[selSub2,1])*draws2$g0_z[selSub2,i,1], 
         mean(draws2$g1_mu[selSub2,1,2]) + mean(draws2$g1_sd[selSub2,1,2])*draws2$g1_z[selSub2,i,1,2], col="red", pch=".", cex=3,  xlab="g0", ylab="g1")
}

par(mfrow=c(1,1),mai=c(1,1,.1,.1), mgp=c(2,1,0))
plot(draws2$g0_mu[selSub2,1], draws2$g1_mu[selSub2,2,1],  pch=".", cex=2,  xlab="g0", ylab="g1")
for(i in 1:stan_dat$N){
  points(mean(draws2$g0_mu[selSub2,1]+ draws2$g0_sd[selSub2,1]*draws2$g0_z[selSub2,i,1]), 
         mean(draws2$g1_mu[selSub2,2,1] + draws2$g1_sd[selSub2,2,1]*draws2$g1_z[selSub2,i,2,1]), col="red", lwd=2)
}

#Looking at just one parameter at a time
plot(density(draws2$g0_mu[selSub2,1]))
for(i in 1:stan_dat$N) lines(density(draws2$g0_z[selSub2,i,1] + mean(draws2$g0_mu[selSub2,1])) , col="red")

plot(density(draws2$b0_mu[selSub2,1]), xlim=c(-6,1))
for(i in 1:stan_dat$N) lines(density(draws2$b0_z[selSub2,i,1] + mean(draws2$b0_mu[selSub2,1])) , col="red")

plot(density(draws2$b0_mu[selSub2,2]), xlim=c(-6,6))
for(i in 1:stan_dat$N) lines(density(draws2$b0_z[selSub2,i,2] + mean(draws2$b0_mu[selSub2,2])) , col="red")

