# -------------------------------------------------------------------------
#  Hierarchical Basic Hidden Markov Process with state prediction
#  Author: Kevin L. McKee (klmckee@ucdavis.edu)
#  Description: This script produces simulated data for a Hidden markov model with conditional state transitions, then fits the same model in Stan.
# -------------------------------------------------------------------------
source("functions.R")

# set.seed(123)
# Set up parameters -------------------------------------------------------

#Transition matrix 
pars<-list(
  "time"=300,
  "J"=2,
  "K"=3,
  "b0"=c(-2, 0, 2),
  "b1"=rbind(c(4,0,-4),0),
  "g1"=rbind(0, c(0,0,8))
)
g0<-matrix(1, pars$K, pars$K)
diag(g0)<-5
pars$g0<-g0


# Generate data -----------------------------------------------------------
softmax<-function(x) exp(x)/sum(exp(x))

genData <- function(pars) {
  with(as.list(pars),{
  z<-matrix(1, ncol=1, nrow=time)                          #Decision state
  y<-p<-matrix(0, ncol=1, nrow=time)                       #Response
  X<-cbind(
    rbinom(time, size=1, .5),
    rbinom(time, size=1, .025))
  
  for(i in 2:time){
    A0<-g0
    for(j in 1:K) A0[j,]<-A0[j,] + c(X[i,] %*% g1[,j])
    A<-apply(A0,2,softmax)
    
    z[i]<-sample(1:K, 1, p=A[,z[i-1]])
    p[i]<-plogis(b0[z[i]]+ X[i,] %*% b1[,z[i]] )
    y[i]<-rbinom(1,1,p[i])
  }
  return(data.frame("Response"=y,"Cov"=X, "z"=z, "p"=p))
  })
}

dat<-genData(pars)

plot(dat$z, type="o", col="darkgreen", ylim=c(1,pars$K))
abline(v=which(dat$Cov.2==1), col="red", lty=2)
lines(dat$Response+1, type="h", col="blue")
lines(dat$Cov.1*.5+1, type="h", col="red")


# create parameter assignment matrix for constrained transition matrix -----
stan_dat<-list(
  "K"=pars$K,
  "J"=pars$J,
  "T"=pars$time,
  "X"=dat[,2:3],
  "Y"=dat$Response,
  "indMat"=genIndMat(pars$K)
)


# Stan model ---------------------------------------------------------------
require(rstan)

model2.str<-
  "
data { 
  int K; // number of hidden states
  int J; // number of covariates
  int T; // length of series
  int Y [T]; // response
  matrix[T, J] X; //covariate
  int indMat [K,K];
}
parameters{
  ordered[K] b0;
  matrix[J,K] b1;
  vector[K*(K-1)] g0;
  vector[K-1] g1_z;
}
transformed parameters{
  real acc[K];
  real gamma[T, K];
  matrix[K, K] A0;
  matrix[K, K] A;
  vector[K] g1 = rep_vector(0.0, K);
  g1[2:K] = g1_z;

  //Set up base transition matrix
   for(j in 1:K){
     for(k in 1:K){
       if(j!=k){
          A0[j,k] = g0[indMat[j,k]];
       }else{
          A0[j,k] = 3.0;
       }
     }
   }

  //First point
  for (k in 1:K){
    gamma[1, k] = log(1.0/K) + bernoulli_logit_lpmf(Y[1] | b0[k] + X[1,]*b1[,k]);
  }

  //Series after first
  for (t in 2:T) {  
    //softmax regression: How best to constrain it?
    A = A0;
    for(k in 1:K){
      A[k,] = A[k,] + X[t,2]*g1[k];
    }
    for(k in 1:K){
      A[,k] = softmax(A[,k]);
    }
    
    //Logic: First loop through prev state probability.
    // For each previous state, add in log probs of each possible new state.
    for (k in 1:K) { 
       for (j in 1:K) {
          acc[j] = gamma[t-1, j] + log(A[k, j]) + bernoulli_logit_lpmf(Y[t] | b0[k] + X[t,]*b1[,k]);
       }
       gamma[t, k] = log_sum_exp(acc);
    }
  }
}
model {
  // priors
  b0 ~ normal(0.0, 2.0);
  to_vector(b1) ~ normal(0.0, 2.0);
  g0 ~ normal(0.0, 2.0);
  g1_z ~ normal(0.0, 2.0);
  
  //Log Likelihood
  target += log_sum_exp(gamma[T]);
}
generated quantities {
  //State probabilities
  vector[K] alpha[T];
  for (t in 1:T)
    alpha[t] = softmax(to_vector(gamma[t]) );
}
"


model2<-stan_model(model_code=model2.str)


# Optimization -------------------------------------------------------------------
fit.opt<-optimizing(object=model2, data=stan_dat )
t(fit.opt$value)

pars$b0
fit.opt$par[grep("b0",names(fit.opt$par))]

pars$b1
matrix(fit.opt$par[grep("b1",names(fit.opt$par))],pars$J, pars$K)

pars$g1
matrix(fit.opt$par[grep("g1_z",names(fit.opt$par))],1, pars$K-1)

pars$g0
A0.est<-matrix(fit.opt$par[grep("A0",names(fit.opt$par))], pars$K, pars$K)
apply(A0.est,2,softmax)


dat$z
alpha<-matrix(fit.opt$par[grep("alpha",names(fit.opt$par))],nrow=pars$time, ncol=3)

par(mfrow=c(3,1),mai=c(.1,.5,.1,.1))
plot(as.numeric(dat$z==1), type="l", col="black")
lines(alpha[,1], col="darkgreen")
plot(as.numeric(dat$z==2), type="l", col="black")
lines(alpha[,2], col="blue")
plot(as.numeric(dat$z==3), type="l", col="black")
lines(alpha[,3], col="red")
abline(v=which(dat$Cov.2==1), col="red", lty=2, lwd=2)

z.est<-apply(alpha,1,function(x)which(x==max(x)))
plot(dat$z, type="o")
lines(z.est+.03, col="red", lwd=2)






# Sampling ----------------------------------------------------------------
fit.samp<-sampling(object=model2, data=stan_dat, 
                   chains=5, cores=5, warmup=300, iter=1000,
                   control=list(max_treedepth=6, adapt_delta=.9),
                   include=F, pars=c("acc","gamma", "g1", "A"))
draws<-as.data.frame(fit.samp)

hist(draws$lp__)


# Examine results ---------------------------------------------------------
pars$g0
matrix(colMeans(draws)[grep("A0",colnames(draws))],pars$K, pars$K)

pars$b0
colMeans(draws)[grep("b0",colnames(draws))]

pars$b1
matrix(colMeans(draws)[grep("b1",colnames(draws))],pars$J, pars$K)


hist(draws$`A0[2,1]`, breaks=48)
hist(draws$`b1[1,1]`, breaks=48)


pairs(draws[,grep("b0",colnames(draws))], pch=".", cex=2)
pairs(draws[,grep("b1",colnames(draws))], pch=".", cex=2)
pairs(draws[,grep("g0",colnames(draws))], pch=".", cex=2)
pairs(draws[,grep("g1",colnames(draws))], pch=".", cex=2)

alpha<-draws[grep("alpha",colnames(draws))]
a1<-alpha[,1:pars$time]
a2<-alpha[,1:pars$time+pars$time]
a3<-alpha[,1:pars$time+pars$time*2]

par(mfrow=c(3,1),mai=c(.1,.5,.1,.1))
plot(as.numeric(dat$z==1), type="l", col="black")
lines(colMeans(a1), col="darkgreen")
plot(as.numeric(dat$z==2), type="l", col="black")
lines(colMeans(a2), col="blue")
plot(as.numeric(dat$z==3), type="l", col="black")
lines(colMeans(a3), col="red")


z.est<-apply(cbind(colMeans(a1),colMeans(a2),colMeans(a3)),1,function(x)which(x==max(x)))
plot(dat$z, type="l")
lines(z.est, col="red")

