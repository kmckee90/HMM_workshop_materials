# -------------------------------------------------------------------------
#  Hierarchical Basic Hidden Markov Process with logistic regression submodels
#  Author: Kevin L. McKee (klmckee@ucdavis.edu)
#  Description: This script produces simulated data for a basic Hidden markov model, then fits the same model in Stan.
# -------------------------------------------------------------------------


set.seed(123)
# Set up parameters -------------------------------------------------------
nState<-3
nCov<-1
time<-300
b0<-c(-2, -2, 2)
# b1<-matrix(rnorm(nState*nCov), nCov, nState)
b1<-matrix(c(4,0,-4), nCov, nState)

# Transition matrix -------------------------------------------------------
A<-matrix(1, nState, nState)
diag(A)<-50
A<-apply(A,2,function(u) u/sum(u))

# Generate data -----------------------------------------------------------
z<-matrix(1, ncol=1, nrow=time)                          #Decision state
y<-p<-matrix(0, ncol=1, nrow=time)                       #Response
X<-matrix(rbinom(time*nCov, size=1, .5), ncol=nCov, nrow=time)   #Time varying covariate

for(i in 2:time){
  z[i]<-sample(1:nState, 1, p=A[,z[i-1]])
  p[i]<-plogis(b0[z[i]]+ X[i,] %*% b1[,z[i]] )
  y[i]<-rbinom(1,1,p[i])
}

plot(z, type="o", col="blue", ylim=c(1,nState))
lines(y+1, type="h", col="red")
lines(X[,1]*.5+1, type="h", col="blue")

stan_dat<-list(
  "K"=nState,
  "J"=nCov,
  "T"=time,
  "X"=X,
  "Y"=c(y),
  "dprior"=diag(20,3,3)+2
)


# Stan model ---------------------------------------------------------------
require(rstan)

model1.str<-
  "
data { 
  int K; // number of hidden states
  int J; // number of covariates
  int T; // length of series
  int Y [T]; // response
  matrix[T, J] X; //covariate
  vector[K] dprior [K];
}
parameters{
  ordered[K] b0;
  matrix[J,K] b1;
  simplex[K] A[K];  // transit probs
}
transformed parameters{
  real acc[K];
  real gamma[T, K];
  
  //First point
  for (k in 1:K){
    gamma[1, k] = bernoulli_logit_lpmf(Y[1] | b0[k] + X[1,]*b1[,k]);
  }
  
  //Series after first
  for (t in 2:T) {
    for (k in 1:K) {
       for (j in 1:K) {
          acc[j] = gamma[t-1, j] + log(A[j, k]) + bernoulli_logit_lpmf(Y[t] | b0[k] + X[t,]*b1[,k]);
       }
       gamma[t, k] = log_sum_exp(acc);
    }
  }
}
model {
  // priors
  b0 ~ normal(0.0, 2.0);
  for(k in 1:K){
    A[k] ~ dirichlet(dprior[k]);
  }
  to_vector(b1) ~ normal(0.0, 2.0);

  target += log_sum_exp(gamma[T]);
}

generated quantities {
  vector[K] alpha[T];
  for (t in 1:T)
    alpha[t] = softmax(to_vector(gamma[t]) );
}
"


model1<-stan_model(model_code=model1.str)


# Optimization -------------------------------------------------------------------
fit.opt<-optimizing(object=model1, data=stan_dat )
# fit.opt<-optimizing(object=model1, data=stan_dat, init=as.list(fit.opt$par+rnorm(length(fit.opt$par))) )

b0
fit.opt$par[grep("b0",names(fit.opt$par))]

b1
matrix(fit.opt$par[grep("b1",names(fit.opt$par))],nCov, nState)

A
matrix(fit.opt$par[grep("A",names(fit.opt$par))],nState, nState)


z
alpha<-matrix(fit.opt$par[grep("alpha",names(fit.opt$par))],nrow=time, ncol=3)

par(mfrow=c(3,1),mai=c(.1,.5,.1,.1))
plot(as.numeric(z==1), type="l", col="black")
lines(alpha[,1], col="darkgreen")
plot(as.numeric(z==2), type="l", col="black")
lines(alpha[,2], col="blue")
plot(as.numeric(z==3), type="l", col="black")
lines(alpha[,3], col="red")

z.est<-apply(alpha,1,function(x)which(x==max(x)))
plot(z, type="l")
lines(z.est, col="red")





# Sampling ----------------------------------------------------------------
fit.samp<-sampling(object=model1, data=stan_dat, 
                   chains=5, cores=5, warmup=300, iter=600,
                   control=list(max_treedepth=6, adapt_delta=.9),
                   include=F, pars=c("acc","gamma"))
draws<-as.data.frame(fit.samp)

hist(draws$lp__)


# Examine results ---------------------------------------------------------
A
matrix(colMeans(draws)[grep("A",colnames(draws))],nState, nState)

b0
colMeans(draws)[grep("b0",colnames(draws))]

b1
matrix(colMeans(draws)[grep("b1",colnames(draws))],nCov, nState)


hist(draws$`A[3,3]`)
hist(draws$`b1[1,3]`)
plot(draws[,1:3])

alpha<-draws[grep("alpha",colnames(draws))]
a1<-alpha[,1:time]
a2<-alpha[,1:time+time]
a3<-alpha[,1:time+time*2]

par(mfrow=c(3,1),mai=c(.1,.5,.1,.1))
plot(as.numeric(z==1), type="l", col="black")
lines(colMeans(a1), col="darkgreen")
plot(as.numeric(z==2), type="l", col="black")
lines(colMeans(a2), col="blue")
plot(as.numeric(z==3), type="l", col="black")
lines(colMeans(a3), col="red")


z.est<-apply(cbind(colMeans(a1),colMeans(a2),colMeans(a3)),1,function(x)which(x==max(x)))
plot(z, type="l")
lines(z.est, col="red")




# Variational Bayes -------------------------------------------------------
fit.vb<-vb(object=model1, data=stan_dat) 
draws<-as.data.frame(fit.vb)

A
matrix(colMeans(draws)[grep("A",colnames(draws))],nState, nState)

b0
colMeans(draws)[grep("b0",colnames(draws))]

b1
matrix(colMeans(draws)[grep("b1",colnames(draws))],nCov, nState)


hist(draws$`A[3,3]`)
hist(draws$`b0[1]`)
plot(draws[,1:3])

alpha<-draws[grep("alpha",colnames(draws))]
a1<-alpha[,1:300]
a2<-alpha[,301:600]
a3<-alpha[,601:900]

par(mfrow=c(3,1),mai=c(.1,.5,.1,.1))
plot(as.numeric(z==2), type="l", col="black")
lines(colMeans(a1), col="darkgreen")
plot(as.numeric(z==1), type="l", col="black")
lines(colMeans(a2), col="blue")
plot(as.numeric(z==3), type="l", col="black")
lines(colMeans(a3), col="red")



z.est<-apply(cbind(colMeans(a1),colMeans(a2),colMeans(a3)),1,function(x)which(x==max(x)))
plot(z, type="l")
lines(z.est, col="red")




