# -------------------------------------------------------------------------
#  Hierarchical Basic Hidden Markov Process with logistic regression submodels
#  Author: Kevin L. McKee (klmckee@ucdavis.edu)
#  Description: This script produces simulated data for a basic Hidden markov model, then fits the same model in Stan.
# -------------------------------------------------------------------------


set.seed(123)
# Set up parameters -------------------------------------------------------
nState<-3  #Number of hidden states
nCov<-1    #Number of covariates
time<-300  #Observations per person; this model only fits data from a single person / series.
b0<-c(-2, 0, 2) #Base response rates
b1<-matrix(c(4,0,-4), nCov, nState) #Regressions on the covariate

# Transition matrix -------------------------------------------------------
A<-matrix(1, nState, nState)
diag(A)<-50
A<-apply(A,2,function(u) u/sum(u))

# Generate data -----------------------------------------------------------
z<-matrix(1, ncol=1, nrow=time)           #Decision state
y<-p<-matrix(0, ncol=1, nrow=time)        #Response
X<-matrix(rbinom(time*nCov, size=1, .5), ncol=nCov, nrow=time)   #Time varying covariate

for(i in 2:time){
  z[i]<-sample(1:nState, 1, p=A[,z[i-1]])  # Draw a random hidden state index using the transition matrix and previous hidden state.
  p[i]<-plogis(b0[z[i]]+ X[i,] %*% b1[,z[i]] ) #Compute response probabilities given that state.
  y[i]<-rbinom(1,1,p[i]) #Generate responses from response probabilities
}

#Plot hidden state and responses:
plot(z, type="o", col="blue", ylim=c(1,nState)) #hidden state
lines(y+1, type="h", col="red") #Response
lines(X[,1]*.5+1, type="h", col="blue") #Covariate


#Data object for Stan model later. These values are named within the stan model and provided externally as a list.
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

#Models are written in the Stan language, compiled, and run in R.

model1.str<-
  "
data { 
  int K; // number of hidden states
  int J; // number of covariates
  int T; // length of series
  int Y [T]; // response
  matrix[T, J] X; //covariate
  vector[K] dprior [K]; //prior distribution for the transition matrix
}
parameters{
  ordered[K] b0; // Base rates are constrained to an ordered vector to make sure the hidden states are ranked
  matrix[J,K] b1; 
  simplex[K] A[K];  // transit probs
}
transformed parameters{
  real acc[K];
  real gamma[T, K];
  
  //Likelihood of first point
  for (k in 1:K){
    gamma[1, k] = bernoulli_logit_lpmf(Y[1] | b0[k] + X[1,]*b1[,k]);
  }
  
  //Likelihood of each successive point. 
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


  //Log-likelihoods are summed in 'target' for use in MCMC.
  target += log_sum_exp(gamma[T]);
}

generated quantities {
 //Hidden state probabilities calculated post-hoc from the state likelihoods at each time.
  vector[K] alpha[T];
  for (t in 1:T)
    alpha[t] = softmax(to_vector(gamma[t]) );
}
"

#Compile model
model1<-stan_model(model_code=model1.str)


# Optimization -------------------------------------------------------------------
#Optimization uses gradient descent and is likely to converge to a local solution. 
#Re-run multiple times and compare the likelihood of each run.

fit.opt<-optimizing(object=model1, data=stan_dat )
fit.opt$value

#Compare base rates
b0
fit.opt$par[grep("b0",names(fit.opt$par))]

#Compare regression coefficients
b1
matrix(fit.opt$par[grep("b1",names(fit.opt$par))],nCov, nState)

#Transition matrix
A
matrix(fit.opt$par[grep("A",names(fit.opt$par))],nState, nState)

alpha<-matrix(fit.opt$par[grep("alpha",names(fit.opt$par))],nrow=time, ncol=3)

par(mfrow=c(3,1),mai=c(.1,.5,.1,.1))
plot(as.numeric(z==1), type="l", col="black")
lines(alpha[,1], col="darkgreen")
plot(as.numeric(z==2), type="l", col="black")
lines(alpha[,2], col="blue")
plot(as.numeric(z==3), type="l", col="black")
lines(alpha[,3], col="red")


# Sampling ----------------------------------------------------------------
#Sampling uses Markov Chain Monte Carlo (MCMC) to obtain an approximation of the complete posterior distribution.
#Hence it takes much longer, but provides more output information.

fit.samp<-sampling(object=model1, data=stan_dat, 
                   chains=5, cores=5, warmup=300, iter=600,
                   control=list(max_treedepth=6, adapt_delta=.9),
                   include=F, pars=c("acc","gamma"))
draws<-extract(fit.samp)

# Examine results ---------------------------------------------------------
#Transition matrix parameters
A
colMeans(draws$A)
pairs(draws$A, pch=".")


#Regression parameter distributions:
b0
colMeans(draws$b0)
pairs(draws$b0, pch=".", cex=3)

b1
colMeans(draws$b1)
pairs(draws$b1, pch=".", cex=3)


#Extract and plot the hidden state probabilities
alpha<-colMeans(draws$alpha)
par(mfrow=c(3,1),mai=c(.1,.5,.1,.1))
plot(as.numeric(z==1), type="l", col="black")
lines(alpha[,1], col="darkgreen")
plot(as.numeric(z==2), type="l", col="black")
lines(alpha[,2], col="blue")
plot(as.numeric(z==3), type="l", col="black")
lines(alpha[,3], col="red")



# Variational Bayes -------------------------------------------------------
# This method is a combination of MCMC and optimization. It uses gradient descent to fit an approximate posterior distribution.
fit.vb<-vb(object=model1, data=stan_dat) 
draws<-extract(fit.vb)

# Examine results ---------------------------------------------------------
#Transition matrix parameters
A
colMeans(draws$A)
pairs(draws$A, pch=".")


#Regression parameter distributions:
b0
colMeans(draws$b0)
pairs(draws$b0, pch=".", cex=3)

b1
colMeans(draws$b1)
pairs(draws$b1, pch=".", cex=3)


#Extract and plot the hidden state probabilities
alpha<-colMeans(draws$alpha)
par(mfrow=c(3,1),mai=c(.1,.5,.1,.1))
plot(as.numeric(z==1), type="l", col="black")
lines(alpha[,1], col="darkgreen")
plot(as.numeric(z==2), type="l", col="black")
lines(alpha[,2], col="blue")
plot(as.numeric(z==3), type="l", col="black")
lines(alpha[,3], col="red")