# -------------------------------------------------------------------------
#  Hierarchical Hidden Markov Process with State Prediction: Data Simulation
#  Author: Kevin L. McKee (klmckee@ucdavis.edu)
#  Description: This script produces simulated data for testing the HHMM.
#  Response variables and covariates are binary. The submodel structure is logistic regression. 
# -------------------------------------------------------------------------
source("functions.R")

set.seed(123)

# Input params ------------------------------------------------------------
#Set up data and model dimensions first:
nCov<-2
nState<-3
nSubj<-12
nObs<-150


genSets<-c("sim", "test") #Generate two data sets, one for estimation and one for testing

# Data-generating parameters
pars<-list(
  "N" = nSubj, #Number of people
  "time"=nObs, #Observations per person 
  "J"=nCov,    #Number of covariates
  "K"=nState,  #Number of hidden states underlying data
  
#These parameters govern what happens within each sub-model (ie hidden state). Columns correspond to submodels, rows correspond to covariates.
  "b0_mu"=as.matrix(c(-2, 0, 2)),  #Base response rates
  "b0_sd"=c(.1,  .1, .1),          #Variation across subjects in the above effects (As std. deviation.)
  
  "b1_mu"=rbind(c(4, 0, -4),0),    #Regressions on covariates
  "b1_sd"=rbind(c(.5, .1, .5),0),    #Variation across subjects in the above effects (As std. deviation.)

#These parameters govern the   
  "g1_mu"=rbind(0, c(0,  3,  0)), #Effect of covariates on transitions
  "g1_sd"=rbind(0, c(0,  .5,  0))  #Variation across subjects in the above effects (As std. deviation.)
)

#Here we set up the base transition matrix.  
#The values of the transition matrix are arbitrary in scale and mean, as they will be normalized by softmax when in use. 
#In this matrix, the diagonal is much larger than the off-diagonals to make states persistent over time. 
g0_mu<-matrix(1, pars$K, pars$K)
diag(g0_mu)<-5 #Larger values on the matrix diagonal result in more persistent states. 
pars$g0_mu<-g0_mu
pars$g0_sd<-matrix(.1, pars$K, pars$K) #Minor variation in transition parameters across people



# Person parameters -------------------------------------------------------
#Here we format the parameters in arrays so that we can generate arrays of random individual parameter values, given the group's mean and standard deviation.
b0_z.mu<-array(pars$b0_mu, dim=c(dim(pars$b0_mu),pars$N))
b1_z.mu<-array(pars$b1_mu, dim=c(dim(pars$b1_mu),pars$N))
g0_z.mu<-array(pars$g0_mu, dim=c(dim(pars$g0_mu),pars$N))
g1_z.mu<-array(pars$g1_mu, dim=c(dim(pars$g1_mu),pars$N))

b0_z.sd<-array(pars$b0_sd, dim=c(dim(pars$b0_sd),pars$N))
b1_z.sd<-array(pars$b1_sd, dim=c(dim(pars$b1_sd),pars$N))
g0_z.sd<-array(pars$g0_sd, dim=c(dim(pars$g0_sd),pars$N))
g1_z.sd<-array(pars$g1_sd, dim=c(dim(pars$g1_sd),pars$N))

b0_z<-array(rnorm(length(b0_z.mu),b0_z.mu, b0_z.sd), dim(b0_z.mu))
b1_z<-array(rnorm(length(b1_z.mu),b1_z.mu, b1_z.sd), dim(b1_z.mu))
g0_z<-array(rnorm(length(g0_z.mu),g0_z.mu, g0_z.sd), dim(g0_z.mu))
g1_z<-array(rnorm(length(g1_z.mu),g1_z.mu, g1_z.sd), dim(g1_z.mu))



# Generate data -----------------------------------------------------------
#Normalization for multinomial or categorical distributions
softmax<-function(x) exp(x)/sum(exp(x))

#Data-generating model. Takes the parameter list and uses it as an internal environment.
genData <- function(pars) {
  with(as.list(pars),{   #Converts list input to local environment
    z<-matrix(1, ncol=1, nrow=time)        #Hidden state vector
    y<-p<-matrix(0, ncol=1, nrow=time)     #Response vector
    X<-cbind(                             #Covariate matrix
      rbinom(time, size=1, .5),   # Rapidly changing binary covariate: 50% chance. Used to define response strategies within sub-models.
      rbinom(time, size=1, .025)) # Sparse binary covariate, 2.5% chance. Will be used to trigger state changes between sub-models.
    
    for(i in 2:time){
      A0<-g0             #Base transition matrix
      for(j in 1:K) A0[j,]<-A0[j,] + c(X[i,] %*% g1[,j]) #Add the effects of covariates. 
      A<-apply(A0,2,softmax)   #Normalize to get the final transition matrix
      
      z[i]<-sample(1:K, 1, p=A[,z[i-1]])   #Generate the next hidden state using the transition probabilities
      p[i]<-plogis(b0[z[i]]+ X[i,] %*% b1[,z[i]] )  #Generate response probabilities using the logistic model associated with current hidden state 
      y[i]<-rbinom(1,1,p[i])  #Generate actual responses from the probabilities
    }
    return(data.frame("Response"=y,"Cov"=X, "z"=z, "p"=p))
  })
}

# Hierarchical data generation: generate parameters, then data from params ---
#The above data-generating function only operates for one person at a time.
#Since we produced arrays of individual parameters, we can run through the arrays 
#and generate data associated with each parameter set: 
for(dataSet in genSets){
  hdat<-array(NA, dim=c(pars$N, pars$time, 5)) #Create an array for the complete data set across subjects
  for(i in 1:pars$N){
    pars.p<-c(pars[1:4],list( #Take values from arrays, put in list format for data-generating function
      "b0"=b0_z[,,i],
      "b1"=b1_z[,,i],       
      "g0"=g0_z[,,i],
      "g1"=g1_z[,,i]
      ))
    dat<-genData(pars.p) #Generate data
    hdat[i,,]<-as.matrix(dat) #Store in the final data array
  }
  
  #Plot individual hidden state trajectories
  par(mfrow=c(6,2),mai=c(.4,.6,.1,.1), mgp=c(2,1,0))
  for(i in 1:nSubj) {
    plot(hdat[i,,4], type="l", ylim=c(1,3))
    lines(hdat[i,,1]*2, type="h", col="green")
    lines(hdat[i,,2]*1.5, type="h", col="red")
    
    # abline(v=which(hdat[i,,2]==1), col="gray")
    # abline(v=which(hdat[i,,1]==1)+.1, col="blue")
    abline(v=which(hdat[i,,3]==1), col="red", lty=2)
    lines(hdat[i,,4], lwd=2)
    
  }
  
  # Save simulated data to realistic format (csv) ---------------------------
  for(i in 1:pars$N){
    outData<-hdat[i,,1:3]
    colnames(outData)<-c("Response","Covariate1","Covariate2")
    write.csv(outData, file=paste0(dataSet, " data/subj_",i,".csv"), row.names = F)
  }
}


