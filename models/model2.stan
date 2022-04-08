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