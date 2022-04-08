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

