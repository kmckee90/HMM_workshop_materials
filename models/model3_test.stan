data { 
  int N; // number of people
  int K; // number of hidden states
  int J; // number of covariates
  int T; // length of series
  int Y [N, T]; // response
  matrix[T, J] X [N]; //covariate
  int indMat [K,K];
  
  int Y_test [N, T]; // response
  matrix[T, J] X_test [N]; //covariate
  
  vector[K] ratePriorMu;
  vector[K] ratePriorSD;
}
parameters{
  ordered[K] b0_mu;
  vector<lower=0>[K] b0_sd;
  vector[K] b0_z [N];
  
  matrix[J,K] b1_mu;
  matrix<lower=0>[J,K] b1_sd;
  matrix[J,K] b1_z [N];
  
  vector[K*(K-1)] g0_mu;
  vector<lower=0>[K*(K-1)] g0_sd;
  vector[K*(K-1)] g0_z [N];
  
  matrix[J,K-1] g1_mu;
  matrix<lower=0>[J,K-1] g1_sd;
  matrix[J,K-1] g1_z [N];
}
transformed parameters{
  real acc[K];
  real gamma[N, T, K];
  matrix[K, K] A0 [N];
  matrix[K, K] A;

  //Parameter distributions
  vector[K]  b0 [N];
  matrix[J,K] b1 [N];
  vector[K*(K-1)] g0 [N];
  matrix[J,K-1] g1 [N];
  matrix[J,K] g1m = rep_matrix(0.0, J, K);

  //cycle through people
  for(n in 1:N){
    //Generate parameters from standard scores
    b0[n] = b0_mu + b0_sd .*b0_z[n]; 
    b1[n] = b1_mu + b1_sd .*b1_z[n]; 
    g0[n] = g0_mu + g0_sd .*g0_z[n]; 
    g1[n] = g1_mu + g1_sd .*g1_z[n]; 
    g1m[,2:K] = g1[n];
  
  
    //Set up base transition matrix
     for(j in 1:K){
       for(k in 1:K){
         if(j!=k){
            A0[n,j,k] = g0[n, indMat[j,k]];
         }else{
            A0[n,j,k] = 3.0;
         }
       }
     }

  
    //First point
    for (k in 1:K){
      gamma[n,1, k] = log(1.0/K) + bernoulli_logit_lpmf(Y[n,1] | b0[n,k] + X[n,1,]*b1[n,,k]);
    }
  
    //Series after first
    for (t in 2:T) {  
      //softmax regression: How best to constrain it?
      A = A0[n];
      for(k in 1:K){
        A[k,] = A[k,] + X[n,t,]*g1m[,k];
      }
      for(k in 1:K){
        A[,k] = softmax(A[,k]);
      }
      
      //Logic: First loop through prev state probability.
      // For each previous state, add in log probs of each possible new state.
      for (k in 1:K) { 
         for (j in 1:K) {
            acc[j] = gamma[n, t-1, j] + log(A[k, j]) + bernoulli_logit_lpmf(Y[n, t] | b0[n, k] + X[n, t,]*b1[n, ,k]);
         }
         gamma[n, t, k] = log_sum_exp(acc);
      }
    }
  }
}

model {
  // priors
  b0_mu ~ normal(ratePriorMu, ratePriorSD);
  to_vector(b1_mu) ~ normal(0.0, 5.0);
  g0_mu ~ normal(0.0, 10.0);
  to_vector(g1_mu) ~ normal(0.0, 10.0);

  b0_sd ~ gamma(2.0, 10.0);
  to_vector(b1_sd) ~ gamma(2.0, 10.0);
  g0_sd ~ gamma(2.0, 10.0);
  to_vector(g1_sd) ~ gamma(2.0, 10.0);
  
  for(n in 1:N){
    b0_z[n] ~ std_normal();
    to_vector(b1_z[n]) ~ std_normal();
    g0_z[n] ~ std_normal();
    to_vector(g1_z[n]) ~ std_normal();
    
    //Log Likelihood
    target += log_sum_exp(gamma[n,T,]);
  }
}

generated quantities {
  //State probabilities
  vector[K] alpha[N, T];
  vector[K] alpha_test[N, T];
  real gamma_test[N, T, K];
  vector[N] log_lik;
  vector[N] log_lik_test;
  matrix[J,K] g1m_test = rep_matrix(0.0, J, K);
  matrix[K, K] A_test;
  real acc_test[K];


  for(n in 1:N){
    log_lik[n] = log_sum_exp(gamma[n,T,]);
    for (t in 1:T) {
      alpha[n,t] = softmax(to_vector(gamma[n,t,]) );
    }
  }
  
 
  //cycle through people
  for(n in 1:N){
    g1m_test[,2:K] = g1[n];
    
    //First point
    for (k in 1:K){
      gamma_test[n,1, k] = log(1.0/K) + bernoulli_logit_lpmf(Y_test[n,1] | b0[n,k] + X_test[n,1,]*b1[n,,k]);
    }
  
    //Series after first
    for (t in 2:T) {  
      //softmax regression: How best to constrain it?
      A_test = A0[n];
      for(k in 1:K){
        A_test[k,] = A_test[k,] + X_test[n,t,]*g1m_test[,k];
      }
      for(k in 1:K){
        A_test[,k] = softmax(A_test[,k]);
      }
      
      //Logic: First loop through prev state probability.
      // For each previous state, add in log probs of each possible new state.
      for (k in 1:K) { 
         for (j in 1:K) {
            acc_test[j] = gamma_test[n, t-1, j] + log(A_test[k, j]) + bernoulli_logit_lpmf(Y_test[n, t] | b0[n, k] + X_test[n, t,]*b1[n, ,k]);
         }
         gamma_test[n, t, k] = log_sum_exp(acc_test);
      }
    }
  }
  
  for(n in 1:N){
    log_lik_test[n] = log_sum_exp(gamma_test[n,T,]);
    for (t in 1:T) {
      alpha_test[n,t] = softmax(to_vector(gamma_test[n,t,]) );
    }
  }
  
  
}

