data {
  // number of observations
  int<lower=0> N;
  // number of columns in design matrix excluding A
  int<lower=0> P;
  // design matrix, excluding A, M, U
  matrix[N, P] X;
  // observed treatment
  vector[N] A;
  // observed mediator
  int<lower=0,upper=1> M[N];
  // outcome
  int<lower=0,upper=1> Y[N];
  // mean of regression priors
  vector[P + 3] alpha_m;
  vector[P + 2] beta_m;
  vector[P] gamma_m;
  // variance-covariance of regression priors
  cov_matrix[P + 3] alpha_vcv;
  cov_matrix[P + 2] beta_vcv;
  cov_matrix[P] gamma_vcv;
}

transformed data {
  // vectors of ones and zeros
  vector[N] all_ones = rep_vector(1, N);
  vector[N] all_zeros = rep_vector(0, N);
  
  // make vector of 1/N for (classical) bootstrapping
  real one = 1;
  vector[N] boot_probs = rep_vector(one/N, N);
  
  // make vector version of M
  vector[N] Mv = to_vector(M);
}

parameters {
  // regression coefficients (confounder model)
  vector[P] gamma;
  
  // regression coefficients (outcome model)
  vector[P + 3] alpha;
  
  // regression coefficients (mediator model)
  vector[P + 2] beta;
}

transformed parameters {
  // P(U = 1) for mixture weights
  vector[N] pU1 = inv_logit(X * gamma);
  
  // partial M coefficient parameters
  vector[P] betaZ = head(beta, P);
  real betaU = beta[P + 1];
  real betaA = beta[P + 2];
  
  // partial Y coefficient parameters
  vector[P] alphaZ = head(alpha, P);
  real alphaU = alpha[P + 1];
  real alphaA = alpha[P + 2];
  real alphaM = alpha[P + 3];
}

model {
  // informative priors
  alpha ~ multi_normal(alpha_m, alpha_vcv);
  beta  ~ multi_normal(beta_m, beta_vcv);
  gamma ~ multi_normal(gamma_m, gamma_vcv);
  
  // likelihoods
  M ~ bernoulli(inv_logit(X * betaZ + A * betaA + betaU).*pU1 + 
                  inv_logit(X * betaZ + A * betaA).*(1-pU1));
  Y ~ bernoulli(inv_logit(X * alphaZ + A * alphaA + Mv * alphaM + alphaU).*pU1 +
                  inv_logit(X * alphaZ + A * alphaA + Mv * alphaM).*(1-pU1));
}

generated quantities {
  // weights for the bootstrap
  int<lower=0> counts[N] = multinomial_rng(boot_probs, N);
  
  // calculate NDE in the bootstrapped sample
  real NDE = 0;
  vector[N] U;
  vector[N] M_a0;
  vector[N] Y_a1Ma0;
  vector[N] Y_a0Ma0;
  for (n in 1:N) {
    // sample U
    U[n] = bernoulli_logit_rng(pU1[n]);
    
    // sample Ma where a = 0
    M_a0[n] = bernoulli_rng(inv_logit(X[n] * betaZ + A[n] * betaA + betaU)*pU1[n] + 
                              inv_logit(X[n] * betaZ + A[n] * betaA)*(1-pU1[n]));
    
    // sample Y_(a=1, M=M_0) and Y_(a=0, M=M_0)
    Y_a1Ma0[n] = bernoulli_rng(inv_logit(X[n] * alphaZ + alphaA + M[n] * alphaM + alphaU)*pU1[n] +
                                 inv_logit(X[n] * alphaZ + alphaA + M[n] * alphaM)*(1-pU1[n]));
    Y_a0Ma0[n] = bernoulli_rng(inv_logit(X[n] * alphaZ + M[n] * alphaM + alphaU)*pU1[n] +
                                 inv_logit(X[n] * alphaZ + M[n] * alphaM)*(1-pU1[n]));
    
    // add this observation's contribution to the bootstrapped NDE
    NDE = NDE + (counts[n] * (Y_a1Ma0[n] - Y_a0Ma0[n]))/N;
  }
}

