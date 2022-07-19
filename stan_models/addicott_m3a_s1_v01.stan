data {
  int<lower = 0> N; // number of observations
  vector[N] ncatch; // catch; response
  vector[N] nets; // nets; covariate
  vector[N] food; // food availability
}

parameters {
  //using alpha here since the next model will have two submodels
  vector[3] alpha; // intercept & two slopes
  real<lower = 0> sigma;
}

model{
  // standard weakly information priors 
  // see https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
  alpha ~ normal(0, 2);
  //sigma ~ gamma(5, 20);
  sigma ~ exponential(20);
  ncatch ~ normal(alpha[1] + alpha[2] * nets + alpha[3] * food, sigma);
}

generated quantities {
  vector[N] log_lik;
  real  p_ncatch[N];
  //for getting ELPD / WAIC
  for(n in 1:N){
    log_lik[n] = normal_lpdf(ncatch[n] | alpha[1] + alpha[2] * nets[n] + alpha[3] * food[n], sigma);
  }
  //predicted values
  p_ncatch = normal_rng(alpha[1] + alpha[2] * nets + alpha[3] * food, sigma);
}
