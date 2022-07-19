data {
  int<lower = 0> N; // number of observations
  vector[N] food; // food availability
  vector[N] ncatch; // catch
  vector[N] growth; // log population growth; response
}

parameters {
  vector[3] beta; // intercept & two slopes
  real<lower = 0> sigma;
}

model{
  // standard weakly information priors 
  // see https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
  beta ~ normal(0, 2);
  //sigma ~ exponential(20);
  sigma ~ gamma(5, 20);
  growth ~ normal(beta[1] + beta[2] * food + beta[3] * ncatch, sigma);
}

//for getting ELPD / WAIC
generated quantities {
  vector[N] log_lik;
  for(n in 1:N){
    log_lik[n] = normal_lpdf(growth[n] | beta[1] + beta[2] * food[n] + beta[3] * ncatch[n], sigma);
  }
}
