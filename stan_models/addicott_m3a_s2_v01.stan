data {
  int<lower = 0> N; // number of observations
  vector[N] growth; // log population growth; response
  vector[N] food; // food availability
  vector[N] p_ncatch; // predicted catch (catch hat); posterior mean from first stage
}

parameters {
  vector[3] beta; // intercept & two slopes
  real<lower = 0> sigma;
}

model{
  // standard weakly information priors 
  // see https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
  beta ~ normal(0, 2);
  sigma ~ gamma(5, 20);
  growth ~ normal(beta[1] + beta[2] * food + beta[3] * p_ncatch, sigma);
}

generated quantities {
  vector[N] log_lik;
  //for getting ELPD / WAIC
  for(n in 1:N){
    log_lik[n] = normal_lpdf(growth[n] | beta[1] + beta[2] * food[n] + beta[3] * p_ncatch[n], sigma);
  }
}
