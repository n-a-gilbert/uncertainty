data {
  int<lower = 0> N;
  vector[N] food; // food availability
  vector[N] ncatch; // catch
  vector[N] nets; // nets
  vector[N] growth; // log pop growth
}

parameters {
  vector[3] alpha; // for the catch submodel; intercept & 2 slopes
  real<lower = 0> sigma1; // we have to have separate variance parameters for the submodels
  vector[N] p_ncatch; // predicted catch
  
  vector[3] beta; // for the growth submodel; intercept & 2 slopes
  real<lower = 0> sigma2;
}

model{
  alpha ~ normal(0, 2); 
  sigma1 ~ exponential(1);
  
  beta ~ normal(0, 2);
  sigma2 ~ exponential(1);
  
  ncatch ~ normal(alpha[1] + alpha[2] * nets + alpha[3] * food, sigma1);
  
  p_ncatch ~ normal(alpha[1] + alpha[2] * nets + alpha[3] * food, sigma1);
  
  growth ~ normal(beta[1] + beta[2] * food + beta[3] * p_ncatch, sigma2);
}

generated quantities {
  vector[N] log_lik;
  for(n in 1:N){
    log_lik[n] = normal_lpdf(growth[n] | beta[1] + beta[2] * food[n] + beta[3] * p_ncatch[n], sigma2);
  }
}
