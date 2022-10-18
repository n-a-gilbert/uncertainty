# 19 july 2022
# Neil Gilbert
# Run models from Addicott et al. (2022) FEE in Stan
### to evaluate consequences of not propagating uncertainty 

library(tidyverse)
library(rstan)
library(here)
library(loo)
library(janitor)
library(MetBrewer)

#NG - three sample sizes to evaluate
sample_sizes <- c(25, 100, 1000)

# data simulation from Addicott et al.
set.seed(1)

# Initialize Parameters and Simulate Data
n <- sample_sizes[3] # number of data points to construct

error.var.1 <- 0.01 # variance of the idiosyncratic error term
error.var.2 <- 0.2 # variance of error process for measurements of catch

true.beta <- 0.5 # coefficient on food availability term
true.gamma <- -0.08 # coefficient on fishing effort term

# Sample food availability data from uniform distribution
food <- (sample.int(101, size = n, replace = TRUE) + 50) / 200

# Sample fishing effort data similarly
effort <- (sample.int(201, size = n, replace = TRUE) - 1) / 100

# # Introduce multicollinearity
# food <- food+ 0.25*effort

# Sample fishing net data that will be used later for the IV approach
nets <- (sample.int(201, size = n, replace = TRUE) - 1) / 100

# Generate random normal error values for logpopgrowth and catch 
error.normal <- rnorm(n, 0, error.var.1)
error.normal2 <- rnorm(n, 0, error.var.2)

# Define data generating process for log population growth rate
logpopgrowth <- true.beta * food + true.gamma * effort + error.normal 

# Define data generating process for the measure of observed catch from fishing
catch <- effort + effort * 5*logpopgrowth +0.3 * nets + error.normal2

df <- tibble(
  growth = logpopgrowth, 
  food = food, 
  ncatch = catch,
  nets = nets)

# loop through and run the models for the three sample sizes

loo_tables <- list(list())
beta_estimates <- list(list())

for(i in 1:length(sample_sizes)) {
  
  #### Model 1 ####
  setwd(here::here("stan_models/"))
  write(
    "data {
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
  // standard weakly information priors for betas
  // see https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
  beta ~ normal(0, 2);
  // originally used weakly informative exp(1) prior for sigma
  // but had convergence issues with model #4;
  // a more informative ~gamma(5, 20) prior got that model to 
  // converge, so am using it throughout for consistency among models
  //sigma ~ exponential(1); 
  sigma ~ gamma(5, 20);
  growth ~ normal(beta[1] + beta[2] * food + beta[3] * ncatch, sigma);
}

//for getting ELPD / WAIC
generated quantities {
  vector[N] log_lik;
  for(n in 1:N){
    log_lik[n] = normal_lpdf(growth[n] | beta[1] + beta[2] * food[n] + beta[3] * ncatch[n], sigma);
  }
}", "addicott_m1_v01.stan")
  
  ni <- 4000
  nt <- 2
  nb <- ni / 2
  nc <- 4
  
  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  set.seed(123)
  
  m1_out <- stan(
    file = "addicott_m1_v01.stan",
    data = list(
      N = sample_sizes[i], 
      food = pull(slice(df, 1:sample_sizes[i]), food),
      ncatch = pull(slice(df, 1:sample_sizes[i]), ncatch), 
      growth = pull(slice(df, 1:sample_sizes[i]), growth)),
    init = lapply(1:nc, function(i)
      list(beta = rnorm(3),
           sigma = rexp(1, 1))),
    pars = c("beta", "sigma", "log_lik"),
    chains = nc, 
    iter = ni, 
    warmup = nb, 
    thin = nt)
  
  # leave one out cross validation
  m1_log_lik <- extract_log_lik(m1_out)
  m1_loo <- loo(m1_log_lik)
  
  #### Model 2 ####
  # Very similar to m1; ncatch covariate is dropped 
  setwd(here::here("stan_models/"))
  write(
    "data {
  int<lower = 0> N; // number of observations
  vector[N] food; // food availability
  vector[N] growth; // log population growth; response
}

parameters {
  vector[2] beta; // intercept and slope
  real<lower = 0> sigma;
}

model{
  // standard weakly information priors for betas
  // see https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
  beta ~ normal(0, 2);
  // originally used weakly informative exp(1) prior for sigma
  // but had convergence issues with model #4;
  // a more informative ~gamma(5, 20) prior got that model to 
  // converge, so am using it throughout for consistency among models
  //sigma ~ exponential(1); 
  sigma ~ gamma(5, 20);
  growth ~ normal(beta[1] + beta[2] * food, sigma);
}

// for getting ELPD / WAIC
generated quantities {
  vector[N] log_lik;
  for(n in 1:N){
    log_lik[n] = normal_lpdf(growth[n] | beta[1] + beta[2] * food[n], sigma);
  }
}", "addicott_m2_v01.stan")
  
  ni <- 4000
  nt <- 2
  nb <- ni / 2
  nc <- 4
  
  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  set.seed(123)
  
  m2_out <- stan(
    file = "addicott_m2_v01.stan",
    data = list(
      N = sample_sizes[i], 
      food = pull(slice(df, 1:sample_sizes[i]), food),
      # ncatch = pull(slice(df, 1:sample_sizes[3]), ncatch), 
      growth = pull(slice(df, 1:sample_sizes[i]), growth)),
    init = lapply(1:nc, function(i)
      list(beta = rnorm(2),
           # beta1 = rnorm(2), 
           # beta2 = rnorm(3),
           sigma = rexp(1, 1))),
    pars = c("beta", "sigma", "log_lik"),
    chains = nc, 
    iter = ni, 
    warmup = nb, 
    thin = nt)
  
  m2_log_lik <- extract_log_lik(m2_out)
  m2_loo <- loo(m2_log_lik)
  
  #### Model 3a ####
  # Here, we follow the 2-stage approach used by Addicott et al. 
  # First, we fit a model in which catch is the response, with nets and food as predictors
  # We generate predicted values from this model, and incorporate point estimates
  # of these predictions as a covariate in the growth model in the second stage
  # Uncertainty is not propogated
  
  # first stage
  setwd(here::here("stan_models/"))
  write(
    "data {
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
  // standard weakly information priors for alpha
  // see https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
  alpha ~ normal(0, 2);
  // originally used weakly informative exp(1) prior for sigma
  // but had convergence issues with model #4;
  // a more informative ~exp(20) prior got that model to 
  // converge, so am using it throughout for consistency among models
  //sigma ~ exponential(1); 
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
}", "addicott_m3a_s1_v01.stan")
  
  ni <- 4000
  nt <- 2
  nb <- ni / 2
  nc <- 4
  
  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  set.seed(123)
  
  m3a_s1_out <- stan(
    file = "addicott_m3a_s1_v01.stan",
    data = list(
      N = sample_sizes[i], 
      food = pull(slice(df, 1:sample_sizes[i]), food),
      ncatch = pull(slice(df, 1:sample_sizes[i]), ncatch), 
      nets = pull(slice(df, 1:sample_sizes[i]), nets)),
    init = lapply(1:nc, function(i)
      list(alpha = rnorm(3),
           sigma = rexp(1, 1))),
    pars = c("alpha", "sigma", "log_lik", "p_ncatch"),
    chains = nc, 
    iter = ni, 
    warmup = nb, 
    thin = nt)
  
  df_temp <- slice(df, 1:sample_sizes[i])
  
  # create a new dataframe with posterior means of catch predictions as a new column
  df_m3a_s2 <- summary(m3a_s1_out, 
                       pars = c("p_ncatch"))$summary %>% 
    as_tibble(rownames = "parameter") %>% 
    dplyr::select(p_ncatch = mean) %>% 
    cbind(df_temp)
  
  # second stage
  setwd(here::here("stan_models/"))
  write(
    "data {
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
  // standard weakly information priors for betas
  // see https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
  beta ~ normal(0, 2);
  // originally used weakly informative exp(1) prior for sigma
  // but had convergence issues with model #4;
  // a more informative ~gamma(5, 20) prior got that model to 
  // converge, so am using it throughout for consistency among models
  //sigma ~ exponential(1); 
  sigma ~ gamma(5, 20);
  growth ~ normal(beta[1] + beta[2] * food + beta[3] * p_ncatch, sigma);
}

generated quantities {
  vector[N] log_lik;
  //for getting ELPD / WAIC
  for(n in 1:N){
    log_lik[n] = normal_lpdf(growth[n] | beta[1] + beta[2] * food[n] + beta[3] * p_ncatch[n], sigma);
  }
}", "addicott_m3a_s2_v01.stan")
  
  ni <- 4000
  nt <- 2
  nb <- ni / 2
  nc <- 4
  
  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  set.seed(123)
  
  m3a_s2_out <- stan(
    file = "addicott_m3a_s2_v01.stan",
    data = list(
      N = nrow(df_m3a_s2), 
      growth = df_m3a_s2$growth,
      food = df_m3a_s2$food,
      p_ncatch = df_m3a_s2$p_ncatch),
    init = lapply(1:nc, function(i)
      list(beta = rnorm(3),
           sigma = rexp(1, 1))),
    pars = c("beta", "sigma", "log_lik"),
    chains = nc, 
    iter = ni, 
    warmup = nb, 
    thin = nt)
  
  m3a_log_lik <- extract_log_lik(m3a_s2_out)
  m3a_loo <- loo(m3a_log_lik)
  
  #### Model 3b ####
  # Here, we will propagate the uncertainty associated with predicted catch
  # in one streamlined model
  
  setwd(here::here("stan_models/"))
  write(
    "data {
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
  sigma1 ~ exponential(20);
  //sigma1 ~ exponential(1);
  
  beta ~ normal(0, 2);
  sigma2 ~ gamma(5, 20);
  //sigma2 ~ exponential(1);
  
  
  ncatch ~ normal(alpha[1] + alpha[2] * nets + alpha[3] * food, sigma1);
  
  p_ncatch ~ normal(alpha[1] + alpha[2] * nets + alpha[3] * food, sigma1);
  
  growth ~ normal(beta[1] + beta[2] * food + beta[3] * p_ncatch, sigma2);
}

generated quantities {
  vector[N] log_lik;
  for(n in 1:N){
    log_lik[n] = normal_lpdf(growth[n] | beta[1] + beta[2] * food[n] + beta[3] * p_ncatch[n], sigma2);
  }
}", "addicott_m3b_v01.stan")
  
  # bumping up the iterations/thinning because effective sample size was low
  ni <- 6000
  nt <- 3
  nb <- ni / 2
  nc <- 4
  
  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  set.seed(123)
  
  m3b_out <- stan(
    file = "addicott_m3b_v01.stan",
    data = list(
      N = sample_sizes[i], 
      ncatch = pull(slice(df, 1:sample_sizes[i]), ncatch), 
      nets = pull(slice(df, 1:sample_sizes[i]), nets), 
      food = pull(slice(df, 1:sample_sizes[i]), food), 
      growth = pull(slice(df, 1:sample_sizes[i]), growth)),
    init = lapply(1:nc, function(i)
      list(
        alpha = rnorm(3),
        beta = rnorm(3),
        sigma1 = rexp(1, 1),
        sigma2 = rexp(1, 1))),
    pars = c("alpha", "beta",
             "sigma1", "sigma2", "log_lik"),
    chains = nc, 
    iter = ni, 
    warmup = nb, 
    thin = nt,
    control = list(adapt_delta = 0.99)) # to avoid divergent transitions
  
  m3b_log_lik <- extract_log_lik(m3b_out)
  m3b_loo <- loo(m3b_log_lik)
  
  # compare models with expected log pointwise predictive density (ELPD)
  loo_table <- loo_compare(m1_loo, m2_loo, m3a_loo, m3b_loo) %>% 
    as_tibble(rownames = "model") %>% 
    add_column(description = c("m1", "m2", "m3: uncertainty", "m3: 2-stage")) %>% 
    dplyr::select(model = description, elpd_diff:se_looic) %>% 
    add_column(n = sample_sizes[i])
  
  m3b_beta <- summary(
    m3b_out, 
    pars = c("beta"),
    probs = c(0.025, 0.975))$summary %>% 
    as_tibble(rownames = "parameter") %>% 
    janitor::clean_names() %>% 
    filter(parameter == "beta[2]" | parameter == "beta[3]") %>% 
    dplyr::select(mean, sd, lower95 = x2_5_percent, upper95 = x97_5_percent) %>% 
    add_column(model = "m3b",
               predictor = c("food", "catch"))
  
  m3a_beta <- summary(
    m3a_s2_out, 
    pars = c("beta"),
    probs = c(0.025, 0.975))$summary %>% 
    as_tibble(rownames = "parameter") %>% 
    janitor::clean_names() %>% 
    filter(parameter == "beta[2]" | parameter == "beta[3]") %>% 
    dplyr::select(mean, sd, lower95 = x2_5_percent, upper95 = x97_5_percent) %>% 
    add_column(model = "m3a",
               predictor = c("food", "catch"))
  
  m2_beta <- summary(
    m2_out, 
    pars = c("beta"),
    probs = c(0.025, 0.975))$summary %>% 
    as_tibble(rownames = "parameter") %>% 
    janitor::clean_names() %>% 
    filter(parameter == "beta[2]") %>% 
    dplyr::select(mean, sd, lower95 = x2_5_percent, upper95 = x97_5_percent) %>% 
    add_column(model = "m2",
               predictor = c("food"))
  
  m1_beta <- summary(
    m1_out, 
    pars = c("beta"),
    probs = c(0.025, 0.975))$summary %>% 
    as_tibble(rownames = "parameter") %>% 
    janitor::clean_names() %>% 
    filter(parameter == "beta[2]" | parameter == "beta[3]") %>% 
    dplyr::select(mean, sd, lower95 = x2_5_percent, upper95 = x97_5_percent) %>% 
    add_column(model = "m1",
               predictor = c("food", "catch"))

  food_beta <- full_join(m1_beta, m2_beta) %>% 
    full_join(m3a_beta) %>% 
    full_join(m3b_beta) %>% 
    add_column(n = sample_sizes[i])
  
  loo_tables[[i]] <- loo_table
  beta_estimates[[i]] <- food_beta
}

str(loo_tables)
str(beta_estimates)

do.call(rbind, loo_tables) %>% 
  dplyr::select(n, model, elpd_diff, se_diff)

pal <- MetPalettes$Demuth[[1]][c(2, 4, 9)]

# visualize estimated effect of food availability from the models
do.call(rbind, beta_estimates) %>% 
  filter(predictor == "food") %>% 
  mutate(model_name = ifelse(model == "m1", "Naive (biased)",
                             ifelse(model == "m2", "Simple (unbiased)",
                                    ifelse(model == "m3a",
                                           "Two-stage (no uncertainty)",
                                           "Two-stage (uncertainty)")))) %>% 
  ggplot(aes(x = mean, y = model_name, color = factor(n,
                                                      levels = c(1000, 100, 25)))) + 
  geom_vline(xintercept = 0.5, color = "gray50", linetype = "dashed") +
  geom_errorbar(aes(xmin = lower95, xmax = upper95), width = 0,
                position = position_dodge(width = 0.4),
                size = 1.25) +
  geom_point(position =  position_dodge(width = 0.4),
             size = 2) +
  scale_color_manual("Sample size",
                     values = pal, 
                     guide = guide_legend(reverse = TRUE)) + 
  theme_classic() +
  xlab("Estimated effect of food availability") +
  theme(axis.title.y = element_blank(),
        axis.title.x = element_text(color = "black", size = 8),
        axis.text = element_text(color = "black", size = 7),
        legend.text = element_text(color = "black", size = 7), 
        legend.title = element_text(color = "black", size = 7),
        axis.line = element_line(color = "black", size = 0.2),
        axis.ticks = element_line(color = "black", size = 0.2))


setwd(here::here("results"))
betas <- do.call(rbind, beta_estimates)
write_csv(betas, "beta_estimates_original_sigmas_v01.csv")
loo_tabs <- do.call(rbind, loo_tables)
write_csv(loo_tabs, "loo_tables_v01.csv")

setwd(here::here("figures"))
ggsave("gilbert_figure1.jpg", 
       width = 5, 
       height = 3, 
       units = "in", 
       dpi = 300)

# compute relative magnitude of posterior SDs
betas %>% 
  filter(predictor == "food") %>%
  filter(grepl("m3", model)) %>% 
  dplyr::select(model, n, sd) %>% 
  pivot_wider(names_from = "model", values_from = sd) %>% 
  mutate(percent_sd = m3b / m3a)
