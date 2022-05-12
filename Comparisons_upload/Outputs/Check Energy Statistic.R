library(energy)
Calc_EnergyStat <- function(n_init, n_seq, TestFunction){
  MCMC_sim <- read.csv(paste0(TestFunction, "_MCMCsim.csv"), header=TRUE)
  MCMC_LHD <- read.csv(paste0(TestFunction, "_MCMCLHD_total", n_init+n_seq, ".csv"), header=TRUE)
  MCMC_RJ <- read.csv(paste0(TestFunction, "_MCMCRanjan2016_init", n_init, "_seq", n_seq, ".csv"), header=TRUE)
  MCMC_ApproxPosterior <- read.csv(paste0(TestFunction, "_MCMCApproxpost_init", n_init, "_seq", n_seq, ".csv"), header=TRUE)
  n_sim <- nrow(MCMC_sim)
  n_RJ <- nrow(MCMC_RJ)
  n_LHD <- nrow(MCMC_LHD)
  n_ApproxPosterior <- nrow(MCMC_ApproxPosterior)
  
  # RJ vs MCMC
  comb_df1 <- rbind(MCMC_RJ, MCMC_sim)
  d1 <- dist(comb_df1)
  set.seed(1234)
  e.stat.RJ <- eqdist.etest(d1, sizes=c(n_RJ, n_sim), distance=TRUE, R = 199)$statistic
  
  # LHD vs MCMC
  comb_df2 <- rbind(MCMC_LHD, MCMC_sim)
  d2 <- dist(comb_df2)
  set.seed(1234)
  e.stat.LHD <- eqdist.etest(d2, sizes=c(n_LHD, n_sim), distance=TRUE, R = 199)$statistic
  
  # ApproxPosterior vs MCMC
  comb_df3 <- rbind(MCMC_ApproxPosterior, MCMC_sim)
  d3 <- dist(comb_df3)
  set.seed(1234)
  e.stat.ApproxPosterior <- eqdist.etest(d3, sizes=c(n_ApproxPosterior, n_sim), distance=TRUE, R = 199)$statistic
  
  return(c(e.stat.RJ, e.stat.LHD, e.stat.ApproxPosterior))
}

### Calculate Energy Statistic:
n_init_grid <- c(10,10,10) 
n_seq_grid <- c(25,25,25)
TestFunction_grid <- c("TF1","TF2","TF3")
res <- rep(NA,6)
for(i in 1:length(n_init_grid)){
  n_init <- n_init_grid[i]
  n_seq <- n_seq_grid[i]
  TestFunction <- TestFunction_grid[i]
  temp <- Calc_EnergyStat(n_init, n_seq, TestFunction)
  res <- rbind(res, c(TestFunction, n_init, n_seq, temp))
}
(res <- res[-1,])
