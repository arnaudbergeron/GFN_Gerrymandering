##############################
## Figure 3 MPI simulations ##
##############################

rm(list = ls())
library("redist"); library("ggplot2"); library("gridExtra")

set.seed(6)
nsims <- 10000

###############################
## 10% population constraint ##
###############################
data(algdat.p10)

redist.mcmc.mpi(algdat.p10$adjlist, algdat.p10$precinct.data$pop,
                nsims = nsims, ndists = 3, beta = -9, verbose = TRUE,
                savename = "mpi10")

