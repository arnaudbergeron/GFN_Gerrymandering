##############################
## Figure 3 MPI simulations ##
##############################

rm(list = ls())
library("redist"); library("ggplot2"); library("gridExtra")

set.seed(6)
nsims <- 10000

###############################
## 20% population constraint ##
###############################
data(algdat.p20)

redist.mcmc.mpi(algdat.p20$adjlist, algdat.p20$precinct.data$pop,
                nsims = nsims, ndists = 3, beta = -5.4, verbose = TRUE,
                savename = "mpi20")
