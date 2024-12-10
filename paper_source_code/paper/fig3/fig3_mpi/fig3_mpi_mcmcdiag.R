##############################
## Figure 3 MPI simulations ##
##############################

rm(list = ls())
library("redist")

set.seed(6)
nsims <- 10000

i <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))

if(i %% 2 == 0){
    ## -------------------------
    ## 20% population constraint
    ## -------------------------
    data(algdat.p20)
    i <- i/2
    redist.mcmc.mpi(algdat.p20$adjlist, algdat.p20$precinct.data$pop,
                    nsims = nsims, ndists = 3, beta = -5.4, verbose = TRUE,
                    savename = paste0("mpi20_mcmcdiag_chain", i))
}else{
    ## -------------------------
    ## 10% population constraint
    ## -------------------------
    data(algdat.p10)
    i <- (i+1)/2
    redist.mcmc.mpi(algdat.p10$adjlist, algdat.p10$precinct.data$pop,
                    nsims = nsims, ndists = 3, beta = -9, verbose = TRUE,
                    savename = paste0("mpi10_mcmcdiag_chain", i))
}
