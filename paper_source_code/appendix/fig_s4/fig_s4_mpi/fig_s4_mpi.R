##############################
## Figure 5 MPI simulations ##
##############################

rm(list = ls())
library("redist"); library("maptools")

## See if in slurm environment
inSLURM <- (Sys.getenv("SLURM_JOB_ID") != "")

if(inSLURM){
    aid <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
}

map <- readShapePoly("../nh/nh.shp")

set.seed(1)

redist.mcmc.mpi(map, map@data$POP100, nsims = 50000,
                ndists = 2, beta = -27, verbose = TRUE,
                savename = paste("nh_chain", aid, sep = ""))

