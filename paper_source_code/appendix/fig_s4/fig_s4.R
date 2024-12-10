#########################################
## Figure 5 - New Hampshire Validation ##
#########################################
rm(list = ls())
library("redist"); library("tidyverse"); library("gridExtra"); library("maptools"); library("doRNG")
library("coda"); library("Rcpp"); library("spdep"); library("igraph")
library("doParallel")

sourceCpp("count_seats.cpp", rebuild = TRUE)

map <- readShapePoly("nh/nh.shp")

## Prep parallel setting
cl <- makeCluster(4, "PSOCK")
registerDoParallel(cl)
ntasks <- 10

#######################################
## Run simulations - hard constraint ##
#######################################
registerDoRNG(1)
sims.hard <- foreach(i = 1:ntasks, .verbose = TRUE, .export = "map", .packages = c("redist", "coda", "igraph", "spdep")) %dopar% {

    ## Run simulations
    out <- redist.mcmc(map, map@data$POP100, 50000, ndists = 2, popcons = .01)

    ## Get dissimilarity
    dissim <- redist.segcalc(out, map@data$PRES_REP08, map@data$POP100)
    vi <- NA

    ## Return
    return(list(dissim = dissim, vi = vi))

}

###########################################
## Run simulations - simulated tempering ##
###########################################
registerDoRNG(1)
sims.st <- foreach(i = 1:ntasks, .verbose = TRUE, .export = "map", .packages = c("redist", "coda", "igraph", "spdep")) %dopar% {

    ## Run simulations
    weights <- rep(NA, 10); for(i in 1:length(weights)){weights[i] <- 4^i}
    out <- redist.mcmc(map, map@data$POP100, 50000, ndists = 2, constraint = "population",
                       temper = "simulated", beta = -27, betaweights = weights)

    ## Subset down
    inds <- which(out$beta_sequence == -27 & out$distance_parity <= 0.01)

    ## Get dissimilarity
    dissim <- redist.segcalc(out, map@data$PRES_REP08, map@data$POP100)[inds]
    vi <- NA

    ## Return
    return(list(dissim = dissim, vi = vi))
    
}

###########################################
## Load simulations - parallel tempering ##
###########################################
sims.pt <- vector(mode = "list", length = 10)
for(i in 1:10){

    load(paste("fig_s4_mpi/nh_chain", i, "_temp-27.RData", sep = ""))

    ## Subset down
    inds <- which(algout$distance_parity <= 0.01)

    ## Get dissimilarity
    dissim <- redist.segcalc(algout, map@data$PRES_REP08, map@data$POP100)[inds]
    vi <- NA

    ## Add to list
    sims.pt[[i]] <- list(dissim = dissim, vi = vi)

}

##################
## Create plots ##
##################

## Get data
sims.hard.ds <- vector(mode = "list", length = length(sims.hard))
sims.pt.ds <- vector(mode = "list", length = length(sims.hard))
for(i in 1:length(sims.hard)){
    sims.hard.ds[[i]] <- sims.hard[[i]]$dissim
    sims.pt.ds[[i]] <- sims.pt[[i]]$dissim
}

## Rename
sims.hard <- sims.hard.ds
sims.pt <- sims.pt.ds

logit.mcmc <- function(x, len = min.len){mcmc(log(x[1:len] / (1 - x[1:len])))}

min.len <- min(c(unlist(lapply(sims.pt, length))))

## Transform hard sims
list.hard <- mcmc.list(lapply(sims.hard, logit.mcmc))
## Transform pt sims
list.pt <- mcmc.list(lapply(sims.pt, logit.mcmc))

pdf(file = "fig_s4.pdf", height = 8, width = 12)
par(mfrow = c(2,3), mai=c(.55,.8,0.45,0.25))
## Hard constraint
autocorr.plot(list.hard[[1]], lag.max = 50, 
              auto.layout = FALSE,
              cex.lab = 1.5,
              cex.axis = 1.4)
mtext("Algorithm 1.1", 2, line = 4.5, cex = 1.4)
mtext("Autocorrelation of a Chain", 3, line = 1.7, cex = 1.4)
traceplot(list.hard[[1]],
          ylab = "Republican Dissimilarity \n (logit transformed)",
          ylim = c(-13,-3),
          xlim = c(0, 10001),
          cex.lab = 1.5,
          cex.axis = 1.4)
mtext("Trace of a Chain", 3, line = 1.7, cex = 1.4)
gelman.plot(list.hard, auto.layout = FALSE, ylim = c(1,2.5),
            cex.lab = 1.5,
            cex.axis = 1.4,
            xlim = c(0, 10001),
            transform = FALSE,
            autoburnin = TRUE)
mtext("Gelman-Rubin Diagnostic", 3, line = 1.7, cex = 1.4)

## Parallel tempering constriant
autocorr.plot(list.pt[[1]], lag.max = 50,
              auto.layout = FALSE,
              cex.lab = 1.5,
              cex.axis = 1.4)
mtext("Algorithm 2", 2, line = 4.5, cex = 1.4)
traceplot(list.pt[[1]],
          ylab = "Republican Dissimilarity \n (logit transformed)",
          ylim = c(-13,-3),
          xlim = c(0, 10001),
          cex.lab = 1.5,
          cex.axis = 1.4)
gelman.plot(list.pt, auto.layout = FALSE, ylim = c(1,2.5),
            xlim = c(0, 10001),
            cex.lab = 1.5,
            cex.axis = 1.4,
            transform = FALSE, autoburnin = TRUE)
dev.off()

