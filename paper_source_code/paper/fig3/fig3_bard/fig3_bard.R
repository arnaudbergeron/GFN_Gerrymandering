##########################
## Figure 3 - BARD sims ##
##########################
rm(list = ls())
library("redist"); library("BARD"); library("maptools")

set.seed(100)
nsims <- 10000
ndists <- 3

## Load data
load("testset253/algdat.RData")
import <- importBardShape(file.path("testset253/testset253"))
df <- import$df
df$POP <- geodat$pop
import$df <- df
nvtd <- ncol(import$centroids)

## No population constraint
bard <- matrix(NA, nvtd, nsims)
for(i in 1:nsims){
    out <- createContiguousPlan(import, ndists, usebb = FALSE, threshold = 100)

    cds <- rep(NA, nvtd)
    for(j in 1:ndists){   
        ind <- which(out == j)
        cds[ind] <- (j - 1)
    }

    bard[,i] <- cds

    if(i %% 100 == 0){
        cat("Done with iteration", i, "out of", nsims, "for no population constraint.\n")
    }
    
}

## bard.full <- data.frame(dissim = redist.segcalc(algout = bard,
##                             grouppop = geodat$mccain,
##                             fullpop = geodat$pop))
bard.full <- bard
cat("Full simulations done\n\n")

## 20% population constraint
bard <- matrix(NA, nvtd, nsims)
for(i in 1:nsims){
    out <- createContiguousPlan(import, ndists, usebb = FALSE, threshold = .2)

    cds <- rep(NA, nvtd)
    for(j in 1:ndists){   
        ind <- which(out == j)
        cds[ind] <- (j - 1)
    }

    bard[,i] <- cds

    if(i %% 100 == 0){
        cat("Done with iteration", i, "out of", nsims, "for 20% population constraint.\n")
    }
    
}

## bard.20 <- data.frame(dissim = redist.segcalc(algout = bard,
##                           grouppop = geodat$mccain,
##                           fullpop = geodat$pop))
bard.20 <- bard
cat("20% simulations done\n\n")

## 10% population constraint
bard <- matrix(NA, nvtd, nsims)
for(i in 1:nsims){
    out <- createContiguousPlan(import, ndists, usebb = FALSE, threshold = .1)

    cds <- rep(NA, nvtd)
    for(j in 1:ndists){   
        ind <- which(out == j)
        cds[ind] <- (j - 1)
    }

    bard[,i] <- cds

    if(i %% 100 == 0){
        cat("Done with iteration", i, "out of", nsims, "for 10% population constraint.\n")
    }
    
}

## bard.10 <- data.frame(dissim = redist.segcalc(algout = bard,
##                           grouppop = geodat$mccain,
##                           fullpop = geodat$pop))
bard.10 <- bard
cat("10% simulations done\n\n")

## Save data
save(bard.full, bard.20, bard.10, file = "fig3_bard.RData")

