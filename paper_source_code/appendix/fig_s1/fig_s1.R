###############################################
## Figure 3 - new metropolis hastings ratios ##
###############################################
rm(list = ls())
library("redist")
library("igraph")
library("parallel")

set.seed(194115) ## random.org
nsims <- 10000

ipw <- function(x, beta, pop){
    indpop <- which(x$distance_parity <= pop)
    indbeta <- which(x$beta_sequence == beta)
    inds <- intersect(indpop, indbeta)
    psi <- x$constraint_pop[inds]
    w <- 1 / exp(beta * psi)
    inds <- sample(inds, length(inds), replace = TRUE, prob = w)
    x <- x$partitions[,inds]
    return(x)
}

data(algdat.p20)

## Run simulated tempering algorithm - 20%
## 1 weight for each temperature in ladder, always strictly going up to target temp
## 4 is low, 10485876 is target temperature
## Beta has to be negative, not necessarily -5.4
betaweights <- rep(NA, 10); for(i in 1:10){betaweights[i] <- 2^i}
mcmc.out.st <- redist.mcmc(adjobj = algdat.p20$adjlist,
                           popvec = algdat.p20$precinct.data$pop,
                           ndists = 3,
                           nsims = nsims,
                           beta = -5.4,
                           lambda = 2,
                           constraint = "population",
                           temper = "simulated",
                           betaweights = betaweights)
mcmc.out.st <- ipw(mcmc.out.st, beta = -5.4, pop = .2)

rep.seg.st.20 <- data.frame(redist.segcalc(algout = mcmc.out.st,
                                        grouppop = algdat.p20$precinct.data$repvote,
                                        fullpop = algdat.p20$precinct.data$pop))
names(rep.seg.st.20) <- "dissim"

## Run simulated tempering algorithm - 10%
data(algdat.p10)
betaweights <- rep(NA, 10); for(i in 1:10){betaweights[i] <- 2^i}
mcmc.out.st <- redist.mcmc(adjobj = algdat.p10$adjlist,
                           popvec = algdat.p10$precinct.data$pop,
                           ndists = 3,
                           nsims = nsims,
                           beta = -9,
                           lambda = 2,
                           constraint = "population",
                           temper = "simulated",
                           betaweights = betaweights)
mcmc.out.st <- ipw(mcmc.out.st, beta = -9, pop = .1)

rep.seg.st.10 <- data.frame(redist.segcalc(algout = mcmc.out.st,
                                        grouppop = algdat.p10$precinct.data$repvote,
                                        fullpop = algdat.p10$precinct.data$pop))
names(rep.seg.st.10) <- "dissim"

## ---------------------------
## Simulated tempering results
## ---------------------------
## Empty frame

xlim <- c(0, .3)
ylim <- c(0, 55)

pdf(file = "fig_s1.pdf", height = 4, width = 8)
par(mfrow = c(1,2), mai=c(.8,.92,0.3,0.1))

## 20% population constraint, tempered
plot(density(algdat.p20$segregation.index$repdiss),
     xlab = "Republican Dissimilarity Index",
     main = "",
     xlim = xlim,
     ylim = ylim,
     cex.lab = 1.1,
     cex.axis = 1.1)
polygon(density(algdat.p20$segregation.index$repdiss), col = "grey", border = "grey")
lines(density(rep.seg.st.20$dissim, from = 0, to = 1), lty = 4, col = "black",
      lwd = 2)
mtext("Constrained Simulations (20%)", 3, line = .6, cex = 1.2)
legend("topright", c("True Distribution", 
                     "Algorithm S1"),
       lty = c(1, 4), lwd = c(4, 2),
       col = c("grey", "black"), cex = 1.1, bty = "n")

## 10% population constraint, tempered
plot(density(algdat.p10$segregation.index$repdiss),
     xlab = "Republican Dissimilarity Index",
     main = "",
     xlim = xlim,
     ylim = ylim,
     cex.lab = 1.1,
     cex.axis = 1.1)
polygon(density(algdat.p10$segregation.index$repdiss), col = "grey", border = "grey")
lines(density(rep.seg.st.10$dissim, from = 0, to = 1), lty = 4, col = "black",
      lwd = 2)
mtext("Constrained Simulations (10%)", 3, line = .6, cex = 1.2)
dev.off()

