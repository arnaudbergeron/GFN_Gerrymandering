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

## Load bard data ##
load("fig3_bard/fig3_bard.RData")
data(algdat.pfull)

## Calculate segregation index for BARD
bard.full <- data.frame(dissim = redist.segcalc(
                            algout = bard.full,
                            grouppop = algdat.pfull$precinct.data$repvote,
                            fullpop = algdat.pfull$precinct.data$pop
                        ))
bard.20 <- data.frame(dissim = redist.segcalc(
                          algout = bard.20,
                          grouppop = algdat.pfull$precinct.data$repvote,
                          fullpop = algdat.pfull$precinct.data$pop
                      ))
bard.10 <- data.frame(dissim = redist.segcalc(
                          algout = bard.10,
                          grouppop = algdat.pfull$precinct.data$repvote,
                          fullpop = algdat.pfull$precinct.data$pop
                      ))

##############################
## No population constraint ##
##############################
xlim <- range(algdat.pfull$segregation.index$repdiss)

## Run mcmc algorithm
## For 25 preceint, 3 dist test set. Full redistricting, not local
mcmc.out <- redist.mcmc(algdat.pfull$adjlist,
                        algdat.pfull$precinct.data$pop,
                        nsims,
                        ndists = 3,
                        lambda = 2)

## 
## Coerce objects
rep.seg.true <- data.frame(algdat.pfull$segregation.index$repdiss)
names(rep.seg.true) <- "dissim"
rep.seg.mcmc <- data.frame(redist.segcalc(algout = mcmc.out,
                                          grouppop = algdat.pfull$precinct.data$repvote,
                                          fullpop = algdat.pfull$precinct.data$pop))
names(rep.seg.mcmc) <- "dissim"

## Store
nopop <- vector(mode = "list")
nopop$true <- rep.seg.true
nopop$sim.hard <- rep.seg.mcmc
nopop$sim.cr <- bard.full

###############################
## 20% population constraint ##
###############################
data(algdat.p20)

## Run mcmc algorithm
mcmc.out <- redist.mcmc(algdat.p20$adjlist,
                        algdat.p20$precinct.data$pop,
                        nsims,
                        popcons = .2,
                        ndists = 3,
                        lambda = 2)

## Coerce objects
rep.seg.true <- data.frame(algdat.p20$segregation.index$repdiss)
names(rep.seg.true) <- "dissim"
rep.seg.mcmc <- data.frame(redist.segcalc(algout = mcmc.out,
                                          grouppop = algdat.p20$precinct.data$repvote,
                                          fullpop = algdat.p20$precinct.data$pop))
names(rep.seg.mcmc) <- "dissim"

## Run simulated tempering algorithm
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

## Run soft constraint, no tempering
mcmc.out.soft <- redist.mcmc(adjobj = algdat.p20$adjlist,
                             popvec = algdat.p20$precinct.data$pop,
                             ndists = 3,
                             lambda = 2,
                             nsims = nsims,
                             beta = -5.4,
                             constraint = "population")
mcmc.out.soft <- ipw(mcmc.out.soft, beta = -5.4, pop = .2)

## Load parallel tempering results
load("fig3_mpi/mpi20_temp-5.4.RData"); mcmc.out.pt <- algout
mcmc.out.pt <- ipw(mcmc.out.pt, beta = -5.4, pop = .2)

## Coerce objects - st, pt
rep.seg.st <- data.frame(redist.segcalc(algout = mcmc.out.st,
                                        grouppop = algdat.p20$precinct.data$repvote,
                                        fullpop = algdat.p20$precinct.data$pop))
names(rep.seg.st) <- "dissim"
rep.seg.pt <- data.frame(redist.segcalc(algout = mcmc.out.pt,
                                        grouppop = algdat.p20$precinct.data$repvote,
                                        fullpop = algdat.p20$precinct.data$pop))
names(rep.seg.pt) <- "dissim"
rep.seg.soft <- data.frame(redist.segcalc(algout = mcmc.out.soft,
                                        grouppop = algdat.p20$precinct.data$repvote,
                                        fullpop = algdat.p20$precinct.data$pop))
names(rep.seg.soft) <- "dissim"

## Store sims
pop20 <- vector(mode = "list")
pop20$true <- rep.seg.true
pop20$sim.hard <- rep.seg.mcmc
pop20$sim.st <- rep.seg.st
pop20$sim.pt <- rep.seg.pt
pop20$sim.soft <- rep.seg.soft
pop20$sim.cr <- bard.20

###############################
## 10% population constraint ##
###############################
data(algdat.p10)

## Run mcmc algorithm
mcmc.out <- redist.mcmc(algdat.p10$adjlist,
                        algdat.p10$precinct.data$pop,
                        nsims,
                        popcons = .1,
                        lambda = 2,
                        ndists = 3)

## Coerce objects
rep.seg.true <- data.frame(algdat.p10$segregation.index$repdiss)
names(rep.seg.true) <- "dissim"
rep.seg.mcmc <- data.frame(redist.segcalc(algout = mcmc.out,
                                          grouppop = algdat.p10$precinct.data$repvote,
                                          fullpop = algdat.p10$precinct.data$pop))
names(rep.seg.mcmc) <- "dissim"

## Run simulated tempering algorithm
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

## Run soft constraint, no tempering
mcmc.out.soft <- redist.mcmc(adjobj = algdat.p10$adjlist,
                             popvec = algdat.p10$precinct.data$pop,
                             ndists = 3,
                             nsims = nsims,
                             lambda = 2,
                             beta = -9,
                             constraint = "population")
mcmc.out.soft <- ipw(mcmc.out.soft, beta = -9, pop = .1)

## Load parallel tempering results
load("fig3_mpi/mpi10_temp-9.RData"); mcmc.out.pt <- algout
mcmc.out.pt <- ipw(mcmc.out.pt, beta = -9, pop = .1)

## Coerce object
rep.seg.st <- data.frame(redist.segcalc(algout = mcmc.out.st,
                                        grouppop = algdat.p10$precinct.data$repvote,
                                        fullpop = algdat.p10$precinct.data$pop))
names(rep.seg.st) <- "dissim"
rep.seg.pt <- data.frame(redist.segcalc(algout = mcmc.out.pt,
                                        grouppop = algdat.p10$precinct.data$repvote,
                                        fullpop = algdat.p10$precinct.data$pop))
names(rep.seg.pt) <- "dissim"
rep.seg.soft <- data.frame(redist.segcalc(algout = mcmc.out.soft,
                                        grouppop = algdat.p10$precinct.data$repvote,
                                        fullpop = algdat.p10$precinct.data$pop))
names(rep.seg.soft) <- "dissim"

## Store sims
pop10 <- vector(mode = "list")
pop10$true <- rep.seg.true
pop10$sim.hard <- rep.seg.mcmc
pop10$sim.st <- rep.seg.st
pop10$sim.pt <- rep.seg.pt
pop10$sim.soft <- rep.seg.soft
pop10$sim.cr <- bard.10

######################
## Create full plot ##
######################
xlim <- c(0, max(c(nopop$true$dissim, pop20$true$dissim, pop10$true$dissim)) + .05)
ylim <- c(0, max(density(pop10$sim.cr$dissim)$y))

## -------------------
## Dissimilarity index
## -------------------
pdf(file = "fig3.pdf", height = 6.25, width = 13)
par(mfrow = c(1,3), mai=c(.55,.65,0.3,0.1))
## No constraint
plot(density(nopop$true$dissim),
     xlab = "Republican Dissimilarity Index",
     main = "",
     xlim = xlim,
     ylim = ylim,
     cex.lab = 1.6,
     cex.axis = 1.7)
polygon(density(nopop$true$dissim), col = "grey", border = "grey")
lines(density(nopop$sim.hard$dissim, from = 0, to = 1), lwd = 2)
lines(density(nopop$sim.cr$dissim, from = 0, to = 1), col = "red",
      lty = 5, lwd = 2)
mtext("No Equal Population Constraint", 3, line = .75, cex = 1.3)
legend(x = .15, y = 50,
       c("True Distribution",
         "Algorithm 1",
         "Algorithm 3"),
       lty = c(1, 1, 5), lwd = c(4, 2, 2),
       col = c("grey", "black", "red"),
       cex = 1.2, bty = "n", ncol = 1)

## 20% population constraint, hard
plot(density(pop20$true$dissim),
     xlab = "Republican Dissimilarity Index",
     main = "",
     xlim = xlim,
     ylim = ylim,
     cex.lab = 1.6,
     cex.axis = 1.7)
polygon(density(pop20$true$dissim), col = "grey", border = "grey")
lines(density(pop20$sim.hard$dissim, from = 0, to = 1), lwd = 2)
lines(density(pop20$sim.cr$dissim, from = 0, to = 1), col = "red",
      lty = 5, lwd = 2)
lines(density(pop20$sim.pt$dissim, from = 0, to = 1), lty = 6, col = "blue",
      lwd = 2)
lines(density(pop20$sim.soft$dissim, from = 0, to = 1), lty = 4, col = "purple",
      lwd = 2)
mtext("20% Equal Population Constraint", 3, line = .75, cex = 1.3)
legend(x = .015, y = 50,
       c("True Distribution",
                     "Algorithm 3",
                     "",
                     "Algorithm 1.1",
                     "Algorithm 1.2",
                     "Algorithm 2"),
       lty = c(1, 5, 1, 1, 4, 6), lwd = c(4, 2, 2, 2, 2, 2),
       col = c("grey", "red", "white", "black", "purple", "blue"),
       cex = 1.2, bty = "n", ncol = 2)
text(x = .23, y = 50.25, labels = "Our Algorithms", cex = 1.3)

## 10% population constraint, hard
plot(density(pop10$true$dissim),
     xlab = "Republican Dissimilarity Index",
     main = "",
     xlim = xlim,
     ylim = ylim,
     cex.lab = 1.6,
     cex.axis = 1.7)
polygon(density(pop10$true$dissim), col = "grey", border = "grey")
lines(density(pop10$sim.hard$dissim, from = 0, to = 1), lwd = 2)
lines(density(pop10$sim.cr$dissim, from = 0, to = 1), col = "red",
      lty = 5, lwd = 2)
lines(density(pop10$sim.pt$dissim, from = 0, to = 1), lty = 6, col = "blue",
      lwd = 2)
lines(density(pop10$sim.soft$dissim, from = 0, to = 1), lty = 4, col = "purple",
      lwd = 2)
mtext("10% Equal Population Constraint", 3, line = .75, cex = 1.3)
legend(x = .015, y = 50,
       c("True Distribution",
                     "Algorithm 3",
                     "",
                     "Algorithm 1.1",
                     "Algorithm 1.2",
                     "Algorithm 2"),
       lty = c(1, 5, 1, 1, 4, 6), lwd = c(4, 2, 2, 2, 2, 2),
       col = c("grey", "red", "white", "black", "purple", "blue"),
       cex = 1.2, bty = "n", ncol = 2)
text(x = .23, y = 50.25, labels = "Our Algorithms", cex = 1.3)
dev.off()

