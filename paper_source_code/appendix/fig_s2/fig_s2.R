## ----------------------------------------
## Results don't get better with long chain
## ----------------------------------------
rm(list = ls())
library("redist")
library("igraph")
library("parallel")

set.seed(194115) ## random.org
nsims <- 10000

data(algdat.p10)

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

## Run mcmc algorithm - short
mcmc.out <- redist.mcmc(algdat.p10$adjlist,
                        algdat.p10$precinct.data$pop,
                        10000,
                        popcons = .1,
                        lambda = 2,
                        ndists = 3)

## Coerce objects
rep.seg.true <- data.frame(algdat.p10$segregation.index$repdiss)
names(rep.seg.true) <- "dissim"
rep.seg.mcmc.short <- data.frame(
    redist.segcalc(algout = mcmc.out,
                   grouppop = algdat.p10$precinct.data$repvote,
                   fullpop = algdat.p10$precinct.data$pop)
)
names(rep.seg.mcmc.short) <- "dissim"

## Run mcmc algorithm - med
mcmc.out <- redist.mcmc(algdat.p10$adjlist,
                        algdat.p10$precinct.data$pop,
                        100000,
                        popcons = .1,
                        lambda = 2,
                        ndists = 3)

## Coerce objects
rep.seg.true <- data.frame(algdat.p10$segregation.index$repdiss)
names(rep.seg.true) <- "dissim"
rep.seg.mcmc.med <- data.frame(
    redist.segcalc(algout = mcmc.out,
                   grouppop = algdat.p10$precinct.data$repvote,
                   fullpop = algdat.p10$precinct.data$pop)
)
names(rep.seg.mcmc.med) <- "dissim"

## Run mcmc algorithm - long
mcmc.out <- redist.mcmc(algdat.p10$adjlist,
                        algdat.p10$precinct.data$pop,
                        1000000,
                        popcons = .1,
                        lambda = 2,
                        ndists = 3)

## Coerce objects
rep.seg.true <- data.frame(algdat.p10$segregation.index$repdiss)
names(rep.seg.true) <- "dissim"
rep.seg.mcmc.long <- data.frame(
    redist.segcalc(algout = mcmc.out,
                   grouppop = algdat.p10$precinct.data$repvote,
                   fullpop = algdat.p10$precinct.data$pop)
)
names(rep.seg.mcmc.long) <- "dissim"

## Store sims
pop10 <- vector(mode = "list")
pop10$true <- rep.seg.true
pop10$short <- rep.seg.mcmc.short
pop10$med <- rep.seg.mcmc.med
pop10$long <- rep.seg.mcmc.long

## 10% population constraint, hard
xlim <- c(0, .3)
ylim <- c(0, 55)

pdf(file = "fig_s2.pdf", height = 5, width = 7)
plot(density(pop10$true$dissim),
     xlab = "Republican Dissimilarity Index",
     main = "",
     xlim = xlim,
     ylim = ylim,
     cex.lab = 1.3,
     cex.axis = 1.4)
polygon(density(pop10$true$dissim), col = "grey", border = "grey")
lines(density(pop10$short$dissim, from = 0, to = 1), lwd = 2)
lines(density(pop10$med$dissim, from = 0, to = 1), col = "red",
      lty = 5, lwd = 2)
lines(density(pop10$long$dissim, from = 0, to = 1), lty = 6, col = "blue",
      lwd = 2)
legend("topright", c("True Distribution",
                     "10,000 Samples",
                     "100,000 Samples",
                     "1,000,000 Samples"),
       lty = c(1, 1, 5, 6), lwd = c(4, 2, 2, 2),
       col = c("grey", "black", "red", "blue"),
       cex = 1.1, bty = "n", ncol = 2)
dev.off()


