## Load functions, packages
library("grDevices")
library("redist")
library("MASS")
library("Cairo")

source("functions.R")

set.seed(434587) ## random.org 12792
ndists <- 2
nsims <- 1000000

## Run sims
lat.32 <- testLat(3, 2, nsims, ndists)
lat.33 <- testLat(3, 3, nsims, ndists)
lat.43 <- testLat(4, 3, nsims, ndists)

rng <- range(c(lat.32$true$dissim, lat.33$true$dissim, lat.43$true$dissim))
xlab <- "Republican Dissimilarity Index - Truth"
ylab <- "Republican Dissimilarity Index - MCMC"

## Plot
cairo_pdf(file = "fig_s7.pdf", height = 8, width = 12)
par(mfrow = c(2, 3), mai = c(.55,.9,0.4,0.05))

qqplot(lat.32$true$dissim, lat.32$sim$exact,
       xlim = rng, ylim = rng, pch = 16,
       xlab = xlab, ylab = ylab, cex.lab = 1.4, cex.axis = 1.3)
abline(0, 1)
mtext(expression(paste(italic(B)^"\u2020", " Approximation",sep="")), 2, line = 4.5, cex = 1.4)
mtext("3x2 Lattice", 3, line = .65, cex = 1.4)

qqplot(lat.33$true$dissim, lat.33$sim$exact,
       xlim = rng, ylim = rng, pch = 16,
       xlab = xlab, ylab = ylab, cex.lab = 1.4, cex.axis = 1.3)
abline(0, 1)
mtext("3x3 Lattice", 3, line = .65, cex = 1.4)

qqplot(lat.43$true$dissim, lat.43$sim$exact,
       xlim = rng, ylim = rng, pch = 16,
       xlab = xlab, ylab = ylab, cex.lab = 1.4, cex.axis = 1.3)
abline(0, 1)
mtext("4x3 Lattice", 3, line = .65, cex = 1.4)

qqplot(lat.32$true$dissim, lat.32$sim$appx,
       xlim = rng, ylim = rng, pch = 16,
       xlab = xlab, ylab = ylab, cex.lab = 1.4, cex.axis = 1.3)
abline(0, 1)
mtext(expression(paste(italic(B), " Approximation", sep = "")), 2, line = 4.5, cex = 1.4)

qqplot(lat.33$true$dissim, lat.33$sim$appx,
       xlim = rng, ylim = rng, pch = 16,
       xlab = xlab, ylab = ylab, cex.lab = 1.4, cex.axis = 1.3)
abline(0, 1)

qqplot(lat.43$true$dissim, lat.43$sim$appx,
       xlim = rng, ylim = rng, pch = 16,
       xlab = xlab, ylab = ylab, cex.lab = 1.4, cex.axis = 1.3)
abline(0, 1)

dev.off()

