## ----------------------------
## Code up figures for figlocal
## ----------------------------
library(redist)
library(maptools)
library(coda)
library(tidyverse)
library(Rcpp)
library(ggthemes)
library(RColorBrewer)
map <- readShapePoly("pa/pa_final.shp")

## -------------------
## MPI Simulations
## -------------------
## Load data
load("data/simulations_mpi_chain1.RData")
c1 <- algout
load("data/simulations_mpi_chain2.RData")
c2 <- algout
load("data/simulations_mpi_chain3.RData")
c3 <- algout

sourceCpp("cpp/count_seats.cpp", rebuild = TRUE)
sourceCpp("cpp/pBias.cpp", rebuild = TRUE)

## Check if chains converged - Rhat = 1.03!
map@data$USPTV2008 <- map@data$POP100
c1_seg <- redist.segcalc(c1, map@data$USPRV2008, map@data$POP100)
c2_seg <- redist.segcalc(c2, map@data$USPRV2008, map@data$POP100)
c3_seg <- redist.segcalc(c3, map@data$USPRV2008, map@data$POP100)

seglist <- mcmc.list(mcmc(c1_seg[!is.na(c1_seg)]), mcmc(c2_seg[!is.na(c2_seg)]),
                     mcmc(c3_seg[!is.na(c3_seg)]))

## --------------------------
## Get partisan bias of plans
## --------------------------
statebase <- sum(map@data$USCDV2008) /
    (sum(map@data$USCDV2008) + sum(map@data$USCRV2008))
equal <- .5 - statebase

range <- .1
inc <- seq(0, range, by = .01)

bias <- unique(c(rev(equal - inc), equal + inc))
algout <- cbind(c1$partitions[,!is.na(c1_seg)], c2$partitions[,!is.na(c2_seg)], 
                c3$partitions[,!is.na(c3_seg)])
repseats <- matrix(NA, nrow = ncol(algout), ncol = length(bias))
for(j in 1:length(bias)){
  repseats[,j] <- pBias(map@data$USCDV2008, map@data$USCRV2008, algout, bias[j])
  print(j)
}

## Make bias correspond to deviation from 50-50; convert to seat share
bias <- seq(-1 * range, range, length = length(bias))
dists <- length(unique(algout[,1]))
repseats <- repseats / dists
repseats <- 1 - repseats

## Plot step function
xmin <- -1 * range
xmax <- range

## Calculate bias - get change points
storebias <- rep(NA, nrow(repseats))
for(j in 1:nrow(repseats)){
  
  swing <- repseats[j,]
  
  ## Get the bias
  mod <- lm(swing ~ bias)
  null <- predict(mod, data.frame(bias = bias))
  
  ## Calculate the area
  gt0_area <- geiger:::.area.between.curves(bias[which(bias > 0)],
                                           null[which(bias > 0)],
                                           swing[which(bias > 0)],
                                           xrange = c(-1,1))
  lt0_area <- geiger:::.area.between.curves(bias[which(bias <= 0)],
                                            null[which(bias <= 0)],
                                            swing[which(bias <= 0)],
                                            xrange = c(-1,1))
  bias_area <- gt0_area + lt0_area
  storebias[j] <- bias_area
  if(j %% 1000 == 0){
    print(j)
  }
  
} 

## Get max and min bias to scale
bias_min <- - (range - -1 * range) * 1 / 2
bias_max <- (range - -1 * range) * 1 / 2

xax <- 1 - c(c1$distance_original[!is.na(c1_seg)],
             c2$distance_original[!is.na(c2_seg)],
             c3$distance_original[!is.na(c3_seg)]
             )

## Plot xax
test <- (1 - -1) * (tapply(storebias,  round(xax, 4), mean) - bias_min) / 
    (bias_max - bias_min) + -1
x <- as.numeric(names(test)) 
n <- table(round(xax, 4))

## -------------------
## Plot map with lowest partisan bias given distance from parity level
## -------------------
mindev <- .02875
maxdev <- .03125
ind <- which(storebias == min(storebias[xax < maxdev & xax > mindev]))[1]
orig_map <- algout[,1]+1
min_map <- algout[,ind]+1
map@data$orig_map <- orig_map
map@data$min_map <- min_map

map@data$unchanged <- orig_map == min_map
map@data %>% group_by(orig_map) %>%
    summarise(demvote = sum(USCDV2008)/sum(USCDV2008 + USCRV2008),
              sharediff = mean(unchanged))

map@data$dem_voteshare <- map@data$USCDV2008 /
    (map@data$USCDV2008 + map@data$USCRV2008)
map@data$col[!is.na(map@data$dem_voteshare)] <- rgb(
    red = (1 - map@data$dem_voteshare[!is.na(map@data$dem_voteshare)]),
    green = 0,
    blue = map@data$dem_voteshare[!is.na(map@data$dem_voteshare)],
    alpha = .5
)

mapsub_1 <- map[map$orig_map == 1 | map$min_map == 1,]
mapsub_1$col <- ifelse(mapsub_1$orig_map == mapsub_1$min_map, "white",
                 ifelse(mapsub_1$orig_map == 1, "blue", "red"))

dist <- 3
mapsub_dist <- map[map$orig_map == dist | map$min_map == dist,]
mapsub_dist$col <- ifelse(mapsub_dist$orig_map == mapsub_dist$min_map, "white",
                   ifelse(mapsub_dist$orig_map == dist, "blue", "red"))

## -------------
## Figure 5 plot
## -------------
pdf(file = "fig5.pdf", height = 6, width = 6.5)
par(mfrow = c(1,1), mar = c(4.1, 4.3, 2.1, 1.0))
colrep <- rep("black", length(test[x<.05]))
colrep[196] <- "red"
plot(x[x<.05], test[x<.05], cex = n^(1/100), pch = 16,
     ylim = c(0, max(test[x<.05])),
     main = "Partisan Bias of Simulated Plans",
     xlab = "% of Precincts Switched From Original District",
     ylab = "Partisan Bias towards Democrats",
     xaxt = "n",
     cex.lab = 1.6,
     cex.axis = 1.7,
     cex.main = 1.6,
     col = colrep
     )
axis(1, seq(0, 0.05, by = 0.01), c("0%", "1%", "2%", "3%", "4%", "5%"), cex.axis = 1.7)
abline(h = (1 - -1) * (storebias[1] - bias_min) / (bias_max - bias_min) + -1)
abline(h = 0, col = "red", lty = "dashed")
dev.off()

## -------------
## Figure 6 plot
## -------------
pdf(file = "fig6.pdf", height = 12, width = 12)
par(mfrow = c(2,2), mar = c(5.1, 4.3, 3.1, 0.2))
## CD 1
plot(
    density(
        na.omit(
            mapsub_1$dem_voteshare[mapsub_1$orig_map == mapsub_1$min_map]
        ), from = 0, to = 1
    ),
    xlim = c(0, 1),
    xlab = "Vote Share for Democratic Congressional Candidate",
    lwd = 2,
    main = "Distribution of Democratic Voteshare for District 1",
    cex.lab = 1.6,
    cex.axis = 1.7,
    cex.main = 1.6
)
lines(
    density(
        na.omit(
            mapsub_1$dem_voteshare[mapsub_1$orig_map == 1 & mapsub_1$min_map != 1]
        ), from = 0, to = 1
    ), col = "blue",
    lwd = 2,
    lty = "dashed"
)
lines(
    density(
        na.omit(
            mapsub_1$dem_voteshare[mapsub_1$orig_map != 1 & mapsub_1$min_map == 1]
        ), from = 0, to = 1
    ), col = "red",
    lwd = 2,
    lty = "dotdash"
)
legend("topleft", col = c("black", "blue", "red"),
       lwd = 2, legend = c("Not Swapped", "Swapped Out", "Swapped In"),
       lty = c(1, 2, 6),
       bty = "n", cex = 1.6)

plot(mapsub_1, col = mapsub_1$col, lwd = 1,
     main = "Congressional District 1",
    cex.lab = 1.6,
    cex.axis = 1.7,
    cex.main = 1.6)
legend("topleft", col = c("black", "blue", "red"),
       pt.bg = c("white", "blue", "red"),
       pch = 22,
       legend = c("Not Swapped", "Swapped Out", "Swapped In"),
       bty = "n", cex = 1.6)

## CD 11
plot(
    density(
        na.omit(
            mapsub_dist$dem_voteshare[mapsub_dist$orig_map == mapsub_dist$min_map]
        ), from = 0, to = 1
    ),
    ylim = c(0,6),
    xlim = c(0, 1),
    xlab = "Vote Share for Democratic Congressional Candidate",
    lwd = 2,
    main = "Distribution of Democratic Voteshare for District 3",
    cex.lab = 1.6,
    cex.axis = 1.6,
    cex.main = 1.6
)
lines(
    density(
        na.omit(
            mapsub_dist$dem_voteshare[mapsub_dist$orig_map == dist & mapsub_dist$min_map != dist]
        ), from = 0, to = 1
    ), col = "blue",
    lwd = 2,
    lty = "dashed"
)
lines(
    density(
        na.omit(
            mapsub_dist$dem_voteshare[mapsub_dist$orig_map != dist & mapsub_dist$min_map == dist]
        ), from = 0, to = 1
    ), col = "red",
    lwd = 2,
    lty = "dotdash"
)

plot(mapsub_dist, col = mapsub_dist$col, lwd = 1,
     main = "Congressional District 3",
    cex.lab = 1.5,
    cex.axis = 1.7,
    cex.main = 1.6)
dev.off()

