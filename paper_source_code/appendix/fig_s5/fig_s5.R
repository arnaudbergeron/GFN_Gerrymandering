## ---------------------------------------------------
## New Hampshire validation - multiple vs single swaps
## ---------------------------------------------------
library(doMC)
library(rgdal)
library(redist)
library(coda)
library(tidyverse)
library(gridExtra)
library(grid)

## ------------------------------
## Single swaps vs multiple swaps
## ------------------------------
set.seed(08540, kind = "L'Ecuyer-CMRG")
map <- readOGR(path.expand("nh"), "nh")
adjlist <- spdep::poly2nb(map, queen = FALSE)
adjlist <- lapply(adjlist, function(x){x-1})
startval <- lapply(1:4, function(x){
    redist.rsg(adjlist, ndists = 2, population = map@data$POP100, thres = .05)$district_membership
})

nsims <- 20000
registerDoMC(4)
seg_out <- foreach(i = 1:4) %dopar% {

    ## ----------------------
    ## With parity constraint
    ## ----------------------
    
    ## Single swaps
    out_ss <- redist.mcmc(
        adjobj = map,
        popvec = map@data$POP100,
        nsims = nsims,
        initcds = startval[[i]],
        constraint = "population",
        beta = -27,
        adapt_lambda = FALSE
    )
    
    ## Multiple swaps
    out_ms <- redist.mcmc(
        adjobj = map,
        popvec = map@data$POP100,
        nsims = nsims,
        initcds = startval[[i]],
        constraint = "population",
        beta = -27,
        adapt_lambda = TRUE
    )

    ## Get segregation index
    seg_ss <- redist.segcalc(
        out_ss,
        grouppop = map@data$PRES_REP08,
        map@data$POP100
    )[which(out_ss$distance_parity <= .01)]
    seg_ms <- redist.segcalc(
        out_ms,
        grouppop = map@data$PRES_REP08,
        map@data$POP100
    )[which(out_ms$distance_parity <= .01)]

    out <- list(ss = seg_ss, ms = seg_ms)
    return(out)
    
}

## ----------------------------
## Format for parity constraint
## ----------------------------
ss_list <- sapply(seg_out, function(x) x$ss)
ms_list <- sapply(seg_out, function(x) x$ms)
min.len <- min(unlist(lapply(ss_list, length)),
               unlist(lapply(ms_list, length)))
logit.mcmc <- function(x, len = min.len){mcmc(log(x[1:len] / (1 - x[1:len])))}

ss_list <- mcmc.list(lapply(ss_list, logit.mcmc))
ms_list <- mcmc.list(lapply(ms_list, logit.mcmc))

ss_ac <- acf(ss_list[[1]], plot = FALSE, lag.max = 50)
ms_ac <- acf(ms_list[[1]], plot = FALSE, lag.max = 50)

## Get data
gd_ss <- gelman.plot(ss_list)
gd_ms <- gelman.plot(ms_list)

x <- as.numeric(rownames(gd_ss$shrink))
pointest_ss <- gd_ss$shrink[1:(length(gd_ss$shrink)/2)]
ci_ss <- gd_ss$shrink[(length(gd_ss$shrink)/2 + 1):length(gd_ss$shrink)]
pointest_ms <- gd_ms$shrink[1:(length(gd_ms$shrink)/2)]
ci_ms <- gd_ms$shrink[(length(gd_ms$shrink)/2 + 1):length(gd_ms$shrink)]

## Create plot object
df_plot <- data.frame(
    x = rep(x, 4),
    est = c(pointest_ss, ci_ss, pointest_ms, ci_ms),
    swaps = c(rep("Single Swaps", length(x)*2), rep("Multiple Swaps", length(x)*2)),
    esttype = c(rep("Median", length(x)), rep("97.5%", length(x)),
                rep("Median", length(x)), rep("97.5%", length(x))),
    parity = "Constrained Simulations (1%)"
)
## ---------
## Plot data
## ---------
df_plot$swaps <- factor(df_plot$swaps, levels = c("Single Swaps", "Multiple Swaps"))

gr <- ggplot(df_plot, aes(x, est, colour = esttype, lty = esttype)) +
    geom_line() +
    facet_grid(~ swaps) + 
    geom_hline(aes(yintercept = 1)) + 
    labs(x = "Last Iteration in Chain", y = "Potential Scale Reduction Factor\n",
         title = "Convergence Diagnostics for Single versus Multiple Swaps on NH Map") +
    scale_colour_manual(values = c("red", "black")) +
    scale_linetype_manual(values = c("dashed", "solid")) + 
    theme_classic() +
    coord_cartesian(ylim = c(.9, 4)) + 
    theme(legend.position = c(.925, .85),
          legend.title = element_blank(),
          plot.title = element_text(size = 18, hjust = 0.5),
          strip.background = element_blank(),
          panel.border = element_rect(colour = "black", fill = NA),
          strip.text.x = element_text(size = 12))

## --------------------
## Autocorrelation plot
## --------------------
df_plot <- data.frame(
    lag = c(ss_ac$lag[,,1], ms_ac$lag[,,1]),
    ac  = c(ss_ac$acf[,,1], ms_ac$acf[,,1]),
    ymin = 0,
    swaps = c(rep("Single Swaps", nrow(ss_ac$lag)),
                  rep("Multiple Swaps", nrow(ms_ac$lag)))
)
df_plot$swaps <- factor(df_plot$swaps, levels = c("Single Swaps", "Multiple Swaps"))

ac <- ggplot(df_plot, aes(x = lag, xend = lag, y = ymin, yend = ac)) +
    geom_segment() + facet_grid(~swaps) + 
    ylim(-1, 1) +
    labs(x = "Lag", y = "Autocorrelation") + 
    theme_classic() +
    theme(legend.position = c(.9, .9),
          legend.title = element_blank(),
          plot.title = element_text(hjust = 0.5),
          strip.background = element_blank(),
          panel.border = element_rect(colour = "black", fill = NA),
          strip.text.x = element_blank())

## Put them together
pdf(file = "fig_s5.pdf", height = 8, width = 9)
grid.arrange(arrangeGrob(gr, left = textGrob("Gelman-Rubin Diagnostic",
                                             rot = 90,
                                             gp = gpar(fontsize = 14))),
             arrangeGrob(ac, left = textGrob("Autocorrelation of a Chain",
                                             hjust = .4, rot = 90,
                                             gp = gpar(fontsize = 14))))
dev.off()

