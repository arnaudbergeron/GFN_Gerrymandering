library(tidyverse)
library(doMC)
library(parallel)
library(ggthemes)
library(scales)
library(MASS)
library(rgdal)
library(Cairo)
library(latex2exp)

## Install branch version of redist that does boundary shatter check
system("R CMD INSTALL redist_1.3-3.tar.gz")
library(redist)
data(algdat.pfull)

ben_theme <- function(){
    theme_classic() +
        theme(panel.background = element_blank(),
              panel.grid.major = element_blank(), 
              panel.grid.minor = element_blank(),
              axis.line = element_line(colour = "black"),
              panel.border = element_rect(colour = "black", fill = NA, size = 1),
              strip.background = element_blank(),
              legend.position = "bottom", legend.title = element_blank(),
              plot.title = element_text(hjust = 0.5))
}

generateLat <- function(nrow, ncol){
    myLat <- list()
    for(i in 1:nrow){
        for(j in 1:ncol){
            index <- ncol*(i-1) + j
            edges <- NULL
            if(i > 1){
                edges <- c(edges, index - ncol)
            }
            if(j > 1){
                edges <- c(edges, index -1)
            }
            if(j < ncol){
                edges <- c(edges, index +1)
            }			
            if(i < nrow){
                edges <- c(edges, index + ncol)
            }
            
            myLat[[index]] = edges - 1
        }
    }
    myLat
}

## --------------------------------
## Check rejection due to adjacency
## --------------------------------
## Loop over grid sizes
params <- expand.grid(latsize = 4:15,
                      ndists = 2:4)
registerDoMC(detectCores())
df_plot <- foreach(i = 1:nrow(params), .combine = "rbind") %dopar% {

    ## Parameters
    lat <- params$latsize[i]
    ndists <- params$ndists[i]
    
    ## Run alg
    pop <- rnorm(lat*lat, 100, 20)
    pop[pop < 0] <- 1
    mcmc_out <- redist.mcmc(
        adjobj = generateLat(lat, lat),
        popvec = pop,
        nsims = 5000,
        ndists = ndists,
        lambda = 1
    )

    ## Store output
    df_out <- data.frame(
        share_rej_adj = sum(mcmc_out$adjacent_count > 0) / length(mcmc_out$adjacent_count),
        ndists = ndists,
        lat = lat
    )

    return(df_out)

}

df_plot %>% filter(ndists == 4) %>%
    mutate(lat = paste0(lat, "x", lat),
           lat = as.factor(lat),
           lat = factor(lat, levels = levels(lat)[c(7:12, 1:6)])) %>%
    ggplot(aes(lat, share_rej_adj, group = 1)) +
    geom_point(size = 3) + geom_line() +
    ylim(0, NA) + 
    labs(x = "Lattice Size", y = "Share of Iterations with Rejected Adjacent Swap") +
    ben_theme() +
    ggsave(filename = "fig_s9.pdf", height = 4, width = 6)

