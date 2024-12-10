library(tidyverse)
library(doMC)
library(parallel)
library(ggthemes)
library(scales)
library(MASS)
library(rgdal)
library(Cairo)
library(latex2exp)

## Install branch version of redist that counts boundary units
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

## Loop over grid sizes
params <- expand.grid(latsize = 4:12,
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
        ndists = ndists
    )

    df_out <- data.frame(bpart = mcmc_out$boundary_partitions,
                          ratio_approx = mcmc_out$boundaryratio_apprx,
                          ratio_exact = mcmc_out$boundaryratio_exact) %>%
        mutate(map = paste0(lat, "x", lat, " Lattice"),
               ndist = ndists)
    return(df_out)

}
df_plot_wide <- df_plot %>%
    mutate(relative_error = abs(ratio_approx - ratio_exact) / ratio_exact,
           ratio_of_ratios = ratio_approx / ratio_exact)
df_plot_wide$map <- as.factor(df_plot_wide$map)
df_plot_wide$map <- factor(df_plot_wide$map, levels = levels(df_plot_wide$map)[c(4:9,1:3)])

df_plot_wide %>% filter(ndist == 4) %>%
    ggplot(aes(abs(ratio_of_ratios))) +
    geom_density(alpha = 0, lwd = 1) + facet_wrap(~ map) +
    labs(x = TeX("$|\\alpha_{weak} / \\alpha_{strong}|"),
         y = "Density") +
    geom_vline(aes(xintercept = 1), lty = "dashed") +
    scale_x_log10() +
    ben_theme() +
    ggsave("fig_s8.pdf", height = 6, width = 6)

## Reinstall CRAN version of package
install.packages("redist")
