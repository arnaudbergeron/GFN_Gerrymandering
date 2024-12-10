library("redist"); library("igraph"); library("rgdal")

## Get distribution of population in NH
nh <- readOGR(path.expand("nh"), "nh")
pop_params <- fitdistr(round(nh@data$POP100), "negative binomial")
rep_params <- fitdistr(round(nh@data$PRES_REP08), "negative binomial")

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

testLat <- function(nrow, ncol, nsims, ndists){

    adjlist <- generateLat(nrow, ncol)
    repeat{
        data <- data.frame(
            pop = rnbinom(
                length(adjlist),
                size = pop_params$estimate[1],
                mu = pop_params$estimate[2]
            ),
            repub = rnbinom(
                length(adjlist),
                size = rep_params$estimate[1],
                mu = rep_params$estimate[2]
            )
        )
        if(all(data$pop > data$repub)){
            break
        }
    }
    allpart <- redist.enumerate(adjlist, ndists = ndists)

    ## Get segregation index
    npart <- length(allpart)
    allpart.mat <- matrix(unlist(allpart), ncol = length(allpart), byrow = FALSE)
    seg.rep.true <- data.frame(dissim = redist.segcalc(algout = allpart.mat,
                                   grouppop = data$repub,
                                   fullpop = data$pop))

    ## Run simulations
    sim.exact <- redist.mcmc(adjlist, data$pop, nsims = nsims,
                             ndists = ndists, exact_mh = 1)
    sim.appx <- redist.mcmc(adjlist, data$pop, nsims = nsims,
                            ndists = ndists, exact_mh = 0)

    ## Get dissimilarity
    seg.sim <- data.frame(exact = redist.segcalc(algout = sim.exact,
                              grouppop = data$repub, fullpop = data$pop),
                          appx = redist.segcalc(algout = sim.appx,
                              grouppop = data$repub, fullpop = data$pop))
    seg.true <- data.frame(dissim = redist.segcalc(algout = allpart.mat,
                               grouppop = data$repub, fullpop = data$pop))

    out <- vector(mode = "list")
    out$sim <- seg.sim
    out$true <- seg.true
    out$npart <- length(allpart)

    return(out)

}

