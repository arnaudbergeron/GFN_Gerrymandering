###############################################
## Figure 3 - new metropolis hastings ratios ##
## RSG simulations ############################
###############################################
rm(list = ls())
library("redist")

set.seed(100)
nsims <- 10000

###################
## No constraint ##
###################
data(algdat.pfull)
rsg <- matrix(NA, length(algdat.pfull$adjlist), nsims)
for(i in 1:nsims){
    rsg[,i] <- redist.rsg(algdat.pfull$adjlist,
                           algdat.pfull$precinct.data$pop,
                           ndists = 3, thresh = 100, verbose = FALSE)$district_membership
}
rsg.full <- data.frame(redist.segcalc(algout = rsg,
                                      grouppop = algdat.pfull$precinct.data$repvote,
                                      fullpop = algdat.pfull$precinct.data$pop))
names(rsg.full) <- "dissim"
print("Full simulations done")

####################
## 20% constraint ##
####################
data(algdat.p20)
rsg <- matrix(NA, length(algdat.pfull$adjlist), nsims)
for(i in 1:nsims){
    rsg[,i] <- redist.rsg(algdat.pfull$adjlist,
                           algdat.pfull$precinct.data$pop,
                           ndists = 3, thresh = .2, verbose = FALSE)$district_membership
}
rsg.20 <- data.frame(redist.segcalc(algout = rsg,
                                      grouppop = algdat.pfull$precinct.data$repvote,
                                      fullpop = algdat.pfull$precinct.data$pop))
names(rsg.20) <- "dissim"
print("20% simulations done")

###############################
## 10% population constraint ##
###############################
data(algdat.p10)
rsg <- matrix(NA, length(algdat.pfull$adjlist), nsims)
for(i in 1:nsims){
    rsg[,i] <- redist.rsg(algdat.pfull$adjlist,
                           algdat.pfull$precinct.data$pop,
                           ndists = 3, thresh = .1, verbose = FALSE)$district_membership
}
rsg.10 <- data.frame(redist.segcalc(algout = rsg,
                                      grouppop = algdat.p10$precinct.data$repvote,
                                      fullpop = algdat.p10$precinct.data$pop))
names(rsg.10) <- "dissim"
print("10% simulations done")

## Save data
save(rsg.full, rsg.20, rsg.10, file = "fig3_rsgsims.RData")
