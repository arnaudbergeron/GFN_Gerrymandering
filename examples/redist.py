import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------
# Placeholder functions to mimic redist
# -------------------------------------------
def redist_mcmc(adjobj, popvec, ndists, nsims, beta, lambd, constraint, temper, betaweights):
    """
    Placeholder for the redist.mcmc function.
    In practice, you'd need a proper MCMC simulation
    that generates partitions given constraints.
    """
    # For illustration, return a dictionary similar in structure to the R output:
    return {
        'distance_parity': np.random.rand(nsims),  # Mock data
        'beta_sequence': np.full(nsims, beta),
        'constraint_pop': np.random.rand(nsims),
        'partitions': np.random.randint(low=1, high=ndists + 1, size=(len(popvec), nsims))
    }


def redist_segcalc(algout, grouppop, fullpop):
    """
    Placeholder for the redist.segcalc function.
    The segregation/dissimilarity index would be computed here.
    """
    # Simple mock calculation of a dissimilarity index:
    # For illustration, let's say dissimilarity index = mean(|grouppop - fullpop|)/mean(fullpop)
    # applied per simulation (column).
    n_sims = algout['partitions'].shape[1]
    dissim = np.zeros(n_sims)
    for i in range(n_sims):
        # Partition assignment for this simulation
        partition = algout['partitions'][:, i]
        # A trivial mock calculation: difference of group proportion in each district
        # Normally, you'd calculate an index per district, then average.
        dissim[i] = np.mean(np.abs(grouppop - fullpop)) / np.mean(fullpop)
    return dissim


def ipw(x, beta, pop):
    """
    Importance sampling reweighting function.
    x: output from redist_mcmc or a similar structure
    beta: scalar
    pop: population threshold
    """
    distance_parity = x['distance_parity']
    beta_seq = x['beta_sequence']
    constraint_pop = x['constraint_pop']

    # Indices matching population criteria and beta
    ind_pop = np.where(distance_parity <= pop)[0]
    ind_beta = np.where(beta_seq == beta)[0]
    inds = np.intersect1d(ind_pop, ind_beta)

    psi = constraint_pop[inds]
    w = 1 / np.exp(beta * psi)  # weights
    w = w / np.sum(w)  # normalize weights for sampling

    # Resample partitions with replacement according to weights
    resampled_inds = np.random.choice(inds, size=len(inds), replace=True, p=w)
    x['partitions'] = x['partitions'][:, resampled_inds]

    return x


# -------------------------------------------
# Mock data in place of algdat.p20, algdat.p10
# -------------------------------------------
# In reality, you'd load your data and define:
# - adjlist (adjacency), popvec (populations)
# - precinct.data with 'pop', 'repvote'
# - segregation.index$repdiss from your dataset

# Mock data setup
num_precincts = 100
pop = np.random.randint(100, 1000, size=num_precincts)
repvote = np.random.randint(0, 500, size=num_precincts)
true_rep_diss = np.random.uniform(0, 0.3, size=1000)  # true distribution mock

# -------------------------------------------
# Run simulated tempering algorithm - 20%
# -------------------------------------------
np.random.seed(194115)
nsims = 10000
betaweights = [2 ** i for i in range(1, 11)]

mcmc_out_st_20 = redist_mcmc(
    adjobj=None,  # placeholder
    popvec=pop,
    ndists=3,
    nsims=nsims,
    beta=-5.4,
    lambd=2,
    constraint="population",
    temper="simulated",
    betaweights=betaweights
)
mcmc_out_st_20 = ipw(mcmc_out_st_20, beta=-5.4, pop=0.2)
rep_seg_st_20 = redist_segcalc(algout=mcmc_out_st_20, grouppop=repvote, fullpop=pop)

# -------------------------------------------
# Run simulated tempering algorithm - 10%
# -------------------------------------------
pop_10 = np.random.randint(100, 1000, size=num_precincts)
repvote_10 = np.random.randint(0, 500, size=num_precincts)
true_rep_diss_10 = np.random.uniform(0, 0.3, size=1000)  # mock

betaweights = [2 ** i for i in range(1, 11)]
mcmc_out_st_10 = redist_mcmc(
    adjobj=None,
    popvec=pop_10,
    ndists=3,
    nsims=nsims,
    beta=-9,
    lambd=2,
    constraint="population",
    temper="simulated",
    betaweights=betaweights
)
mcmc_out_st_10 = ipw(mcmc_out_st_10, beta=-9, pop=0.1)
rep_seg_st_10 = redist_segcalc(algout=mcmc_out_st_10, grouppop=repvote_10, fullpop=pop_10)

# -------------------------------------------
# Plot results - analogous to the R code
# -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# 20% constraint
axes[0].hist(true_rep_diss, bins=30, density=True, alpha=0.5, color="grey", label="True Distribution")
axes[0].hist(rep_seg_st_20, bins=30, density=True, histtype='step', color="black", label="Algorithm S1")
axes[0].set_xlim(0, 0.3)
axes[0].set_ylim(0, 55)
axes[0].set_xlabel("Republican Dissimilarity Index")
axes[0].set_ylabel("Density")
axes[0].set_title("Constrained Simulations (20%)")
axes[0].legend()

# 10% constraint
axes[1].hist(true_rep_diss_10, bins=30, density=True, alpha=0.5, color="grey", label="True Distribution")
axes[1].hist(rep_seg_st_10, bins=30, density=True, histtype='step', color="black", label="Algorithm S1")
axes[1].set_xlim(0, 0.3)
axes[1].set_ylim(0, 55)
axes[1].set_xlabel("Republican Dissimilarity Index")
axes[1].set_title("Constrained Simulations (10%)")

plt.tight_layout()
plt.show()