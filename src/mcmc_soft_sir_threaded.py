from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from mcmc_utils import *


# from mcmc_soft import accept_or_reject_proposal

def accept_or_reject_proposal(current_partition, proposed_partition,
                              G, BCP_current,
                              V_CP, R, q, lambda_param,
                              g_current, g_proposed, df):
    """
    A simple Metropolis-Hastings accept/reject step. A new sample is proposed based on the previous sample,
    then the proposed sample is either added to the chain or rejected based on the acceptance probability.

    This approximation is valid under the assumption that we rarely reject samples drawn in Step 3(b) for adjacency or
    shattering issues in our method "select_nonadjacent_components".

    Assumes symmetric proposals, so the acceptance ratio is g(π')/g(π) and the ratio of the probabilities of the
    Gibbs distribution for the two partitions for the population constraint.

    Parameters:
    - current_partition: The current partition π.
    - proposed_partition: The proposed new partition π'.
    - G: The graph of the precincts.
    - BCP_current: The number of boundary components in the current partition.
    - V_CP: The set of selected boundary components (not used here, but kept for consistency).
    - R: The number of counties to change in the proposed partition.
    - q: Probability threshold for turning on edges (not used directly here).
    - lambda_param: The lambda parameter for the zero-truncated Poisson distribution.
    - g_current: The unnormalized target probability g(π).
    - g_proposed: The unnormalized target probability g(π').

    Returns:
    - A partition (dict): Either the proposed_partition if accepted, or the current_partition if rejected.
    """
    # Precomputed g(π) and g(π') - target distribution ratios

    # If both g_current and g_proposed are zero, just reject to avoid division by zero
    if g_current == 0 and g_proposed == 0:
        return False
    # If g_current == 0 but g_proposed > 0, we can set ratio to something large; effectively always accept
    if g_current == 0 and g_proposed > 0:
        return True

    # |B(CP, π)| -> Precomputed from the main loop BCP_current = |B(CP, π)|
    # Compute |B(CP, π')|
    E_on_proposed = turn_on_edges(G, proposed_partition, q)
    boundary_components_proposed = find_boundary_connected_components(G, proposed_partition, E_on_proposed)
    BCP_proposed = len(boundary_components_proposed)

    # OPT Visualization before changes (optional; can slow down execution by a lot)
    # visualize_map_with_graph_and_geometry(G, E_on_proposed, boundary_components_proposed, V_CP, df, proposed_partition)

    # Compute |C(π, V_CP)|
    C_pi_VCP = compute_swendsen_wang_cut(G, current_partition, V_CP)

    # Compute |C(π', V_CP)| by counting how V_CP is structured under π'
    C_pi_prime_VCP = compute_swendsen_wang_cut(G, proposed_partition, V_CP)

    # Compute F(|B(CP, π)|) and F(|B(CP, π')|)
    # F is the truncated Poisson pmf for the chosen R given the number of boundary components
    F_current = truncated_poisson_pmf(R, lambda_param, BCP_current)
    F_proposed = truncated_poisson_pmf(R, lambda_param, BCP_proposed)

    # Handle cases where F_proposed = 0 (no valid R under π')
    # This would make the ratio infinite, but min(1, ...) caps it at 1.
    if F_proposed == 0:
        # If it's impossible to choose R from π' scenario, ratio becomes 0.
        ratio_F = 0.0
    else:
        ratio_F = F_current / F_proposed

    # Compute (|B(CP, π)| / |B(CP, π')|)^R
    # If BCP_proposed = 0, can't form any boundary components (unlikely, but check)
    if BCP_proposed == 0:
        boundary_ratio = 0.0
    else:
        boundary_ratio = (BCP_current / BCP_proposed) ** R

    # Compute ( (1-q)^{|C(π',V_CP)|} / (1-q)^{|C(π,V_CP)|} ) = (1-q)^{C_pi_prime_VCP - C_pi_VCP}
    pq_ratio = (1 - q) ** (C_pi_prime_VCP - C_pi_VCP)

    # Compute g(π')/g(π)
    if g_current == 0:
        if g_proposed > 0:
            g_ratio = float('inf')
        else:
            g_ratio = 1.0
    else:
        g_ratio = g_proposed / g_current

    # Combine all terms:
    # α = min(1, boundary_ratio * ratio_F * pq_ratio * g_ratio)
    MH_ratio = boundary_ratio * ratio_F * pq_ratio * g_ratio
    alpha = min(1.0, MH_ratio)

    # Accept or reject
    u = random.random()
    return u <= alpha


def g_func_cached(partition, populations, beta, ideal_pop):
    """
    Optimized Gibbs distribution calculation with cached ideal population.
    """
    district_pop = defaultdict(float)
    for node, dist in partition.items():
        district_pop[dist] += populations[node]

    # Compute deviation sum
    deviation_sum = sum(abs((pop / ideal_pop) - 1) for pop in district_pop.values())
    return np.exp(-beta * deviation_sum)


def generate_valid_sample(drawn_partitions, G, q, lambda_param, populations, beta, ideal_pop, df):
    """
    Helper function to generate a single valid sample using the basic algorithm steps.
    """
    # Start from a partition in drawn_partitions
    current_partition = random.choice(drawn_partitions)

    # Step 1-4: Basic algorithm steps
    E_on = turn_on_edges(G, current_partition, q)  # 1. Turn on edges
    boundary_components = find_boundary_connected_components(G, current_partition, E_on)  # 2. Find Boundary CPs
    BCP_current_len = len(boundary_components)  # Store for later use
    V_CP, R = select_nonadjacent_components(boundary_components, G, current_partition, lambda_param=lambda_param)  # 3
    proposed_partition = propose_swaps(current_partition, V_CP, G)  # 4. Propose swaps

    # Step 6: Acceptance check
    g_current = g_func_cached(current_partition, populations, beta, ideal_pop)
    g_proposed = g_func_cached(proposed_partition, populations, beta, ideal_pop)
    accepted = accept_or_reject_proposal(
        current_partition, proposed_partition, G, BCP_current_len, V_CP, R, q, lambda_param, g_current, g_proposed, df
    )
    if accepted:
        return proposed_partition.copy()
    return None


def generate_valid_samples_parallel(drawn_partitions, G, q, lambda_param, M, populations, beta, ideal_pop, df):
    """
    Generate exactly M valid samples using ThreadPoolExecutor.
    """
    valid_samples = []
    attempts = 0
    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), M * 2)) as executor:
        futures = [
            executor.submit(generate_valid_sample,
                            drawn_partitions, G, q,
                            lambda_param, populations, beta, ideal_pop,
                            df)
            for _ in range(M * 2)  # Oversampling for potential rejections in the acceptance step while multiprocessing
        ]

        # Process all submitted futures safely
        for future in as_completed(futures):
            attempts += 1  # Increment attempts for every completed task
            sample = future.result()
            if sample:
                valid_samples.append(sample)
            if len(valid_samples) >= M:
                # Cancel remaining futures to free resources
                for pending_future in futures:
                    pending_future.cancel()
                break  # Stop when M samples are collected

    return valid_samples, attempts


def filter_valid_samples(samples, populations, gdf, delta, compactness_threshold, compactness_constraint):
    """
    Filters samples based on population deviation and compactness constraints.
    """
    valid_samples = []
    for samp in samples:
        max_dev = max_population_deviation(samp, populations)
        if compactness_constraint:
            avg_compactness = compute_avg_compactness(samp, gdf)
            if max_dev <= delta and avg_compactness >= compactness_threshold:
                # Keep only samples that meet the population and compactness constraints
                valid_samples.append(samp)
        elif max_dev <= delta:
            # Keep only samples that meet the population constraint
            valid_samples.append(samp)
    return valid_samples


def run_mcmc_soft_with_sir(df, q=0.04, beta=30,
                           num_iterations=20,  # Number of outer iterations
                           M=10,  # M samples generated each iteration
                           S=5,  # S samples drawn from SIR
                           lambda_param=2,
                           delta=0.05,
                           compactness_threshold=0.22,
                           compactness_constraint=True):
    """
    Closer implementation to the algorithm described in the image of Algorithm 1.2 (soft constraint).

    Steps:
    - We have num_iterations "cycles".
    - In each iteration:
      1) Run the basic algorithm to generate M samples using MH acceptance probability with Gibbs.
      2) Reject samples that fail the population constraint (max population deviation > delta).
      3) From the remaining, perform SIR resampling with weights = 1/g_beta(π) to get S samples.
      4) Append these S samples to the global samples list.

    Arguments:
    - df: DataFrame with nodes and their attributes.
    - reward_w: dictionary of weights for the reward function.
    - q, beta, lambda_param: parameters as in the original code.
    - num_iterations: how many times we run the M-sample generation + filtering + SIR cycle.
    - M: how many samples we generate each iteration.
    - S: how many samples we draw after SIR.
    - delta: population constraint threshold.

    Returns:
    - samples: list of all final samples (S samples per iteration * num_iterations)
    - best_partition: best partition found so far.
    """
    # Prepare DataFrame
    df = df.reset_index(drop=True)
    df['node_id'] = df.index

    # Precompute populations and ideal population
    populations = df.set_index('node_id')['pop'].to_dict()
    num_districts = len(set(df['cd_2020']))
    ideal_pop = sum(populations.values()) / num_districts

    # Precompute current partition and create GeoDataFrame
    initial_partition = df.set_index('node_id')['cd_2020'].to_dict()
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    # Create a NetworkX graph
    G = nx.Graph()
    for i, row in df.iterrows():
        for neighbor in row['adj']:  # row['adj'] should be a list of node ids adjacent to i
            G.add_edge(i, neighbor)

    # Initialize the partition and store initial drawn partitions
    drawn_partitions = [initial_partition.copy()]  # Start from initial partition
    current_partition = initial_partition.copy()

    # Initialize best partition and reward
    best_partition = current_partition.copy()
    best_rep_bias = compute_partisan_bias(df, best_partition,
                                          dem_vote_col="pre_20_dem_bid",
                                          rep_vote_col="pre_20_rep_tru")
    best_efficiency_gap = compute_efficiency_gap(df, best_partition,
                                                 dem_vote_col="pre_20_dem_bid",
                                                 rep_vote_col="pre_20_rep_tru")

    # count iterations
    iteration = 0  # iteration counter (nb time steps actually performed in the chain)
    total_attempts = 0  # total attempts (samples) counter
    start_time = time.time()  # Start time
    samples = []  # List to store partition samples

    print("Starting Algorithm 1.2 (Soft constraint) ...")

    while iteration < num_iterations:
        # Step 1: Generate M samples using the basic algorithm.
        # Important: Do NOT filter by population/compactness constraint here, only use the acceptance probability.
        iteration_samples, attempts = generate_valid_samples_parallel(drawn_partitions, G, q,
                                                                      lambda_param, M,
                                                                      populations, beta,
                                                                      ideal_pop, df)

        # Now we have M samples accepted by the MH step.

        # Step 2: Reject samples failing the population constraint.
        # The paper states "For a specified population constraint δ, discard those samples."
        # Here we interpret the constraint: max_{ℓ} |(Sum_i∈Vℓ p_i)/p̃ - 1| > δ
        # This is basically the same as max_pop_dev > δ.
        valid_samples = filter_valid_samples(iteration_samples,
                                             populations, gdf, delta,
                                             compactness_threshold,
                                             compactness_constraint)
        if not valid_samples:
            print(
                f"Iteration {iteration:02}: No valid samples after applying population constraint. "
                f"Resampling M samples."
            )
            # If no valid samples, we might just continue with next iteration
            # or handle differently. For now, let's just continue.
            # No SIR is performed.
            continue

        # Step 3: SIR resampling from the valid samples using weights = 1/g_β(π)
        weights = np.array([1 / g_func_cached(samp, populations, beta, ideal_pop) for samp in valid_samples])
        weights /= weights.sum()
        chosen_indices = np.random.choice(len(valid_samples), size=S, p=weights, replace=True)
        drawn_partitions = [valid_samples[idx].copy() for idx in chosen_indices]

        # Compute metrics for all drawn partitions
        rep_biases = [compute_partisan_bias(df, p, dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru")
                      for p in drawn_partitions]
        efficiency_gaps = [
            compute_efficiency_gap(df, p, dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru")
            for p in drawn_partitions]
        max_pop_devs = [max_population_deviation(p, populations) for p in drawn_partitions]
        avg_compactnesses = [compute_compactness(gdf, p)[0] for p in drawn_partitions]

        # Find the best partition based on efficiency gap, rep bias, and population deviation
        if compactness_constraint:
            valid_best_indices = [i for i, (eg, rb, mpd, ac) in
                                  enumerate(zip(efficiency_gaps, rep_biases, max_pop_devs, avg_compactnesses))
                                  if abs(eg) <= abs(best_efficiency_gap)
                                  and abs(rb) <= abs(best_rep_bias)
                                  and mpd <= delta
                                  and ac >= compactness_threshold]
        else:
            valid_best_indices = [i for i, (eg, rb, mpd) in
                                  enumerate(zip(efficiency_gaps, rep_biases, max_pop_devs))
                                  if abs(eg) <= abs(best_efficiency_gap)
                                  and abs(rb) <= abs(best_rep_bias)
                                  and mpd <= delta]

        if valid_best_indices:
            # Select the best partition based on criteria
            best_idx = min(valid_best_indices, key=lambda i: (abs(efficiency_gaps[i]), abs(rep_biases[i])))
            best_partition = drawn_partitions[best_idx].copy()
            best_efficiency_gap = efficiency_gaps[best_idx]
            best_rep_bias = rep_biases[best_idx]
            print(
                f"Iteration {iteration:02} | "
                f"Curr Compact: {avg_compactnesses[best_idx]:.6f} | "
                f"Max Pop dev: {max_pop_devs[best_idx] * 100:.2f}% | "
                f"Efficiency Gap: {efficiency_gaps[best_idx]:.6f} | "
                f"Rep Bias: {rep_biases[best_idx]:.3f} New Best! "
            )
        # Append these S samples to global samples
        samples.extend(drawn_partitions)

        # Print iteration info
        avg_compactness = np.mean(avg_compactnesses)
        avg_pop_dev = np.mean(max_pop_devs)
        avg_efficiency_gap = np.mean(efficiency_gaps)
        avg_rep_bias = np.mean(rep_biases)
        print(
            f"Iteration {iteration:02} | "
            f"Avg Compact: {avg_compactness:.6f} | Avg Pop Dev: {avg_pop_dev:.6f} | "
            f"Avg Eff. Gap: {avg_efficiency_gap:.6f} | Avg Rep Bias: {avg_rep_bias:.3f}"
        )
        total_attempts += attempts
        iteration += 1

    print("\nEnd of Algorithm 1.2 (Soft constraint with SIR)")
    print("Total sampling time:", time.time() - start_time)
    print(
        f"Total attempts: {total_attempts} | Total samples: {len(samples)} | "
        f"Acceptance Probability: {len(samples) / total_attempts * 100:.2f}%")
    return samples, best_partition


# Define all hyperparameters for each state
ALL_HYPERPARAMS_DICT = {
    "IA": Hyperparameters(0.05, 9, 1000, 10, 5, 2, 0.10, 0.25, True),
    "PA": Hyperparameters(0.04, 30, 200, 10, 5, 10, 0.10, 0.05, False),
    "MA": Hyperparameters(0.10, 30, 100, 10, 5, 2, 0.08, 0.22, True),
    "MI": Hyperparameters(0.10, 30, 100, 10, 5, 5, 0.08, 0.22, False)
}


def main(state_abrv="PA", seed=6162):
    """
    Main function to run the redistricting algorithm with the specified parameters.

    These are all the parameters possible to tune for the algorithm. The hyperparameters are tuned for the state of
    Iowa (IA) by default.

    Parameters (or hyperparameters tuned for Iowa by default, refer to the markdown for more other states):
    - state_abrv: The state abbreviation to load the data for (default: "IA").
    - seed: Random seed for reproducing results (default: 6162).
    - q: Probability threshold for turning on edges in the graph. (higher => More edges turned on, more components)
    - beta: Inverse temperature for the Gibbs distribution. (Higher beta => More equal districts)
    - num_iterations: Number of iterations to run the algorithm. (more iterations => More samples in output)
    - lambda_param: Lambda parameter for zero-truncated Poisson distribution (higher => Change more counties at once)
    - max_pop_dev_threshold: Maximum population deviation from equal districts. (lower => More equal districts)
    - compactness_threshold: Minimum compactness threshold for districts. (higher => More compact districts)

    For smaller states like Iowa, the algorithm can be run with the default hyperparameters. For larger states, the
    hyperparameters may need to be adjusted to achieve better results.

    Outputs:
    - Prints the best reward and partisan metrics for the best partition.
    - Visualizes the best partition on a map.
    - save all the samples to a CSV file.
    """
    state_abrv = state_abrv.upper()
    params = ALL_HYPERPARAMS_DICT[state_abrv]

    # Print hyperparameters
    print(
        f"Running MCMC W/ HARD CONSTRAINTS for {STATE_ABBREVIATIONS[state_abrv]} with the following hyperparameters:\n")
    print(f"seed: {seed}")
    for field, value in params.__dict__.items():
        print(f"{field}: {value}")
    print("\n")

    # Run the algorithm with tuned hyperparameters to get all samples and
    # the best partition according to bias and efficiency gap
    random.seed(seed)
    df = prep_data(state_abrv)
    samples, best_partition = run_mcmc_soft_with_sir(
        df,
        q=params.q,
        beta=params.beta,
        num_iterations=params.num_iterations,
        M=params.M,
        S=params.S,
        lambda_param=params.lambda_param,
        delta=params.max_pop_dev_threshold,
        compactness_threshold=params.compactness_threshold,
        compactness_constraint=params.compactness_constraint
    )

    # Just for main:
    # Update the DataFrame with the best partition
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    # Save partitions to a file
    save_partitions_as_dataframe(samples, output_file=f"output/mcmc_soft_sir_partitions_{state_abrv}.csv")

    ##############################################################
    # Best run: calculating metrics for the best partition
    df = df.reset_index(drop=True)
    df['node_id'] = df.index
    best_district = np.array([best_partition[node] for node in range(len(df))])
    df['district'] = best_district
    populations = df.set_index('node_id')['pop'].to_dict()

    # Precincts changed
    if best_partition == df.set_index('node_id')['cd_2020'].to_dict():
        print("NO CHANGES made to the initial partition.")
    else:
        # Print the difference between the initial and final partition in pourcentage
        changed_precincts = calculate_percentage_changed(best_partition, df.set_index('node_id')['cd_2020'].to_dict())
        print(f"Percentage of precincts changed: {changed_precincts:.2f}%")

    rep_bias = compute_partisan_bias(df, best_partition, dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru")
    efficiency_gap = compute_efficiency_gap(df, best_partition, dem_vote_col="pre_20_dem_bid",
                                            rep_vote_col="pre_20_rep_tru")
    compactness_mean, compactness_std = compute_compactness(gdf, best_partition)
    pop_variance = compute_population_entropy(df, best_partition)
    # Print the partisan metrics for the best partition
    print()
    print("Republican bias:", round(rep_bias, 6))
    print("Efficiency gap:", round(efficiency_gap, 6))
    print("Compactness Mean:", round(compactness_mean, 6))
    print("Compactness std:", round(compactness_std, 6))
    print("Population entropy:", round(pop_variance, 6))
    print("Max population deviation:", round(max_population_deviation(best_partition, populations), 6))

    # Visualize the best partition
    metrics = {
        "total": [("pop", "Total Population"), ("vap", "Voting Age Population")],
        "mean": [],
        "ratio": [[("pre_20_dem_bid", "Biden"), ("pre_20_rep_tru", "Trump")]]
    }
    visualize_map_with_geometry(df,
                                geometry_col="geometry", district_id_col=best_partition,
                                state=STATE_ABBREVIATIONS[state_abrv], metrics=metrics)


if __name__ == "__main__":
    profile_run = True
    if profile_run:
        profiler = cProfile.Profile()
        profiler.runcall(main)
        stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
        stats.print_stats("mcmc_soft_sir_threaded.py")
        stats.dump_stats("profile_results.pstats")
    else:
        main()
